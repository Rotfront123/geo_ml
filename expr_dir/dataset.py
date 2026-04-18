import json
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rioxarray
import torch
from rasterio import features
from torch.utils.data import Dataset

CLASS_MAPPING = {
    "курганы_целые": 1,
    "курганы_поврежденные": 2,
    "фортификации": 3,
    "городища": 4,
    "архитектуры": 5,
    "_FindsPoints": 6,
    "_ObjectPoly": 7,
}


class ArchaeologyDataset(Dataset):
    """Датасет с загрузкой данных на лету и нарезанием патчей."""

    def __init__(
        self,
        root_dir,
        transform=None,
        split="train",
        valid_regions=None,
        patch_size=512,
        patches_per_image=10,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.cache = {}
        self.cache_size = 3

        print(f"\n🔍 Scanning {root_dir} for regions...")

        # Собираем все подпапки первого уровня
        self.region_paths = []

        for region_path in self.root_dir.iterdir():
            if not region_path.is_dir():
                continue

            region_name = region_path.name

            # Проверяем наличие UTM.json (может быть в подпапках)
            utm_exists = (region_path / "UTM.json").exists() or len(
                list(region_path.glob("**/UTM.json"))
            ) > 0

            # Ищем tiff файлы
            tif_files = self._find_tiff_files(region_path)
            has_tiff = len(tif_files) > 0

            # Ищем geojson файлы
            geojson_files = self._find_geojson_files(region_path)
            has_geojson = len(geojson_files) > 0

            status = "✅" if (utm_exists and has_tiff and has_geojson) else "❌"
            print(
                f"  {status} {region_name}: UTM={utm_exists}, TIFF={has_tiff}, GEOJSON={has_geojson}"
            )

            if utm_exists and has_tiff and has_geojson:
                # Разделение на train/val
                if valid_regions and len(valid_regions) > 0:
                    if split == "train" and region_name not in valid_regions:
                        self.region_paths.append(region_path)
                    elif split == "val" and region_name in valid_regions:
                        self.region_paths.append(region_path)
                else:
                    if split == "train":
                        self.region_paths.append(region_path)

        print(f"\n✅ Found {len(self.region_paths)} valid regions for {split}")

    def __len__(self):
        # Каждый регион дает patches_per_image патчей
        return len(self.region_paths) * self.patches_per_image

    def _find_tiff_files(self, region_path):
        """Гибкий поиск tiff файлов."""
        tiff_files = []

        # Ищем во всех подпапках
        patterns = [
            "**/*.tiff",
            "**/*.tif",
            "**/*.TIF",
            "**/*.TIFF",
            "*SpOr*/**/*.tiff",
            "*SpOR*/**/*.tiff",
            "*Or*/**/*.tiff",
            "*спутник*/**/*.tiff",
            "*спутник*/**/*.tif",
        ]

        for pattern in patterns:
            tiff_files.extend(region_path.glob(pattern))

        # Убираем дубликаты
        tiff_files = list(set(tiff_files))

        return tiff_files

    def _find_geojson_files(self, region_path):
        """Гибкий поиск geojson файлов."""
        patterns = [
            "**/*.geojson",
            "*разметка*/**/*.geojson",
            "*_разметка/**/*.geojson",
        ]

        geojson_files = []
        for pattern in patterns:
            geojson_files.extend(region_path.glob(pattern))

        return list(set(geojson_files))

    def _load_region_data(self, region_idx: int):
        """Загружает данные региона."""
        if region_idx in self.cache:
            return self.cache[region_idx]

        region_path = self.region_paths[region_idx]
        region_name = region_path.name
        print(f"\n📂 Loading region: {region_name}")

        # Загружаем UTM.json
        utm_path = region_path / "UTM.json"
        if not utm_path.exists():
            # Пробуем найти UTM.json в подпапках
            utm_candidates = list(region_path.glob("**/UTM.json"))
            if utm_candidates:
                utm_path = utm_candidates[0]
            else:
                raise FileNotFoundError(f"UTM.json not found in {region_path}")

        with open(utm_path, "r") as f:
            utm_data = json.load(f)

        # Получаем CRS
        target_crs = utm_data.get("crs", "")
        if "::" in target_crs:
            target_crs = target_crs.split("::")[-1]
        if not target_crs.startswith("EPSG:"):
            target_crs = f"EPSG:{target_crs}"

        print(f"  CRS: {target_crs}")

        # Ищем tiff файлы
        tif_files = self._find_tiff_files(region_path)
        if not tif_files:
            raise FileNotFoundError(f"No tiff files found in {region_path}")

        print(f"  Found {len(tif_files)} tiff files")

        # Берем первый tiff
        example = tif_files[0]
        print(f"  Using: {example.name}")

        # Загружаем растр
        ref_ds = rioxarray.open_rasterio(example)
        ref_ds = ref_ds.rio.reproject(target_crs)

        rgb_image = ref_ds.values.transpose(1, 2, 0)
        if rgb_image.shape[2] > 3:
            rgb_image = rgb_image[:, :, :3]

        # Нормализуем в uint8
        if rgb_image.dtype != np.uint8:
            rgb_min, rgb_max = rgb_image.min(), rgb_image.max()
            if rgb_max > rgb_min:
                rgb_image = ((rgb_image - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
            else:
                rgb_image = np.zeros_like(rgb_image, dtype=np.uint8)

        print(f"  Image shape: {rgb_image.shape}")

        # Создаем маску
        geojson_files = self._find_geojson_files(region_path)
        print(f"  Found {len(geojson_files)} geojson files")

        final_mask = np.zeros((ref_ds.rio.height, ref_ds.rio.width), dtype=np.uint8)

        # Определяем "тип" tiff файла (SpOr или Or)
        tif_name_lower = example.stem.lower()
        tif_has_spor = "spor" in tif_name_lower
        tif_has_or = "or" in tif_name_lower and "spor" not in tif_name_lower

        objects_rasterized = 0

        for g_file in geojson_files:
            g_name_lower = g_file.stem.lower()

            # Более гибкое сопоставление geojson и tiff
            should_use = False

            if tif_has_spor and "spor" in g_name_lower:
                should_use = True
            elif tif_has_or and "spor" not in g_name_lower and "or" in g_name_lower:
                should_use = True
            elif not tif_has_spor and not tif_has_or:
                # Если тип не определен, используем все geojson
                should_use = True

            if should_use:
                class_id = self._get_class_id(g_file.name)
                if class_id > 0:
                    try:
                        gdf = gpd.read_file(g_file)
                        gdf = gdf.to_crs(target_crs)

                        # Фильтруем валидные геометрии
                        valid_geometries = []
                        for geom in gdf.geometry:
                            if geom is not None and not geom.is_empty and geom.is_valid:
                                valid_geometries.append(geom)
                            else:
                                print(f"    ⚠️ Skipping invalid geometry in {g_file.name}")

                        if valid_geometries:
                            file_mask = features.rasterize(
                                [(shape, class_id) for shape in valid_geometries],
                                out_shape=(ref_ds.rio.height, ref_ds.rio.width),
                                transform=ref_ds.rio.transform(),
                                fill=0,
                                dtype=np.uint8,
                            )
                            final_mask = np.maximum(final_mask, file_mask)
                            objects_rasterized += len(valid_geometries)

                    except Exception as e:
                        print(f"    ⚠️ Error processing {g_file.name}: {e}")

        print(f"  Rasterized {objects_rasterized} objects")

        # Проверяем, что маска не пустая
        unique_classes = np.unique(final_mask)
        print(f"  Classes in mask: {unique_classes}")

        # Сохраняем в кэш
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[region_idx] = (rgb_image, final_mask)
        return rgb_image, final_mask

    def _load_region_data(self, region_idx: int):
        """Загружает данные региона."""
        if region_idx in self.cache:
            return self.cache[region_idx]

        region_path = self.region_paths[region_idx]
        region_name = region_path.name
        print(f"\n📂 Loading region: {region_name}")

        # Загружаем UTM.json
        utm_path = region_path / "UTM.json"
        if not utm_path.exists():
            utm_candidates = list(region_path.glob("**/UTM.json"))
            if utm_candidates:
                utm_path = utm_candidates[0]
            else:
                raise FileNotFoundError(f"UTM.json not found in {region_path}")

        with open(utm_path, "r") as f:
            utm_data = json.load(f)

        # Получаем CRS
        target_crs = utm_data.get("crs", "")
        if "::" in target_crs:
            target_crs = target_crs.split("::")[-1]
        if not target_crs.startswith("EPSG:"):
            target_crs = f"EPSG:{target_crs}"

        # Ищем tiff файлы
        tif_files = self._find_tiff_files(region_path)
        if not tif_files:
            raise FileNotFoundError(f"No tiff files in {region_path}")

        example = tif_files[0]
        print(f"  Using tiff: {example.name}")

        # Загружаем растр
        ref_ds = rioxarray.open_rasterio(example)

        # 🔧 Пробуем reproject, если не получается - оставляем как есть
        try:
            ref_ds = ref_ds.rio.reproject(target_crs)
            print(f"  Reprojected to {target_crs}")
        except Exception as e:
            print(f"  ⚠️ Cannot reproject: {e}")
            print(f"  Keeping original CRS: {ref_ds.rio.crs}")
            target_crs = ref_ds.rio.crs  # Обновляем target_crs для geojson

        rgb_image = ref_ds.values.transpose(1, 2, 0)
        if len(rgb_image.shape) == 2:
            rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=2)
        elif rgb_image.shape[2] == 1:
            rgb_image = np.repeat(rgb_image, 3, axis=2)
        elif rgb_image.shape[2] > 3:
            rgb_image = rgb_image[:, :, :3]
            if rgb_image.shape[2] > 3:
                rgb_image = rgb_image[:, :, :3]

        # 🔧 Безопасная нормализация
        if rgb_image.dtype != np.uint8:
            rgb_image_float = rgb_image.astype(np.float64)
            rgb_min = rgb_image_float.min()
            rgb_max = rgb_image_float.max()
            if rgb_max > rgb_min:
                rgb_normalized = (rgb_image_float - rgb_min) / (rgb_max - rgb_min)
                rgb_image = (rgb_normalized * 255).astype(np.uint8)
            else:
                rgb_image = np.zeros_like(rgb_image, dtype=np.uint8)

        print(f"  Image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

        # Создаем маску
        geojson_files = self._find_geojson_files(region_path)
        final_mask = np.zeros((ref_ds.rio.height, ref_ds.rio.width), dtype=np.uint8)

        tif_name_lower = example.stem.lower()
        tif_has_spor = "spor" in tif_name_lower

        objects_rasterized = 0

        for g_file in geojson_files:
            g_name_lower = g_file.stem.lower()

            # Гибкое сопоставление
            should_use = False
            if tif_has_spor and "spor" in g_name_lower:
                should_use = True
            elif not tif_has_spor:
                should_use = True

            if should_use:
                class_id = self._get_class_id(g_file.name)
                if class_id > 0:
                    try:
                        gdf = gpd.read_file(g_file)

                        # Пробуем преобразовать CRS
                        try:
                            if gdf.crs is not None and target_crs is not None:
                                if str(gdf.crs).lower() != str(target_crs).lower():
                                    gdf = gdf.to_crs(target_crs)
                        except Exception:
                            pass  # Не можем преобразовать - оставляем как есть

                        # 🔧 Фильтруем валидные геометрии
                        valid_geometries = []
                        for geom in gdf.geometry:
                            if geom is not None and not geom.is_empty:
                                if hasattr(geom, "is_valid") and not geom.is_valid:
                                    try:
                                        geom = geom.buffer(0)  # Пытаемся исправить
                                    except:
                                        continue
                                if geom is not None and not geom.is_empty:
                                    valid_geometries.append(geom)

                        if valid_geometries:
                            file_mask = features.rasterize(
                                [(shape, class_id) for shape in valid_geometries],
                                out_shape=(ref_ds.rio.height, ref_ds.rio.width),
                                transform=ref_ds.rio.transform(),
                                fill=0,
                                dtype=np.uint8,
                            )
                            final_mask = np.maximum(final_mask, file_mask)
                            objects_rasterized += len(valid_geometries)

                    except Exception as e:
                        print(f"    ⚠️ Error processing {g_file.name}: {e}")
                        continue

        print(f"  Rasterized {objects_rasterized} objects")

        # Кэшируем
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[region_idx] = (rgb_image, final_mask)
        return rgb_image, final_mask

    def _get_class_id(self, file_name: str) -> int:
        """Определяет класс по имени файла."""
        for keyword, class_id in CLASS_MAPPING.items():
            if keyword in file_name:
                return class_id
        return 0

    def _extract_patch(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        """Извлекает случайный патч. ГАРАНТИРОВАННО возвращает patch_size x patch_size."""
        h, w = image.shape[:2]

        # 🔧 Если изображение меньше patch_size - ВСЕГДА паддим до patch_size
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            # Pad для изображения
            if len(image.shape) == 3:
                image = np.pad(
                    image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
                )
            else:
                # Для grayscale
                image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

            # Pad для маски
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

            h, w = image.shape[:2]

        # 🔧 Случайные координаты (гарантированно h >= patch_size, w >= patch_size)
        y = np.random.randint(0, h - self.patch_size + 1)
        x = np.random.randint(0, w - self.patch_size + 1)

        # Вырезаем патч
        patch_image = image[y : y + self.patch_size, x : x + self.patch_size].copy()
        patch_mask = mask[y : y + self.patch_size, x : x + self.patch_size].copy()

        # 🔧 ПРОВЕРКА: убеждаемся что размер правильный
        assert patch_image.shape[0] == self.patch_size, f"Wrong height: {patch_image.shape[0]}"
        assert patch_image.shape[1] == self.patch_size, f"Wrong width: {patch_image.shape[1]}"
        assert patch_mask.shape[0] == self.patch_size, f"Wrong mask height: {patch_mask.shape[0]}"
        assert patch_mask.shape[1] == self.patch_size, f"Wrong mask width: {patch_mask.shape[1]}"

        return patch_image, patch_mask

    def __getitem__(self, idx: int):
        region_idx = idx // self.patches_per_image

        # Загружаем данные региона
        image, mask = self._load_region_data(region_idx)

        # 🔧 ВАЖНО: Вырезаем патч ДО трансформаций
        patch_image, patch_mask = self._extract_patch(image, mask)

        # 🔧 Применяем трансформации
        if self.transform is not None:
            try:
                augmented = self.transform(image=patch_image, mask=patch_mask)
                patch_image = augmented["image"]
                patch_mask = augmented["mask"]
            except Exception as e:
                print(f"Transform error: {e}")
                print(f"Image shape: {patch_image.shape}, dtype: {patch_image.dtype}")
                print(f"Mask shape: {patch_mask.shape}, dtype: {patch_mask.dtype}")
                raise e
        else:
            # Ручное преобразование в тензор
            patch_image = torch.from_numpy(patch_image).permute(2, 0, 1).float() / 255.0
            patch_mask = torch.from_numpy(patch_mask).long()

        return patch_image, patch_mask
