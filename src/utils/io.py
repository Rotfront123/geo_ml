import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray
from rasterio import features

CLASS_NAMES = [
    "kurgany_tselye",
    "kurgany_povrezhdennye",
    "fortifikatsii",
    "gorodishcha",
    "arkhitektury",
    "finds_points",
    "object_poly",
]

# Сопоставление числа обьектам, чтобы создавать
# пиксельную маску из чисел где 0 - ничего, i - номер класса
# Не факт что это правильно, ибо нейронка может неправильно это воспринять но пока так
CLASS_MAPPING = {
    "курганы_целые": 1,
    "курганы_поврежденные": 2,
    "фортификации": 3,
    "городища": 4,
    "архитектуры": 5,
    "_FindsPoints": 6,
    "_ObjectPoly": 7,
}


def get_class_id(file_name):  # определеям что за обьект по имени файла разметки
    for keyword, class_id in CLASS_MAPPING.items():
        if keyword in file_name:
            return class_id
    return 0


def process_region(region_path):
    """
    Возвращает rgb mask(фотография) и final mask(маска объекта)
    """
    region_path = Path(region_path)

    utm_path = region_path / "UTM.json"
    with open(utm_path, "r") as f:
        utm_data = json.load(f)
    target_crs = utm_data["crs"].split("::")[-1]  # В эти координаты переводим все
    if not target_crs.startswith("EPSG:"):
        target_crs = f"EPSG:{target_crs}"

    tif_files = []
    tif_files.extend(region_path.glob("*SpOR*/*.tiff"))
    tif_files.extend(region_path.glob("*Or*/*.tiff"))
    # Ищем все геоджесоны в папке разметки
    geojson_files = list(region_path.glob("*_разметка/*.geojson"))
    if not tif_files:
        print("Нету ортофоток")
        return
    if not tif_files or not geojson_files:
        print("А где данные")
        return
    # возьмем первую ортофотку
    example = tif_files[0]
    ref_ds = rioxarray.open_rasterio(example)
    ref_ds = ref_ds.rio.reproject(target_crs)
    rgb_mask = ref_ds.values.transpose(1, 2, 0)  # (высота, ширина, (р, г, б))
    if rgb_mask.shape[2] > 3:
        rgb_mask = rgb_mask[:, :, :3]  # там 4 канал лишний какой-то фулл 255 забит

    # СОЗДАНИЕ МНОГОКЛАССОВОЙ МАСКИ
    final_mask = np.zeros((ref_ds.rio.height, ref_ds.rio.width), dtype="uint8")
    for g_file in geojson_files:
        if ("SpOR" in str(g_file) and "SpOR" in str(example)) or (
            "Or" in str(g_file) and "Or" in str(example)
        ):  # пока работает токо с ортофотками
            gdf = gpd.read_file(g_file)
            # переводим вектор в ту же систему координат, что и растр
            gdf = gdf.to_crs(target_crs)
            class_id = get_class_id(g_file.name)
            # рисуем объекты этого файла на маске
            file_mask = features.rasterize(
                [(shape, class_id) for shape in gdf.geometry],
                out_shape=(ref_ds.rio.height, ref_ds.rio.width),
                transform=ref_ds.rio.transform(),
                fill=0,
                dtype="uint8",
            )
            final_mask = np.maximum(final_mask, file_mask)
    return rgb_mask, final_mask
