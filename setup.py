from setuptools import setup, find_packages

setup(
    name="archaeology-segmentation",
    version="0.1.0",
    description="Segmentation of geological objects",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "rasterio>=1.3.0",
    ],
)
