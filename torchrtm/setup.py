from setuptools import setup, find_packages

setup(
    name="torchrtm",
    version="0.1.0",
    description="Radiative Transfer Modeling in PyTorch",
    author="Peng Sun, Marco D. Visser",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "torchrtm": [
            "data/*.csv",
            "data/sensor_information/*.pkl",
        ],
    },
    
)
