from setuptools import setup, find_packages

setup(
    name='torchrtm',
    version='0.1.0',
    description='Torch-based Radiative Transfer Model library (PROSPECT, SAIL, SMAC)',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scipy'
    ],
    python_requires='>=3.7',
)
