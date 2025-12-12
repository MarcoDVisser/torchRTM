from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='torchrtm',
    version='1.3.4',
    description='Torch-based Vegetation Radiative Transfer Model library (PROSPECT, SAIL, SMAC)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Peng Sun, Marco D. Visser',
    author_email='sunpeng18@mails.ucas.ac.cn',
    url='https://github.com/Schimasuperbra/torchrtm',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.18',
        'pandas>=1.1',
        'scipy>=1.5',
        'matplotlib',
        'tqdm',
    ],
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='radiative transfer, PROSAIL, SMAC, torch, remote sensing, inversion',
    project_urls={
        'Source': 'https://github.com/Schimasuperbra/torchrtm',
        'Bug Tracker': 'https://github.com/Schimasuperbra/torchrtm/issues',
    },
    package_data={
        'torchrtm': [
            'data/*.csv',
            'data/sensor_information/*.pkl',
        ],
    },
)
