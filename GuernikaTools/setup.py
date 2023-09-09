from setuptools import setup, find_packages

from guernikatools._version import __version__

setup(
    name='guernikatools',
    version=__version__,
    url='https://github.com/GuernikaCore/GuernikaModelConverter',
    description="Run Stable Diffusion on Apple Silicon with Guernika",
    author='Guernika',
    install_requires=[
        "coremltools>=7.0b2",
        "diffusers[torch]",
        "torch",
        "transformers>=4.30.0",
        "scipy",
        "numpy<1.24",
        "pytest",
        "scikit-learn==1.1.2",
        "pytorch_lightning",
        "OmegaConf",
        "six",
        "safetensors",
        "pyinstaller",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
