from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="synaptic-ml",
    version="0.1.3",
    author="Hrishikesh Rajulu",
    description="The TensorFlow for neuromorphic computing — high-level SNN framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HrishikeshRajulu/synaptic-ml",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "tqdm>=4.62",
    ],
    extras_require={
        "torch": ["torch>=1.9"],
        "loihi": ["nxsdk"],
        "brainscales": ["pynn_brainscales", "hxtorch"],
        "sklearn": ["scikit-learn>=0.24"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "scikit-learn>=0.24",
        ],
        "all": [
            "torch>=1.9",
            "scikit-learn>=0.24",
            "tqdm>=4.62",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "neuromorphic", "spiking neural network", "SNN",
        "deep learning", "Loihi", "BrainScaleS",
        "energy efficient AI", "edge computing",
    ],
    entry_points={
        "console_scripts": [
            "synaptic-info=synaptic_ml.__main__:main",
        ],
    },
)
