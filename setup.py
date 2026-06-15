from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="adn",
    version="1.0.0",
    description="ADN: Adaptive Image Downscaling Network (unofficial reproduction)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ADN Reproduction Authors",
    license="MIT",
    packages=find_packages(exclude=("tests", "scripts")),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23",
        "Pillow>=9.0",
        "pyyaml>=6.0",
        "tqdm>=4.64",
        "scipy>=1.9",
    ],
    entry_points={
        "console_scripts": [
            "adn-train=scripts.train:main",
            "adn-test=scripts.test:main",
            "adn-infer=scripts.infer:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
