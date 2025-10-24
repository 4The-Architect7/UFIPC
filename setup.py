"""
UFIPC - Universal Framework for Information Processing Complexity

Copyright (c) 2025 Joshua Contreras / Aletheia Cognitive Technologies
Patent Pending: US Provisional Application No. 63/904,588

Licensed under MIT License for research and educational use only.
Commercial use requires separate licensing.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ufipc",
    version="1.0.0",
    author="Joshua Contreras",
    author_email="Josh.47.contreras@gmail.com",
    description="Physics-based AI benchmark measuring information processing complexity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/4The-Architect7/UFIPC",
    project_urls={
        "Bug Tracker": "https://github.com/4The-Architect7/UFIPC/issues",
        "Documentation": "https://github.com/4The-Architect7/UFIPC/blob/main/README.md",
        "Source Code": "https://github.com/4The-Architect7/UFIPC",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "anthropic>=0.25.0",
        "openai>=1.12.0",
        "google-generativeai>=0.4.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ufipc=ufipc.cli:main",
        ],
    },
    keywords=[
        "artificial intelligence",
        "benchmark",
        "information processing",
        "complexity",
        "AI evaluation",
        "neuroscience",
        "physics-based AI",
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
