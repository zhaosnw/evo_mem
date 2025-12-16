"""Setup script for Evo-Memory package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="evo-memory",
    version="0.1.0",
    author="Evo-Memory Team",
    author_email="",
    description="Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evo-memory/evo-memory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "openai>=1.0.0",
        "sentence-transformers>=2.2.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "anthropic": ["anthropic>=0.18.0"],
        "google": ["google-generativeai>=0.3.0"],
        "all": [
            "anthropic>=0.18.0",
            "google-generativeai>=0.3.0",
            "faiss-cpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evo-memory=evo_memory.main:main",
        ],
    },
)
