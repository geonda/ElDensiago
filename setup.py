from setuptools import setup, find_packages
import os

# Read the README file if it exists
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="eldensiago",
    version="0.1.0",
    author="ElDensiago Team",
    description="Machine learning-based electronic density prediction for materials and molecules",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ElDensiago",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
    },
    include_package_data=True,
    package_data={
        "DenistyPredictor": [
            "pretrained_models/*/*.json",
            "pretrained_models/*/*.pth",
            "pretrained_models/*/*.txt",
            "pretrained_models/*/*.log",
            "pretrained_models/*/*.sh",
        ],
    },
    entry_points={
        "console_scripts": [
            "eldensiago=DenistyPredictor.__init__:main",
        ],
    },
    keywords="machine learning, density functional theory, materials science, chemistry, physics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ElDensiago/issues",
        "Source": "https://github.com/yourusername/ElDensiago",
        "Documentation": "https://eldensiago.readthedocs.io/",
    },
) 