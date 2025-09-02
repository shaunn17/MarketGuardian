"""
Setup script for Financial Anomaly Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="financial-anomaly-detection",
    version="1.0.0",
    author="Financial Anomaly Detection Team",
    author_email="contact@example.com",
    description="A comprehensive machine learning system for detecting unusual trading patterns and potential fraud in financial markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/financial-anomaly-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "dashboard": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anomaly-detection=examples.run_analysis:main",
            "anomaly-simple=examples.simple_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
