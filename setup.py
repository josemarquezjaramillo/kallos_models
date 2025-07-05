from setuptools import find_packages, setup

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kallos_models",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for training and tuning deep learning models for cryptocurrency price prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/kallos_models", # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "darts[pytorch]>=0.21.0",
        "optuna>=3.0.0",
        "SQLAlchemy>=1.4.0",
        "psycopg2-binary", # Assuming a PostgreSQL database
        "matplotlib>=3.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "kallos-run=main:main",
        ],
    },
)
