from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ecommerce-recommender",
    version="0.1.0",
    author="MLE Team",
    author_email="mle@example.com",
    description="Рекомендательная система для электронной коммерции",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ecommerce-recommender",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
            "pre-commit>=3.3.3",
        ],
        "monitoring": [
            "prometheus-client>=0.17.1",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecommerce-api=api.main:main",
        ],
    },
) 