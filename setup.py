from setuptools import setup, find_packages

setup(
    name="ecm_factory",
    version="1.0.0",
    packages=find_packages(),
    description="A Python package containing classes that enable estimation and forecasting with different variants of the error correction models.",
    author="Jakub Kulesz",
    author_email="kuleszjakub@gmail.com",
    license="MIT",
    #url="https://github.com/JakubKulesz/time_series_regression_factory",
    install_requires=[
        "pandas>=2.1.1",
        "numpy>=1.26.0",
        "statsmodels>=0.14.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
