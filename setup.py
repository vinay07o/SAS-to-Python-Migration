"""Setup file api installation"""
import setuptools

with open("README.md", "r") as f:
    long_discription = f.read()
    print(long_discription)

setuptools.setup(
    name='stats',
    version="0.1.0",
    author='Team Yash',
    description='data science models',
    long_discription=long_discription,
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'statsmodels',
        'scipy',
        'jinja2==3.0.1'
    ],
    classifiers=[
        "programming Language :: Python :: 3.8",
        "Operating System :: Linux",
        "Operating System :: Microsoft",
    ],
)