from setuptools import setup, find_packages


setup(
    name="RAHATAI",
    version="0.1.0",
    description="RahatAI - Multilingual Crisis Response NLP",
    packages=find_packages(),
    include_package_data=True,
    # Do not force-install heavy deps here; developers should install required packages separately.
    install_requires=[],
)
