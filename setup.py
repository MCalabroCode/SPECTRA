from setuptools import setup, find_packages

setup(
    name="spectra",
    version="0.1.0",
    description="SPECTRA: Causal Interventions in Single-Cell Perturbation Data",
    author="Michele Calabro'",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)