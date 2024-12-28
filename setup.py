from setuptools import setup, find_packages

setup(
    name="mvica_lingam",
    description="Multiview ICA-based LiNGAM",
    version="0.0.1",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=['numpy>=1.12', 'scikit-learn>=0.23', 'python-picard',
                      'scipy>=0.18.0', 'matplotlib>=2.0.0']
)