from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Closed skew normal unscented Kalman filter"
LONG_DESCRIPTION = "Implementation of the algorithm mainly described in Javad Rezaie & Jo Eidsvik (2016) A skewed unscented Kalman filter, International Journal of Control, 89:12, 2572-2583, DOI: 10.1080/00207179.2016.1171912 "

setup(
    name='csnukf',
    version = VERSION,
    url='https://github.com/FelipeGiro/csnukf',
    author='Felipe Giro',
    author_email='felipelac.giro@gmail.com',
    description=DESCRIPTION,
    long_description= LONG_DESCRIPTION,
    packages=find_packages,
    install_requires=['numpy', "scipy"],
    license='MIT',
    keywords=["python", "csn", "csnukf", "closed skew normal", "unscented Kalman filter"]
)