from setuptools import setup, find_packages
import sys

class NullWriter(object):
    def write(self, arg):
        pass

nullwrite = NullWriter()
oldstderr = sys.stderr
sys.stderr = nullwrite  # silence stderr

setup(
    name="scentinel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[       
    "scanpy==1.9.3",
    "pandas==1.5.3",
    "numpy==1.22.1",
    "scipy==1.7.3",
    "matplotlib==3.7.1",
    "seaborn==0.12.2",
    "scikit-learn",
    "requests==2.31.0",
    "psutil==5.9.5",
    "mygene==3.2.2",
    "gseapy==1.0.5",
    "pymc3",
    "scikit-optimize"
    ],
)

sys.stderr = oldstderr  # reset stderr