from setuptools import setup, find_packages

setup(name='neowise',
      version='0.0.1',
      description='A Deep Learning library for beginners',
      author='Pranav Sastry',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'math', 'sklearn', 'pandas',
                        'seaborn', 'tensorflow', 'keras', 'hdfdict',
                        'prettytable', 'tqdm', 'time', 'sys'])
