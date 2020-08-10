from setuptools import setup, find_packages

setup(name="neowise",
      version='0.0.1',
      description="A Deep Learning library for built from scratch using Python and NumPy",
      author="Pranav Sastry",
      author_email="pranava.sri@gmail.com",
      url="https://github.com/pranavsastry/neowise",
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'math', 'sklearn', 'pandas',
                        'seaborn', 'tensorflow', 'keras', 'hdfdict',
                        'prettytable', 'tqdm', 'time', 'sys'])
