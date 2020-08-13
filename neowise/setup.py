from setuptools import setup, find_packages

with open("DOCUMENTATION.md", "r") as fh:
    long_description = fh.read()
setup(name="neowise",
      version='0.1.0',
      description="A Deep Learning library built from scratch using Python and NumPy",
      author="Pranav Sastry",
      author_email="pranava.sri@gmail.com",
      long_description=long_description,
      long_description_content_type='text/markdown',
      maintainer="Pranav Sastry",
      url="https://github.com/pranavsastry/neowise",
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'hdfdict', 'prettytable', 'tqdm'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License"
      ])
