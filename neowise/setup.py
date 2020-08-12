from setuptools import setup, find_packages

with open("LICENSE.md", "r") as fh:
    license = fh.read()
with open("DOCUMENTATION.md", "r") as fg:
    long_description = fg.read()
setup(name="neowise",
      version='0.0.5',
      description="A Deep Learning library built from scratch using Python and NumPy",
      author="Pranav Sastry",
      author_email="pranava.sri@gmail.com",
      license=license,
      license_content_type="text/markdown",
      long_description=long_description,
      long_description_content_type="text/markdown",
      maintainer="Pranav Sastry",
      url="https://github.com/pranavsastry/neowise",
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'hdfdict', 'prettytable', 'tqdm'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License"
      ])
