from setuptools import setup

setup(name='OpenGNN',
      version='0.1',
      description='Open source machine learning for graph-structured data',
      author='Patrick Fernandes',
      author_email='pattuga@gmail.com',
      packages=['opengnn'],
      install_requires=[
          "docopt",
          "dpu_utils",
          "tensorflow >= 1.10"],
      entry_points={
          "console_scripts": [
              "ognn-build-vocab=opengnn.bin.build_vocab:main",
              "ognn-main=opengnn.bin.main:main",
          ]}
      )
