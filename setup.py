from setuptools import setup, Extension

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'torchRDS',
  packages = ['torchRDS'],
  version = '0.2',
  license='MIT',
  description = 'Reinforced Data Sampling for Model Diversification',
  author = 'Harry Nguyen',
  author_email = 'harry.nguyen@outlook.com',
  url = 'https://github.com/probeu/RDS',
  download_url = 'https://github.com/probeu/RDS/archive/v_02.tar.gz',
  keywords = ['Data-Sampling', 'Reinforcement-Learning', 'Machine-Learning'],
  install_requires=[
          'numpy',
          'torch',
          'scikit-learn',
          'pandas',
          'pickle'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
  long_description=long_description,
  long_description_content_type='text/markdown'
)