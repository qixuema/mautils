from setuptools import setup, find_packages

setup(
  name = 'qixuema',
  # packages = find_packages(exclude=['examples']),
  version = '0.0.6',
  license='MIT',
  description = 'qixuema utils',
  author = 'Xueqi Ma',
  author_email = 'qixuemaa@gmail.com',
  url = 'https://https://github.com/qixuema/m-utils',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'utils',
  ],
  install_requires=[
    'einx>=0.3.0',
    'einops>=0.8.0',
    'packaging>=21.0',
    'deprecated>=1.2.14',
    'numpy==1.26.0',
    'pyyaml>=6.0.1',
  ],
  setup_requires=[
    'pytest-runner',
  ],

  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)
