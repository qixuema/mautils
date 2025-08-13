# mautils
utils for saving my life

```
pip install qixuema
```

# Clean up old build files
rm -rf build dist *.egg-info

pip install -U packaging

python -m build

# Upgrade twine to the latest version
pip install --upgrade twine


twine check dist/*

# Upload the package to PyPI
twine upload dist/*

# Upgrade and install the package
pip install --upgrade qixuema -i https://pypi.org/simple