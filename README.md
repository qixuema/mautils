## mautils
utils for saving my life

```
pip install qixuema -i https://pypi.org/simple
```

Clean up old build files
```
rm -rf build dist *.egg-info
```

install and update build tools
```
uv pip install -U packaging
```

build
```
python -m build
```
Upgrade twine to the latest version
```
pip install --upgrade twine
```

check
```
twine check dist/*
```

Upload the package to PyPI
```
twine upload dist/*
```
Upgrade and install the package
```
pip install --upgrade qixuema -i https://pypi.org/simple
```

install on local
```
python -m pip install -e . 
```
