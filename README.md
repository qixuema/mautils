## mautils
utils for saving my life

```bash
pip install qixuema -i https://pypi.org/simple
```

### Development & Release

This project uses `uv` for dependency management and building.

**1. Clean up old build files**
```bash
rm -rf build dist *.egg-info
```

**2. Build the package**
```bash
uv build
```

**3. Publish to PyPI**
```bash
uv publish
```

*Note: `uv publish` handles both checking and uploading. It will prompt for your PyPI token if not set via environment variable.*

**Install locally for development**
```bash
uv pip install -e ".[dev]"
```
