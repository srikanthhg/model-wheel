#!/usr/bin/env bash
set -euo pipefail
rm -rf build dist *.egg-info
python -m pip install --upgrade build
python -m build
ls -lh dist/*.whl
