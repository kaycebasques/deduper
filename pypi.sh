python3 -m venv venv
. venv/bin/activate
pip install pip-tools
pip-compile -o pypi.lock pypi.txt --strip-extras
deactivate
rm -rf venv
