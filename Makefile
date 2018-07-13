.EXPORT_ALL_VARIABLES:

PYTHONPATH = ./src
PYTHON=./venv/bin/python
PIP=./venv/bin/pip
SOURCE_VENV=. ./venv/bin/activate

clean-pyc:
	find . -name "*.pyc" -exec rm -f {} \;

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

install:
	virtualenv .venv
	$(SOURCE_VENV) && $(PIP) install -e PACKAGE
	$(SOURCE_VENV) && $(PIP) install -r requirements.txt # other required packages

test: clean-pyc clean-build
	py.test --verbose --color=yes src/tests

run:
	python manage.py runserver
