.EXPORT_ALL_VARIABLES:

PYTHONPATH = ./annotator
PYTHON=./venv/bin/python
PIP=./venv/bin/pip
SOURCE_VENV=. ./venv/bin/activate

clean-pyc:
	find . -name "*.pyc" -exec rm -f {} \;

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache

install:
	virtualenv .venv
	$(SOURCE_VENV) && $(PIP) install -e PACKAGE
	$(SOURCE_VENV) && $(PIP) install -r requirements.txt # other required packages

test: clean-pyc clean-build
	py.test --verbose --color=yes ./annotator

run:
	python $(PYTHONPATH)/manage.py runserver

celery:
	celery -A annotator worker -l info --concurrency=4 --workdir $(PYTHONPATH)

migratedb:
	python $(PYTHONPATH)/manage.py makemigrations && python $(PYTHONPATH)/manage.py migrate

