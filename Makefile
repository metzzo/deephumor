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
	$(SOURCE_VENV) && $(PIP) install -r requirements.txt # other required packages

test: clean-pyc clean-build
	$(SOURCE_VENV) && py.test --verbose --color=yes ./annotator

run:
	$(SOURCE_VENV) && cd $(PYTHONPATH) && python manage.py runserver

celery:
	$(SOURCE_VENV) && celery -A annotator worker -l info --concurrency=4 --workdir $(PYTHONPATH)

migratedb:
	$(SOURCE_VENV) && cd $(PYTHONPATH) && python manage.py makemigrations && python $(PYTHONPATH)/manage.py migrate

filter_duplicates:
	$(SOURCE_VENV) && cd $(PYTHONPATH) && python manage.py filter_duplicates