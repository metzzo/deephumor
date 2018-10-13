.EXPORT_ALL_VARIABLES:

PYTHONPATH=./annotator
PYTHON=./venv/bin/python3.7
PIP=./venv/bin/pip3.7
SOURCE_VENV=./venv/bin/activate

clean-pyc:
	find . -name "*.pyc" -exec rm -f {} \;

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache

install:
	source $(SOURCE_VENV); \
	$(PIP) install -r requirements.txt

test: clean-pyc clean-build
	source $(SOURCE_VENV) && py.test --verbose --color=yes ./annotator

run:
	source $(SOURCE_VENV) && $(PYTHON) $(PYTHONPATH)/manage.py runserver

celery:
	source $(SOURCE_VENV) && celery -A annotator worker -l info --concurrency=4 --workdir $(PYTHONPATH)

migratedb:
	source $(SOURCE_VENV); \
	$(PYTHON) $(PYTHONPATH)/manage.py makemigrations; \
	$(PYTHON) $(PYTHONPATH)/manage.py migrate; \

filter_duplicates_by_punchlines:
	source $(SOURCE_VENV); \
	$(PYTHON) $(PYTHONPATH)/manage.py filter_duplicates_by_punchlines;