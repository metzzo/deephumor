.EXPORT_ALL_VARIABLES:

PYTHONPATH = ./src


clean-pyc:
	find . -name "*.pyc" -exec rm -f {} \;

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

install:
	- pip install -r requirements.txt

test: clean-pyc clean-build
	py.test --verbose --color=yes src/tests

run:
	python manage.py runserver