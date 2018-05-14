# TODO: this is not customized for this project

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	name '*~' -exec rm --force  {}

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

# isort:
#    sh -c "isort --skip-glob=.tox --recursive . "


test: clean-pyc
	py.test --verbose --color=yes ./src/tests/

run:
	python manage.py runserver