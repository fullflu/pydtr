dev:
	pip install --upgrade pip
	pip install -e ./.
	pip install category_encoders
	pip install pytest
	pip install coverage
	pip install twine

package:
	python setup.py sdist
	python setup.py bdist_wheel

test:
	coverage run --source=src/pydtr -m pytest
	coverage report
	coverage xml
