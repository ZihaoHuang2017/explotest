build:
	pip install --editable ./
	rm -rf ./dist
	python -m build