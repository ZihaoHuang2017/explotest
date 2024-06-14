build:
	pip install --editable ./
	rm -r ./dist
	python -m build