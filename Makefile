init:
	git config core.hooksPath .githooks
	pip install -r requirements.txt

test:
	python unit_tests.py
