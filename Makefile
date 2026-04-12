PYTHON ?= python

.PHONY: install lint test ci train train-push demo api

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m py_compile scripts/create_dataset.py scripts/train.py scripts/predict.py api.py app.py space_demo/app.py

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

ci: lint test

train:
	$(PYTHON) scripts/train.py

train-push:
	$(PYTHON) scripts/train.py --push-to-hub --repo-id Bangkah/atha-text-classifier

demo:
	$(PYTHON) app.py

api:
	uvicorn api:app --host 0.0.0.0 --port 8000
