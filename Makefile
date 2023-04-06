.PHONY : quickstart test coverage coverage-html docs package help
.DEFAULT_GOAL:= help

PROJECT_NAME = stats
PACKAGE_NAME = yash_$(PROJECT_NAME)
VERSION_LOCATION = stats/__init__.py 
PY_VERSION_MINOR = 3.8

SHELL = /bin/bash

CONDA = conda
JUPYTER_HOST=localhost
JUPYTER_PORT=8888
JUPYTER_DOMAIN=jupyter notebook

ACTIVATE_VENV=$(CONDA) activate ./venv/
PYTHON=$(ACTIVATE_VENV) && python

quickstart = venv test

venv: requirements.txt setup.py 
ifneq ($(wildcard venv/),)
	@echo -e "\n### Found existing conda environment.....deleting it"
	rmdir venv /s /q 2>NUL
endif
	@echo -e "\n### Creating conda environment from requirements.txt"
	$(CONDA) create -y --prefix ./venv python=$(PY_VERSION_MINOR)
	$(ACTIVATE_VENV) && pip install -r ./requirements.txt

jupyter : venv
	$(PYTHON) -m ipykernel install --user --name $(PROJECT_NAME) --display-name "$(PROJECT_NAME) (Python $(PY_VERSION_MINOR))"
	@echo -e "\n\tJupyterHub is available with the '$(PROJECT_NAME) (Python $(PY_VERSION_MINOR))' development kernel here:"
	@echo -e "\n\thttp://$(JUPYTER_HOST):$(JUPYTER_PORT)\n"

test :
	$(ACTIVATE_VENV) && pytest

coverage :
	$(ACTIVATE_VENV) && coverage run -m pytest
	$(ACTIVATE_VENV) && coverage report 

coverage-html :
	$(ACTIVATE_VENV) && coverage html
	$(PYTHON) -m http.server -d htmlcov --bind "0.0.0.0" 9999

CURRENT_PKG_VERSION=$(shell sed -n 's/__version__ = "\(.*\)"/\1/p' $(VERSION_LOCATION))

package = dist/$(PACKAGE_NAME)-$(CURRENT_PKG_VERSION)-py3-none-any.whl dist/$(PACKAGE_NAME)-$(CURRENT_PKG_VERSION).tar.gz

dist/$(PROJECT_NAME)-$(CURRENT_PKG_VERSION)-py3-none-any.whl : setup.py
	$(PYTHON) setup.py bdist_wheel
	
dist/$(PROJECT_NAME)-$(CURRENT_PKG_VERSION).tar.gz : setup.py
	$(PYTHON) setup.py sdist

format-python:
		$(ACTIVATE_VENV) && isort stats/ --settings-file setup.cfg
		$(ACTIVATE_VENV) && black stats/ --config black.toml
		$(ACTIVATE_VENV) && isort tests/ --settings-file setup.cfg
		$(ACTIVATE_VENV) && black tests/ --config black.toml
	
format-python-check:
		$(ACTIVATE_VENV) && isort stats/ --diff --check --settings-file setup.cfg
		$(ACTIVATE_VENV) && black stats/ --diff --check --config black.toml
		$(ACTIVATE_VENV) && isort tests/ --diff --check --settings-file setup.cfg
		$(ACTIVATE_VENV) && black tests/ --diff --check --config black.toml