PYTHON ?= $(if $(wildcard venv_mska/bin/python),venv_mska/bin/python,python3)
PROFILE ?= main

.PHONY: help check summary env vendor vendor-check paper verify-archive test-canonicalize test-matched-blocks test-coherent-blocks test-sign-stability

help:
	@$(PYTHON) tools/repository.py help

check:
	@$(PYTHON) tools/repository.py check

summary:
	@$(PYTHON) tools/repository.py summary

env:
	@$(PYTHON) tools/repository.py env --profile $(PROFILE)

vendor:
	@$(PYTHON) tools/repository.py vendor

vendor-check:
	@$(PYTHON) tools/repository.py vendor --check

paper:
	@$(PYTHON) tools/repository.py paper

verify-archive:
	@$(PYTHON) tools/repository.py verify-archive

test-canonicalize:
	@$(PYTHON) -m unittest scripts.test_canonicalize_motion -v

test-matched-blocks:
	@$(PYTHON) -m unittest scripts.test_matched_motion_blocks -v

test-coherent-blocks:
	@$(PYTHON) -m unittest scripts.test_coherent_motion_blocks -v

test-sign-stability:
	@$(PYTHON) -m unittest scripts.test_sign_stability_controls -v
