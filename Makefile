# Makefile
.PHONY: integration-tests

# Developing
default:
	@echo No make target specified.

lint: lint-fix

lint-fix:
	black src

lint-check:
	black src --diff --check

module.tar.gz:
	make pyinstaller
	tar czf module.tar.gz dist/main meta.json

pyinstaller:
	./build.sh
