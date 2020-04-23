.PHONY: clean-readme
clean-readme:
	rm README.md

README.md: clean-readme
	cargo readme > README.md

.PHONY: lint
lint:
	cargo readme | cmp README.md
	cargo fmt -- --check

.PHONY: test
test:
	cargo test

.PHONY: release
release: README.md test lint
	cargo release ${BUMP}
