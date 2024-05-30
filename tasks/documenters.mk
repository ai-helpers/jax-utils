document-api: ## Document the project API.
	poetry run sphinx-apidoc -o docs/source/_api jax_utils

document-%: ## Document the project in format %.
	poetry run sphinx-build -b $* docs/source docs/build

documenters: clean-docs document-api document-html ## Run all the documenters.