format-sources: ## Format the project sources.
	poetry run ruff format jax_utils/ tests/

formatters: format-sources ## Run all the formatters.