check-types: ## Check the project code types with mypy.
	poetry run mypy jax_utils/ tests/ --config-file .mypy.ini

check-tests: ## Check the project unit tests with pytest.
	poetry run pytest --numprocesses="auto" tests/

check-format: ## Check the project source format with ruff.
	poetry run ruff format --check jax_utils/ tests/

check-poetry: ## Check the project pyproject.toml with poetry.
	poetry check --lock

check-quality: ## Check the project code quality with ruff.
	poetry run ruff check jax_utils/ tests/

check-security: ## Check the project code security with bandit.
	poetry run bandit --recursive --configfile=pyproject.toml jax_utils/

check-coverage: ## Check the project test coverage with coverage.
	poetry run pytest --cov=jax_utils/ --cov-fail-under=50 --numprocesses="auto" tests/

checkers: check-types check-format check-quality check-security ## Run all the checkers. (add: check-coverage)