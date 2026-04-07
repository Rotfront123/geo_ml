.PHONY: help install install-dev test lint format clean

help:
	@echo "Доступные команды:"
	@echo "  make install      - Установка зависимостей"
	@echo "  make install-dev  - Установка dev зависимостей"
	@echo "  make test         - Запуск тестов"
	@echo "  make lint         - Проверка кода"
	@echo "  make format       - Форматирование кода"
	@echo "  make clean        - Очистка"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/
	black --check src/ --line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ --line-length=100
	isort src/ --profile black

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache
verify:
	@echo "🔍 Проверка установки..."
	@pip list --format=freeze > /tmp/installed.txt && \
	cat requirements-dev.txt | grep -v "^#" | grep -v "^-r" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | while read pkg; do \
		if grep -qi "$$pkg==" /tmp/installed.txt; then \
			echo "   ✅ $$pkg"; \
		else \
			echo "   ❌ $$pkg"; \
			exit 1; \
		fi; \
	done && echo "✨ Все пакеты установлены!" || echo "❌ Некоторые пакеты отсутствуют"
