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
	@missing=0; \
	pip list --format=freeze > /tmp/installed.txt; \
	while read pkg; do \
		[ -z "$$pkg" ] && continue; \
		echo "$$pkg" | grep -q "^#" && continue; \
		echo "$$pkg" | grep -q "^-r" && continue; \
		pkg_name=$$(echo "$$pkg" | sed -E 's/([a-zA-Z0-9_-]+).*/\1/'); \
		if [ "$$pkg_name" = "pre-commit" ]; then \
			if grep -qi "pre_commit==" /tmp/installed.txt; then \
				echo "   ✅ $$pkg_name"; \
			else \
				echo "   ❌ $$pkg_name"; \
				missing=1; \
			fi; \
		elif grep -qi "^$$pkg_name==" /tmp/installed.txt 2>/dev/null; then \
			echo "   ✅ $$pkg_name"; \
		else \
			echo "   ❌ $$pkg_name"; \
			missing=1; \
		fi; \
	done < requirements-dev.txt; \
	if [ $$missing -eq 0 ]; then \
		echo "✨ Все пакеты установлены!"; \
	else \
		echo "❌ Некоторые пакеты отсутствуют"; \
		exit 1; \
	fi
