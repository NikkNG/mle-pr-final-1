# Эпик: Рекомендательная система для электронной коммерции

## Общее описание проекта

Создание полноценной рекомендательной системы для электронной коммерции с фокусом на добавления товаров в корзину. Проект включает исследование данных, разработку модели, создание API-сервиса, настройку автоматического дообучения и мониторинга.

## Структура проекта

```
ecommerce-recommender-fin/
├── config/
│   ├── deployment_config.json
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       ├── datasources/
│       └── provisioning/
├── data/
│   └── raw/
├── models/
│   ├── production/
│   └── backup/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_experiments.ipynb
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── model_manager.py
│   │   └── retraining_endpoints.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py
│   │   ├── content_based.py
│   │   └── hybrid.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── dashboard.py
│   │   └── alerting.py
│   └── retraining/
│       ├── __init__.py
│       ├── scheduler.py
│       ├── metrics_monitor.py
│       ├── model_trainer.py
│       ├── deployment_manager.py
│       └── retraining_service.py
├── scripts/
│   ├── start_api.sh
│   └── mlflow/
│       └── start_mlflow_server.sh
├── experiments/
├── logs/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── config.py
├── setup.py
├── requirements.txt
└── README.md
```

---

## Бизнес-цели

### Основные KPI:
1. **Увеличение конверсии**: Повышение процента покупок после просмотра рекомендаций
2. **Увеличение среднего чека**: Рекомендация дополнительных товаров
3. **Улучшение пользовательского опыта**: Релевантные и персонализированные рекомендации
4. **Увеличение времени на сайте**: Вовлечение пользователей через интересные товары



## Пользовательские истории

### История 1: Исследование и анализ данных
**Как** аналитик данных  
**Я хочу** провести исследовательский анализ данных  
**Чтобы** понять структуру данных, выявить закономерности и определить подходы к моделированию

#### Definition of Done:
- Jupyter Notebook `01_eda.ipynb` с полным EDA
- Документированы все найденные инсайты
- Предложены гипотезы для моделирования


### История 2: Настройка инфраструктуры MLOps - ЗАВЕРШЕНО
**Как** ML-инженер  
**Я хочу** настроить MLflow для отслеживания экспериментов  
**Чтобы** иметь возможность версионировать модели и сравнивать результаты

#### Definition of Done:
- [x] Скрипт `setup_mlflow.sh` для развертывания
- [x] Документация по запуску MLflow
- [x] Работающий MLflow сервер

#### 🚀 Использование:
```bash
# Перейти в директорию проекта
cd /home/mle-user/mle_projects/mle-pr-final/ecommerce-recommender-fin

# Запуск MLflow сервера
source .venv_fin_project/bin/activate && bash scripts/mlflow/start_mlflow_server.sh

# Доступ к UI: http://127.0.0.1:5001
```

---

### История 3: Разработка и обучение моделей (ЗАВЕРШЕНА)
**Как** ML-инженер  
**Я хочу** создать и сравнить различные подходы к рекомендациям  
**Чтобы** выбрать наилучшую модель для продакшена

#### Definition of Done:
- [x] Jupyter Notebook `02_experiments.ipynb` с экспериментами
- [x] Модули в `src/models/` с реализацией алгоритмов
- [x] Пайплайн предобработки в `src/data/preprocessing.py`
- [x] Обученная модель в бинарном формате


---

### История 4: Создание API-сервиса
**Как** разработчик приложения  
**Я хочу** получить рекомендации через HTTP API  
**Чтобы** интегрировать их в веб-приложение


#### Definition of Done:
- FastAPI приложение в `src/api/`
- Dockerfile и docker-compose.yml
- Документация API (автогенерируемая через Swagger)
- Работающий контейнер с API

## Развертывание

### Локальная разработка

```bash

# Запуск сервера разработки
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Доступ к документации
open http://localhost:8000/docs
```
### Продакшен развертывание

```bash
# Сборка Docker образа
docker build -t ecommerce-recommender-api .

# Запуск контейнера
docker run -p 8000:8000 ecommerce-recommender-api

# Или с Docker Compose
docker-compose up -d
```

---

### История 5: Автоматическое дообучение модели
**Как** ML-инженер  
**Я хочу** настроить регулярное дообучение модели  
**Чтобы** рекомендации учитывали новые данные о поведении пользователей

#### Definition of Done:
- Python файл с DAG в `airflow/dags/retrain_pipeline.py`
- Конфигурация расписания обучения
- Логирование всех этапов пайплайна

---

### История 6: Система мониторинга 
**Как** DevOps инженер  
**Я хочу** отслеживать работу системы рекомендаций  
**Чтобы** быстро выявлять и устранять проблемы

#### Definition of Done:
- [x] Код для сбора метрик в `src/monitoring/`
- [x] Описание всех метрик
- [x] Конфигурация дашбордов

После запуска системы мониторинг доступен по следующим адресам:

| 📊 Сервис | 🌐 URL | 🔐 Доступ | 📄 Назначение |
|-----------|--------|-----------|---------------|
| **🎨 Grafana Dashboard** | [http://localhost:3000](http://localhost:3000) | `admin/admin123` | Визуализация и алерты |
| **📈 Prometheus** | [http://localhost:9090](http://localhost:9090) | Открытый | Сбор метрик |
| **🔍 Monitoring API** | [http://localhost:8000/monitoring/health](http://localhost:8000/monitoring/health) | Открытый | API мониторинга |


## Метрики

### Метрики API

| Метрика | Описание | Тип |
|--------|-------------|------|
| `request_count` | Общее количество API запросов | Counter |
| `error_count` | Общее количество ошибок API | Counter |
| `error_rate` | Процент неудачных запросов | Gauge |
| `avg_response_time` | Среднее время ответа в секундах | Gauge |
| `min_response_time` | Минимальное время ответа | Gauge |
| `max_response_time` | Максимальное время ответа | Gauge |

### Бизнес-метрики

| Метрика | Описание | Тип |
|--------|-------------|------|
| `recommendations_shown` | Общее количество показанных рекомендаций | Counter |
| `recommendations_clicked` | Общее количество кликов по рекомендациям | Counter |
| `items_added_to_cart` | Товары, добавленные в корзину из рекомендаций | Counter |
| `ctr` | Коэффициент кликабельности (клики/показы) | Gauge |
| `conversion_rate` | Коэффициент конверсии (добавления в корзину/показы) | Gauge |
| `catalog_coverage` | Процент каталога в рекомендациях | Gauge |
| `unique_items_count` | Количество уникальных рекомендованных товаров | Gauge |

### Метрики модели

| Метрика | Описание | Тип |
|--------|-------------|------|
| `model_prediction_time` | Время, затраченное на предсказания модели | Histogram |
| `model_confidence` | Оценки уверенности модели | Gauge |




---

## Доступные сервисы

После запуска системы через `docker-compose up` все сервисы становятся доступными:

| 🎯 Сервис | 🌐 URL | 🔐 Доступ | 📄 Описание |
|-----------|--------|-----------|-------------|
| **🤖 API Рекомендаций** | [http://localhost:8000](http://localhost:8000) | Открытый | REST API для рекомендаций |
| **📚 API Документация** | [http://localhost:8000/docs](http://localhost:8000/docs) | Открытый | Swagger UI документация |
| **🎨 Grafana Dashboard** | [http://localhost:3000](http://localhost:3000) | `admin/admin123` | Мониторинг и визуализация |
| **📈 Prometheus** | [http://localhost:9090](http://localhost:9090) | Открытый | Сбор метрик системы |
| **🧪 MLflow** | [http://localhost:5000](http://localhost:5000) | Открытый | Управление экспериментами |
| **🗄️ PostgreSQL** | `localhost:5432` | `postgres/postgres` | База данных |

### Основные эндпоинты API

| Эндпоинт | Описание | Пример |
|----------|----------|---------|
| `GET /health` | Статус системы | `curl http://localhost:8000/health` |
| `GET /recommendations/{user_id}` | Персональные рекомендации | `curl http://localhost:8000/recommendations/1` |
| `GET /similar_items/{item_id}` | Похожие товары | `curl http://localhost:8000/similar_items/131` |
| `GET /popular_items` | Популярные товары | `curl http://localhost:8000/popular_items` |
| `GET /models/status` | Статус всех моделей | `curl http://localhost:8000/models/status` |

### Быстрый запуск

```bash
# Запуск всех сервисов
docker-compose up -d

# Проверка статуса
curl http://localhost:8000/health

# Тестирование рекомендаций
curl http://localhost:8000/recommendations/1?num_recommendations=5
```
