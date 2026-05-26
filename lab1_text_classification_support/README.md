# Лабораторная работа №1

Тема: классификация обращений пользователей интернет-магазина по типу проблемы.

## Установка зависимостей

```bash
py -m pip install -r requirements.txt
```

## Обучение модели

```bash
py code.py
```

После запуска будут созданы:

- `reports/report.txt`
- `reports/confusion_matrix.png`
- `models/model.pkl`
- `models/vectorizer.pkl`

## Проверка новых сообщений

```bash
py predict.py
```

Пример сообщения:

```text
I was charged twice and payment failed
```

Ожидаемая категория:

```text
Payment Problem
```
