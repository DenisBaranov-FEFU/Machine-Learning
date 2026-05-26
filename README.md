# Лабораторные работы по машинному обучению

Репозиторий содержит две лабораторные работы по дисциплине **«Машинное обучение в системах искусственного интеллекта»**.

## Структура репозитория

```text
ml_course_labs_repository/
├── lab1_text_classification_support/
│   ├── data.csv
│   ├── code.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── README.md
│   ├── models/
│   └── reports/
│
├── lab2_streaming_user_profiles/
│   ├── data/
│   │   └── streaming_users.csv
│   ├── project.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── README.md
│   ├── models/
│   └── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Лабораторная работа 1

**Тема:** автоматическая классификация обращений пользователей интернет-магазина по типу проблемы.

Запуск:

```bash
cd lab1_text_classification_support
py -m pip install -r requirements.txt
py code.py
py predict.py
```

## Лабораторная работа 2

**Тема:** анализ профилей пользователей стримингового сервиса методами машинного обучения.


Запуск:

```bash
cd lab2_streaming_user_profiles
py -m pip install -r requirements.txt
py project.py
py predict.py
```


## Примечание

Файлы моделей `.pkl` могут быть пересозданы запуском скриптов обучения.
