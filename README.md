# FantasticGradientBoosting
# Авторы: Иван Лахтанов (lakhtanov) и Роман Александров (rialeksandrov)

В данном библиотеке реализована утилита, позволяющая обучать градиентный бустинг на решающих деревьях для задач классификации, а также применять обученный градиентный бустинг на тестовых данных.

### Сборка проекта
Для сборки проекта необходимо:
1. Склонировать репозиторий с проектом:

`git clone --recurse-submodules https://github.com/lakhtanov/FantasticGradientBoosting`

2. Собрать утилиту при помощи `cmake`:

`cmake CMakeLists.txt`

В результате этих действий в папке с репозиторием появится исполняемый файл `FantasticGradientBoosting`, являющийся утилитой, использующейся для обучения/применения модели.

### Интерфейс утилиты
Настройки утилиты `FantasticGradientBoosting` задаются файлом в формате json (пример можно найти в `config.json` в корне репозитория), который для запуска утилиты необходимо передать ей в качестве параметра командной строки. 

### Архитектура проекта
Интерфейсом для обучения модели является класс `GradientBoosting`, имеющий методы `GradientBoosting::Fit()` и `GradientBoosting::Predict()`, отвечающие за обучение и применение обученной модели, соответственно. Настройка параметров обучения/применения градиентного бустинга производится путем передачи в соответсвующие методы класса `GradientBoosting` объекта типа `GradientBoostingConfig`, содержащего внутри себя всю информацию.

#### Обучение модели
Обучение модели (осуществляющееся в результате вызова метода `GradientBoosting::Fit()`) происходит в несколько этапов:

0. Все объекты обучающей выборки случайным образом перемешиваются.
1. Для признаков данных из обучающей выборки проводится бинаризация всеми методами, указанными при конфигурации.
2. Каждое последующее дерево в градиентном бустинге строится следующим образом: параллельно строятся несколько обучающих деревьев(на данный момент в коде явно прописано, что строится одно дерево), каждое из которых строится по некоторой случайной подвыборке объектов обучающей выборки и по некоторому случайному подмножеству признаков (на данный момент в коде явно прописано, что необходимо использовать все объекты обучающей выборки и все признаки). Среди построенных на данной итерации деревьев выбирается дерево, приближающее градиент для всей выборки наилучшим способом в смысле заданной функции потерь .
3. Выбранное на шаге `2` обучающее дерево добавляется в ансамбль уже обученных решающих деревьев с весом `learning_rate`.

#### Применение модели
Применение модели (осуществляющееся в результате вызова метода `GradientBoosting::Predict()`) происходит в несколько этапов:

1. Для признаков данных из тестовой выборки проводится бинаризация, аналогичная той, что производилась для обучающей выборки.
2. Для каждого дерева вычисляются предсказания для всех объектов тестовой выборки, которые аггрегируются в итоговые предсказания.

### Описание экспериментов
Сравнение скорости работы и качества данной утилиты производилось с популярными аналогами *xgboost* и *lightgbm*. Были проведены эксперименты на двух датасетах

#### Бенчмарки на датасете [Higgs](https://www.kaggle.com/c/higgs-boson/data)

###### Выводы

#### Бенчмарки на датасете [BCI](https://www.kaggle.com/c/inria-bci-challenge#evaluation)

###### Выводы
