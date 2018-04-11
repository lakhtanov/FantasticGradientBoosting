# FantasticGradientBoosting
# Авторы: Иван Лахтанов (lakhtanov) и Роман Александров (rialeksandrov)

В данном библиотеке реализована утилита, позволяющая обучать градиентный бустинг на решающих деревьях для задач классификации, а также применять обученный градиентный бустинг на тестовых данных.

### Сборка проекта
Для сборки проекта необходимо:
1. Склонировать репозиторий с проектом:

`git clone https://github.com/lakhtanov/FantasticGradientBoosting`

2. Собрать утилиту при помощи `cmake`:

`cmake CMakeLists.txt`

В результате этих действий в папке с репозиторием появится исполняемый файл `FantasticGradientBoosting`, являющийся утилитой, использующейся для обучения/применения модели.

### Интерфейс утилиты
У утилиты `FantasticGradientBoosting` есть два режима: обучение и применение.

#### Обучение
Для обучения модели необходимо запустить утилиту `FantasticGradientBoosting`, передав ей в качестве параметра `config=train.conf` конфигурационный файл `train.conf`, описывающий параметры тренируемой модели (такие как: тип тренируемых деревьев, количество тренируемых деревьев, глубина тренируемых деревьев, learning rate, количество ядер процессора, участвующих в обучении и др.). Также соответствующие параметры можно передавать, используя параметры командной строки, при этом в этом случае они будут иметь больший приоритет. Например:

`./FantasticGradientBoosting config=train.conf num_trees=10`

запустит обучение модели с параметрами, описанными в конфигурационном файле `train.conf`, при этом количество деревьев тренируемой модели будет равно `10`, так как соответствующий параметр был передан через параметры командной строки и, соответственно, имеет больший приоритет по сравнению с параметрами, описанными в конфигурационном файле.

##### Параметры обучения модели
`tree_type` - тип деревьев, используемых в обучаемой модели (вариант по умолчанию `tree_type=GradientBoostingTreeOblivious`)

`num_trees` - количество деревьев в обучаемой модели (вариант по умолчанию `num_trees=10`)

`depth` - глубина тренируемых деревьев (вариант по умолчанию `depth=6`)

`learning_rate` - learning rate модели (вариант по умолчанию `learning_rate=1e-2`)

`num_cores` - количество ядер, используемых для обучения модели (вариант по умолчанию `num_cores=-1` - использовать все доступные ядра процессора)

`loss_funciton` - оптимизируемая функция потерь (вариант по умолчанию `loss_function=GradientBoostingMSELossFunction` - квадратичная функция потерь)

`input_file` - путь к файлу в формате `.csv`, в котором расположены обучающие данные (вариант по умолчанию `input.csv`)

#### Применение
Перед запуском применения модели обязательно необходимо провести процедуру обучения модели, иначе модель запускаться не будет. Для применения модели необходимо запустить утилиту `FantasticGradientBoosting`, передав ей в качестве параметра `config=test.conf` конфигурационный файл `test.conf`, описывающий параметры, использующиеся при предсказании модели. Также соответствующие параметры можно передавать, используя параметры командной строки, при этом в этом случае они будут иметь больший приоритет. Например:

`./FantasticGradientBoosting config=test.conf output_file=output.csv`

запустит применение модели с параметрами, описанными в конфигурационном файле `test.conf`, при этом результаты применения модели будут записаны в файл `output.csv`, так как соответствующий параметр был передан через параметры командной строки и, соответственно, имеет больший приоритет по сравнению с параметрами, описанными в конфигурационном файле.

##### Параметры применения модели

`num_cores` - количество ядер, используемых при применении модели (вариант по умолчанию `num_cores=-1` - использовать все доступные ядра процессора)

`output_file` - путь к файлу в формате `.csv`, в который будут выведены результаты предсказания модели (вариант по умолчанию `output.csv`)

Plan by priority:
* HOW TO BUILD?    (rialeksandrov)
* utils/io/AbstractCSVReader.cpp    (rialeksandrov)
* utils/io/SlowCSVReader.cpp        (rialeksandrov)
* gradient_boosting/binarization/BinContainer.cpp    (lakhtanov)
* gradient_boosting/binarization/creators/BinCreator.cpp    (lakhtanov)
* gradient_boosting/binarization/creators/BinCreatorByStatistics.cpp    (lakhtanov)
* gradient_boosting/binarization/creators/BinCreatorByAbsoluteValue.cpp    (lakhtanov)
* gradient_boosting/config/GradientBoostingConfig.cpp      (rialeksandrov)
* gradient_boosting/loss_functions/GradientBoostingLossFunction.cpp    (lakhtanov)
* gradient_boosting/loss_functions/GradientBoosringMSELossFunction.cpp   (lakhtanov)
* gradient_boosting/tree/GradientBoostingTree.cpp    (rialeksandrov, lakhtanov)
* gradient_boosting/GradientBoosting.cpp      (rialeksandrov, lakhtanov)
* gradient_boosting_app   (rialeksandrov)
* utils/io/FastCSVReader.cpp      (rialeksandrov)
