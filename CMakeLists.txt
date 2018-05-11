cmake_minimum_required(VERSION 3.8)
project(FantasticGradientBoosting)

set(CMAKE_CXX_STANDARD 14)

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")
set(HEADER_FILES
		gradient_boosting/GradientBoosting.h
		gradient_boosting/binarization/ThresholdContainer.h
		gradient_boosting/binarization/ThresholdCreator.h
		gradient_boosting/binarization/ThresholdCreatorByValue.h
		gradient_boosting/binarization/ThresholdCreatorByStatistics.h
		gradient_boosting/categories/CategoricalContainer.h
		gradient_boosting/categories/CategoricalConverter.h
		gradient_boosting/config/GradientBoostingConfig.h
		gradient_boosting/data_transformer/DataTransformer.h
		gradient_boosting/internal_data_container/InternalDataContainer.h
		gradient_boosting/loss_functions/GradientBoostingLossFunction.h
		gradient_boosting/loss_functions/GradientBoostingMSELossFunction.h
		gradient_boosting/loss_functions/GradientBoostingSplitInfo.h
		gradient_boosting/trees/GradientBoostingTree.h
		gradient_boosting/trees/GradientBoostingTreeOblivious.h
		utils/data_containers/ElementContainer.h
		utils/data_containers/DataContainer.h
		utils/io/CSVReaderBase.h
		utils/io/SimpleCSVReader.h)

set(SOURCE_FILES
		gradient_boosting/GradientBoosting.cpp
		gradient_boosting/binarization/ThresholdContainer.cpp
		gradient_boosting/binarization/ThresholdCreator.cpp
		gradient_boosting/binarization/ThresholdCreatorByValue.cpp
		gradient_boosting/binarization/ThresholdCreatorByStatistics.cpp
		gradient_boosting/categories/CategoricalContainer.cpp
		gradient_boosting/categories/CategoricalConverter.cpp
		gradient_boosting/config/GradientBoostingConfig.cpp
		gradient_boosting/data_transformer/DataTransformer.cpp
		gradient_boosting/internal_data_container/InternalDataContainer.cpp
		gradient_boosting/loss_functions/GradientBoostingLossFunction.cpp
		gradient_boosting/loss_functions/GradientBoostingMSELossFunction.cpp
		gradient_boosting/loss_functions/GradientBoostingSplitInfo.cpp
		gradient_boosting/trees/GradientBoostingTree.cpp
		gradient_boosting/trees/GradientBoostingTreeOblivious.cpp
		gradient_boosting_app.cpp
		utils/data_containers/ElementContainer.cpp
		utils/data_containers/DataContainer.cpp
		utils/io/CSVReaderBase.cpp
		utils/io/SimpleCSVReader.cpp)

add_executable(FantasticGradientBoosting ${HEADER_FILES} ${SOURCE_FILES})
