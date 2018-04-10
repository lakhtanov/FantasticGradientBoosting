cmake_minimum_required(VERSION 3.8)
project(FantasticGradientBoosting)

set(CMAKE_CXX_STANDARD 17)

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")
set(HEADER_FILES
	gradient_boosting/GradientBoosting.h
	gradient_boosting/binarization/BinCreator.h
	gradient_boosting/binarization/BinCreatorByAbsoluteValue.h
	gradient_boosting/binarization/BinCreatorByStatistics.h
	gradient_boosting/config/GradientBoostingConfig.h
	gradient_boosting/loss_functions/GradientBoostingLossFunction.h
	gradient_boosting/loss_functions/GradientBoostingMSELossFunction.h
	gradient_boosting/trees/GradientBoostingTree.h
	gradient_boosting/trees/GradientBoostingTreeOblivious.h
	utils/data_containers/ElementContainer.h
)
set(SOURCE_FILES
	gradient_boosting/GradientBoosting.cpp
	gradient_boosting/binarization/BinCreatorByAbsoluteValue.cpp
	gradient_boosting/binarization/BinCreatorByStatistics.cpp
	gradient_boosting/config/GradientBoostingConfig.cpp
	gradient_boosting/loss_functions/GradientBoostingMSELossFunction.cpp
	gradient_boosting/trees/GradientBoostingTreeOblivious.cpp
	gradient_boosting_app.cpp
	utils/data_containers/ElementContainer.cpp
)
add_executable(FantasticGradientBoosting ${HEADER_FILES} ${SOURCE_FILES})
