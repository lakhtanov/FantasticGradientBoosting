#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gradient_boosting/binarization/ThresholdContainer.h"
#include "gradient_boosting/binarization/ThresholdCreator.h"
#include "gradient_boosting/binarization/ThresholdCreatorByValue.h"
#include "gradient_boosting/binarization/ThresholdCreatorByStatistics.h"
#include "gradient_boosting/config/GradientBoostingConfig.h"
#include "utils/data_containers/ElementContainer.h"

#include "third_party/json/single_include/nlohmann/json.hpp"

using json = nlohmann::json;


void TestElementContainer();
void TestJSONReadPrint();
void TestThresholdContainer();
void TestThresholdCreator(
    const gradient_boosting::binarization::ThresholdCreator& creator,
    const std::vector<double>& features,
    const std::string& name);
void TestThresholdCreators();

int main(int argc,  char** argv) {
  TestElementContainer();
  TestJSONReadPrint();
  TestThresholdContainer();
  TestThresholdCreators();
  if (argc <= 1) {
    std::cout << "JSON path was not specified" << std::endl;
    return 0;
  }
  const std::string path(argv[1]);
  std::ifstream in(path);
  json config;
  in >> config;
  const gradient_boosting::config::GradientBoostingConfig gb_config(config);
  std::cout << "Achtung working json reading to config : "
            << gb_config.GetNumberOfStatisticsThresholds() << " "
            << gb_config.GetNumberOfValueThresholds() << std::endl;
  return 0;
}

void TestElementContainer() {
  std::cout << "TestElementContainer" << std::endl;

  using utils::data_containers::ElementContainer;

  ElementContainer el1("abcb 12123");
  ElementContainer el2("12123.12 abcd");
  std::cout << std::fixed;
  std::cout << el1.IsDouble() << " " << el2.IsDouble()
      << " "<< el2.GetDouble() << std::endl;
}

void TestJSONReadPrint() {
  using gradient_boosting::config::GradientBoostingConfig;
  const json j2 = {
      {"Verbose", "v1"},
      {"BoostingConfig",
          {
              {"NumberOfValueThresholds", 10},
              {"NumberOfStatisticsThresholds", 20},
              {"LossFunction", "MSE"}
          }
      }
  };
  std::cout << j2.dump(4) << std::endl;

  const GradientBoostingConfig config(j2);
  std::cout << config.GetNumberOfValueThresholds() << " "
            << config.GetNumberOfStatisticsThresholds() << std::endl;

}

void TestThresholdContainer() {
  std::cout << "TestThresholdContainer" << std::endl;

  using gradient_boosting::binarization::ThresholdContainer;
  using gradient_boosting::binarization::ThresholdCreator;
  using gradient_boosting::binarization::ThresholdCreatorByValue;
  using gradient_boosting::binarization::ThresholdCreatorByStatistics;

  std::vector<double> features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const size_t num_thresholds = 3;

  std::vector<std::unique_ptr<ThresholdCreator>> creators;
  creators.push_back(std::make_unique<ThresholdCreatorByValue>(num_thresholds));
  creators.push_back(
      std::make_unique<ThresholdCreatorByStatistics>(num_thresholds));

  const ThresholdContainer threshold_container(creators, features);
  const std::vector<double>& thresholds = threshold_container.GetThresholds();

  std::cout << "ThresholdContainer thresholds: ";
  for (double threshold : thresholds) {
    std::cout << threshold << " ";
  }
  std::cout << std::endl;
}

void TestThresholdCreator(
    const gradient_boosting::binarization::ThresholdCreator& creator,
    const std::vector<double>& features,
    const std::string& name) {
  std::cout << "TestThresholdCreator" << std::endl;

  using gradient_boosting::binarization::ThresholdCreator;

  std::cout << "ThresholdCreator: " << name << std::endl;

  const std::vector<double> thresholds =
      ThresholdCreator::CreateThresholds(creator, features);

  std::cout << "threholds: ";
  for (double threshold : thresholds) {
    std::cout << threshold << " ";
  }
  std::cout << std::endl;
}

void TestThresholdCreators() {
  std::cout << "TestThresholdCreators" << std::endl;

  using gradient_boosting::binarization::ThresholdCreatorByValue;
  using gradient_boosting::binarization::ThresholdCreatorByStatistics;

  std::vector<double> features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const size_t num_thresholds = 3;

  TestThresholdCreator(
      ThresholdCreatorByValue(num_thresholds),
      features,
      "ThresholdCreatorByValue");

  TestThresholdCreator(
      ThresholdCreatorByStatistics(num_thresholds),
      features,
      "ThresholdCreatorByStatistics");
}
