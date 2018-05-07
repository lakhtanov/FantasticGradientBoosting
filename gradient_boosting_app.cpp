#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gradient_boosting/binarization/ThresholdContainer.h"
#include "gradient_boosting/binarization/ThresholdCreator.h"
#include "gradient_boosting/binarization/ThresholdCreatorByValue.h"
#include "gradient_boosting/binarization/ThresholdCreatorByStatistics.h"
#include "utils/data_containers/ElementContainer.h"

void TestElementContainer();
void TestThresholdContainer();
void TestThresholdCreator(
    const gradient_boosting::binarization::ThresholdCreator& creator,
    const std::vector<double>& features,
    const std::string& name);
void TestThresholdCreators();

int main() {
  TestElementContainer();
  TestThresholdContainer();
  TestThresholdCreators();
}

void TestElementContainer() {
  std::cout << "TestElementContainer" << std::endl;

  using namespace utils::data_containers;

  ElementContainer el1("abcb 12123");
  ElementContainer el2("12123.12 abcd");
  std::cout << std::fixed;
  std::cout << el1.IsDouble() << " " << el2.IsDouble()
      << " "<< el2.GetDouble() << std::endl;
}

void TestThresholdContainer() {
  std::cout << "TestThresholdContainer" << std::endl;

  using namespace gradient_boosting::binarization;

  std::vector<double> features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const size_t num_thresholds = 3;

  std::vector<std::unique_ptr<ThresholdCreator>> creators; 
  creators.push_back(std::make_unique<ThresholdCreatorByValue>(num_thresholds));
  creators.push_back(std::make_unique<ThresholdCreatorByStatistics>(num_thresholds));

  const ThresholdContainer threshold_container(creators, features);
  const std::vector<double>& thresholds = threshold_container.GetThresholds();

  std::cout << "ThresholdContainer thresholds: ";
  for (double threshold: thresholds) {
    std::cout << threshold << " ";
  }
  std::cout << std::endl;
}

void TestThresholdCreator(
    const gradient_boosting::binarization::ThresholdCreator& creator,
    const std::vector<double>& features,
    const std::string& name) {
  std::cout << "TestThresholdCreator" << std::endl;

  using namespace gradient_boosting::binarization;

  std::cout << "ThresholdCreator: " << name << std::endl;

  const std::vector<double> thresholds =
      ThresholdCreator::CreateThresholds(creator,features);

  std::cout << "threholds: ";
  for (double threshold : thresholds) {
    std::cout << threshold << " ";
  }
  std::cout << std::endl;
}

void TestThresholdCreators() {
  std::cout << "TestThresholdCreators" << std::endl;

  using namespace gradient_boosting::binarization;

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
