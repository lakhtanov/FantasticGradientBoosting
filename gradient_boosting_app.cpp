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
#include "gradient_boosting/GradientBoosting.h"
#include "utils/data_containers/ElementContainer.h"
#include "utils/io/ResultDumper.h"
#include "utils/io/SimpleCSVReader.h"

#include "third_party/ctpl/ctpl_stl.h"
#include "third_party/json/single_include/nlohmann/json.hpp"

using json = nlohmann::json;
using thread_pool = ctpl::thread_pool;

int main(int argc,  char** argv) {
  if (argc <= 1) {
    std::cout << "JSON path was not specified" << std::endl;
    return 0;
  }
  const std::string path(argv[1]);

  std::ifstream in(path);
  json config;
  in >> config;
  const gradient_boosting::config::GradientBoostingConfig gb_config(config);
  gradient_boosting::GradientBoosting gb(gb_config);

  utils::io::SimpleCSVReader train_reader(gb_config.GetTrainData());
  const auto train_data = train_reader.ReadFile();
  gb.Fit(train_data);

  utils::io::SimpleCSVReader test_reader(gb_config.GetTestData());
  const auto test_data = test_reader.ReadFile();
  const auto res = gb.PredictProba(test_data);

  utils::io::ResultDumper out(
      gb_config.GetResultFile(),
      gb_config.GetIdValueName(),
      gb_config.GetTargetValueName());

  for (const auto& el : res) {
    out.AddResult(el.first, el.second);
  }

  return 0;
}
