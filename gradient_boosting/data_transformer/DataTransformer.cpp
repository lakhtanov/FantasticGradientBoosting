#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "gradient_boosting/binarization/ThresholdCreatorByStatistics.h"
#include "gradient_boosting/binarization/ThresholdCreatorByValue.h"
#include "gradient_boosting/data_transformer/DataTransformer.h"


namespace gradient_boosting {
namespace data_transformer {

using std::string;
using std::vector;

using ThresholdCreatorByStatistics = gradient_boosting::binarization::ThresholdCreatorByStatistics;
using ThresholdCreatorByValue = gradient_boosting::binarization::ThresholdCreatorByValue;
using InternalDataContainer = gradient_boosting::internal_data_container::InternalDataContainer;
using DataContainer = utils::data_containers::DataContainer;
using GradientBoostingConfig = gradient_boosting::config::GradientBoostingConfig;

DataTransformer::DataTransformer(const GradientBoostingConfig& config)
    : target_value_name_(config.GetTargetValueName())
    , task_type_(config.GetTaskType()) {
  creators_.push_back(
      std::make_unique<ThresholdCreatorByValue>(config.GetNumberOfValueThresholds()));
  creators_.push_back(
      std::make_unique<ThresholdCreatorByStatistics>(config.GetNumberOfValueThresholds()));
}

InternalDataContainer DataTransformer::FitAndTransform(const DataContainer& data) {
  Fit(data);
  return Transform(data);
}

void DataTransformer::Fit(const DataContainer& data) {
  containers_.clear();
  converters_.clear();
  assert(task_type_ == GradientBoostingConfig::TaskType::Classification);
  const auto res = FindTargetValueIndex(data, target_value_name_);
  assert(res.first);
  size_t target_value_index = res.second;
  FitTargetValue(data, target_value_index);
  vector<string> categorical_buffer(data.columns());
  vector<double> numerical_buffer(data.columns());
  const vector<double> target_values = GetTargetValues(data, target_value_name_);
  for (size_t index = 0; index < data.columns(); ++index) {
    if (index == target_value_index_) {
      continue;
    }
    if (data[0][index].IsString()) {
      for (size_t idx = 0; idx < data.rows(); ++idx) {
        categorical_buffer[idx] = data[idx][index].GetString();
      }
      FitCategorical(index, categorical_buffer, target_values);
    }
    if (data[0][index].IsDouble()) {
      for (size_t idx = 0; idx < data.rows(); ++idx) {
        numerical_buffer[idx] = data[idx][index].GetDouble();
      }
      FitNumerical(index, numerical_buffer);
    }
  }
}

void DataTransformer::FitTargetValue(const DataContainer& data,
                                     size_t target_value_index) {
  if (data[0][target_value_index].IsDouble()) {
    return;
  }
  vector<string> categorical_buffer(data.rows());
  for (size_t index = 0; index < data.columns(); ++index) {
    categorical_buffer[index] = data[index][target_value_index].GetString();
  }
  target_values_converter_ =
      std::make_unique<gradient_boosting::categories::CategoricalContainer>(
          categorical_buffer);
}

std::pair<bool, size_t> DataTransformer::FindTargetValueIndex(
    const DataContainer& data,
    const string& target_value_name_) const {
  size_t target_value_index = data.columns();
  size_t matched = 0;
  for (size_t index = 0; index < data.GetNames().size(); ++index) {
    const auto& el = data.GetNames()[index];
    if (el == target_value_name_) {
      target_value_index = index;
      matched++;
    }
  }
  assert(matched <= 1);
  return {target_value_index != data.columns(), target_value_index};
}

vector <double> DataTransformer::GetTargetValues(const DataContainer& data,
                                                 const string& target_value_name_) const {
  const auto result = FindTargetValueIndex(data, target_value_name_);
  if (!result.first) {
    return {};
  }
  size_t target_value_index = result.second;
  vector<double> target_values(data.rows());
  if (data[0][target_value_index].IsDouble()) {
    for (size_t index = 0; index < data.rows(); ++index) {
      target_values[index] = data[index][target_value_index].GetDouble();
    }
  } else {
    for (size_t index = 0; index < data.columns(); ++index) {
      const auto res = target_values_converter_->GetId(data[index][target_value_index].GetString());
      assert(res.first);
      target_values[index] = res.second;
    }
  }
  return target_values;
}

void DataTransformer::FitCategorical(size_t index,
                                     const vector<string>& features,
                                     const vector<double>& target_values) {
  if (task_type_ == GradientBoostingConfig::TaskType::Classification) {
    vector<size_t> classes;
    classes.reserve(target_values.size());
    for (double el : target_values) {
      classes.push_back(static_cast<size_t>(el));
    }
    converters_.insert({index, gradient_boosting::categories::CategoricalConverter(features, classes)});
    FitNumerical(index, converters_.at(index).GetConversionResult());
  }
  assert(false);
}

void DataTransformer::FitNumerical(size_t index,
                                   const vector<double>& features) {
  containers_.insert({index, gradient_boosting::binarization::ThresholdContainer(creators_, features)});
}

InternalDataContainer DataTransformer::Transform(const DataContainer& data) const {
  vector<vector<size_t>> feature_object;
  vector<string> categorical_buffer(data.columns());
  vector<double> numerical_buffer(data.columns());
  for (size_t index = 0; index < data.columns(); ++index) {
    if (data[0][index].IsString()) {
      for (size_t idx = 0; idx < data.rows(); ++idx) {
        categorical_buffer[idx] = data[idx][index].GetString();
      }
      feature_object.push_back(TransformCategorical(index, categorical_buffer));
    }
    if (data[0][index].IsDouble()) {
      for (size_t idx = 0; idx < data.rows(); ++idx) {
        numerical_buffer[idx] = data[idx][index].GetDouble();
      }
      feature_object.push_back(TransformNumerical(index, numerical_buffer));
    }
  }
  const auto target_values = GetTargetValues(data, target_value_name_);
  return InternalDataContainer(feature_object, target_values, data.GetNames());
}

vector<size_t> DataTransformer::TransformCategorical(size_t index,
                                                     const vector<string>& features) const {
  std::vector<double> probs;
  probs.reserve(features.size());
  for (const auto& feature : features) {
    probs.emplace_back(converters_.at(index).Convert(feature));
  }
  return TransformNumerical(index, probs);
}

vector<size_t> DataTransformer::TransformNumerical(size_t index,
                                                   const vector<double>& features) const {
  vector<size_t> result;
  result.reserve(features.size());
  for (const auto& feature : features) {
    result.push_back(containers_.at(index).GetBin(feature));
  }
  return result;
}

}  // namespace data_transformer
}  // namespace gradient_boosting

