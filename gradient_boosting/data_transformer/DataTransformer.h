#ifndef GRADIENT_BOOSTING_DATA_TRANSFORMATOR_DATATRANSFORMER_H_
#define GRADIENT_BOOSTING_DATA_TRANSFORMATOR_DATATRANSFORMER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gradient_boosting/binarization/ThresholdContainer.h"
#include "gradient_boosting/binarization/ThresholdCreator.h"
#include "gradient_boosting/categories/CategoricalConverter.h"
#include "gradient_boosting/config/GradientBoostingConfig.h"
#include "gradient_boosting/internal_data_container/InternalDataContainer.h"
#include "utils/data_containers/DataContainer.h"


namespace gradient_boosting {
namespace data_transformer {

class DataTransformer {
 public:
  explicit DataTransformer(
      const gradient_boosting::config::GradientBoostingConfig& config);

  gradient_boosting::internal_data_container::InternalDataContainer FitAndTransform(
      const utils::data_containers::DataContainer& data);

  void Fit(const utils::data_containers::DataContainer& data);

  gradient_boosting::internal_data_container::InternalDataContainer Transform(
      const utils::data_containers::DataContainer& data) const;

 private:
  std::pair<bool, size_t> FindTargetValueIndex(
      const utils::data_containers::DataContainer& data,
      const std::string& target_value_name_) const;
  std::vector <double> GetTargetValues(
      const utils::data_containers::DataContainer& data,
      const std::string& target_value_name_) const;

  void FitTargetValue(
      const utils::data_containers::DataContainer& data,
      size_t target_value_index);
  void FitCategorical(
      size_t index,
      const std::vector<std::string>& features,
      const std::vector<double>& target_values);
  void FitNumerical(
      size_t index,
      const std::vector<double>& features);

  std::vector<size_t> TransformCategorical(
      size_t index,
      const std::vector<std::string>& features) const;
  std::vector<size_t> TransformNumerical(
      size_t index,
      const std::vector<double>& features) const;

  std::unordered_map<size_t, gradient_boosting::binarization::ThresholdContainer> containers_;
  std::unordered_map<size_t, gradient_boosting::categories::CategoricalConverter> converters_;
  std::vector<std::unique_ptr<gradient_boosting::binarization::ThresholdCreator>> creators_;
  std::unique_ptr<gradient_boosting::categories::CategoricalContainer> target_values_converter_;
  const std::string target_value_name_;
  size_t target_value_index_;
  gradient_boosting::config::GradientBoostingConfig::TaskType task_type_;
};

}  // namespace data_transformer
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_DATA_TRANSFORMATOR_DATATRANSFORMER_H_

