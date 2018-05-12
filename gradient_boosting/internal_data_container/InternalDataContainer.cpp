#include <string>
#include <vector>

#include "gradient_boosting/internal_data_container/InternalDataContainer.h"

namespace gradient_boosting {
namespace internal_data_container {

using std::string;
using std::vector;

InternalDataContainer::InternalDataContainer(
    const vector<vector<size_t>>& features_objects,
    const vector<double>& target_values)
    : InternalDataContainer(features_objects, target_values, {}, {}) {
}

InternalDataContainer::InternalDataContainer(
    const std::vector<std::vector<size_t>>& features_objects,
    const std::vector<std::string>& features_names)
    : InternalDataContainer(features_objects, {}, features_names, {}) {
}

InternalDataContainer::InternalDataContainer(
    const vector<vector<size_t>>& features_objects,
    const vector<double>& target_values,
    const vector<string>& features_names,
    const vector<string>& id_names)
    : features_objects_(features_objects)
    , features_names_(features_names)
    , target_values_(target_values)
    , id_names_(id_names) {
  objects_features_.assign(
      features_objects.front().size(), vector<size_t>(features_objects.size()));
  for (size_t index = 0; index < features_objects.size(); ++index) {
    for (size_t jindex = 0; jindex < features_objects[index].size(); ++jindex) {
      objects_features_[jindex][index] = features_objects[index][jindex];
    }
  }
}

size_t InternalDataContainer::GetNumberOfObject() const {
  return objects_features_.size();
}

size_t InternalDataContainer::GetNumberOfFeatures() const {
  return features_objects_.size();
}

const vector<vector<size_t>>&
InternalDataContainer::GetFeaturesObjects() const {
  return features_objects_;
}

const vector<vector<size_t>>&
InternalDataContainer::GetObjectsFeatures() const {
  return objects_features_;
}

const vector<double>& InternalDataContainer::GetTargetValues() const {
  return target_values_;
}

const vector<string>& InternalDataContainer::GetFeaturesNames() const {
  return features_names_;
}

const vector<string>& InternalDataContainer::GetIdNames() const {
  return id_names_;
}

}  // namespace internal_data_container
}  // namespace gradient_boosting
