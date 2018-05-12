#ifndef GRADIENT_BOOSTING_INTERNAL_DATA_CONTAINER_INTERNALDATACONTAINER_H_
#define GRADIENT_BOOSTING_INTERNAL_DATA_CONTAINER_INTERNALDATACONTAINER_H_

#include <string>
#include <vector>

namespace gradient_boosting {
namespace internal_data_container {

class InternalDataContainer {
 public:
  InternalDataContainer(const std::vector<std::vector<size_t>>& features_objects,
                        const std::vector<std::string>& features_names);

  InternalDataContainer(const std::vector<std::vector<size_t>>& features_objects,
                        const std::vector<double>& target_values);

  InternalDataContainer(const std::vector<std::vector<size_t>>& features_objects,
                        const std::vector<double>& target_values,
                        const std::vector<std::string>& features_names,
                        const std::vector<std::string>& id_names);

  size_t GetNumberOfObject() const;
  size_t GetNumberOfFeatures() const;

  const std::vector<std::vector<size_t>>& GetFeaturesObjects() const;
  const std::vector<std::vector<size_t>>& GetObjectsFeatures() const;

  const std::vector<double>& GetTargetValues() const;
  const std::vector<std::string>& GetFeaturesNames() const;
  const std::vector<std::string>& GetIdNames() const;

 private:
  const std::vector<std::vector<size_t>> features_objects_;
  const std::vector<std::string> features_names_;
  std::vector<std::vector<size_t>> objects_features_;
  std::vector<double> target_values_;
  const std::vector<std::string> id_names_;
};

}  // namespace internal_data_container
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_INTERNAL_DATA_CONTAINER_INTERNALDATACONTAINER_H_
