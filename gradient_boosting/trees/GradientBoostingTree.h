#ifndef GRADIENT_BOOSTING_TREES_GRADIENTBOOSTINGTREE_H_
#define GRADIENT_BOOSTING_TREES_GRADIENTBOOSTINGTREE_H_

#include <vector>

#include "gradient_boosting/loss_functions/GradientBoostingLossFunction.h"
#include "third_party/ctpl/ctpl_stl.h"

namespace gradient_boosting {
namespace trees {

class GradientBoostingTree {
 public:
  explicit GradientBoostingTree(
      const gradient_boosting::loss_functions::GradientBoostingLossFunction&
      loss_function);
  virtual ~GradientBoostingTree() = default;
  virtual void Fit(
      const std::vector<std::vector<size_t>>& features_objects,
      const std::vector<std::vector<size_t>>& objects_features,
      const std::vector<double>& target_values,
      const std::vector<size_t>& train_objects,
      const std::vector<size_t>& train_features,
      ctpl::thread_pool& thread_pool) = 0;
  double Predict(
      const std::vector<std::vector<size_t>>& objects_features,
      size_t test_object) const;
  std::vector<double> Predict(
      const std::vector<std::vector<size_t>>& objects_features,
      const std::vector<size_t>& test_objects,
      ctpl::thread_pool& thread_pool) const;
  virtual double Predict(
      const std::vector<size_t>& test_object_features) const = 0;

 protected:
  const gradient_boosting::loss_functions::GradientBoostingLossFunction&
      loss_function_;
};

}  // namespace trees
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_TREES_GRADIENTBOOSTINGTREE_H_
