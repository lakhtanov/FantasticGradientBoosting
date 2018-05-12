#ifndef GRADIENT_BOOSTING_TREES_GRADIENTBOOSTINGTREEOBLIVIOUS_H_
#define GRADIENT_BOOSTING_TREES_GRADIENTBOOSTINGTREEOBLIVIOUS_H_

#include <utility>
#include <vector>

#include "gradient_boosting/loss_functions/GradientBoostingSplitInfo.h"
#include "gradient_boosting/trees/GradientBoostingTree.h"

namespace gradient_boosting {
namespace trees {

class GradientBoostingTreeOblivious : public GradientBoostingTree {
 public:
  explicit GradientBoostingTreeOblivious(
      const gradient_boosting::loss_functions::GradientBoostingLossFunction&
      loss_function,
      size_t height);
  void Fit(
      const std::vector<std::vector<size_t>>& features_objects,
      const std::vector<std::vector<size_t>>& objects_features,
      const std::vector<double>& target_values,
      const std::vector<size_t>& train_objects,
      const std::vector<size_t>& train_features,
      ctpl::thread_pool& thread_pool) override;
  double Predict(
      const std::vector<size_t>& test_object_features) const override;

 private:
  std::pair<size_t, std::pair<double, std::vector<
      gradient_boosting::loss_functions::GradientBoostingSplitInfo>>>
      GetBestSplit(
          const std::vector<std::vector<size_t>>& features_objects,
          const std::vector<std::vector<size_t>>& objects_in_nodes,
          size_t train_feature) const;

  size_t GetLeftChildNum(size_t node_num) const;
  size_t GetRightChildNum(size_t node_num) const;

  std::vector<size_t> GetLeftSplit(
      const std::vector<std::vector<size_t>>& features_objects,
      size_t feature,
      size_t feature_split_value,
      const std::vector<size_t>& objects) const;
  std::vector<size_t> GetRightSplit(
      const std::vector<std::vector<size_t>>& features_objects,
      size_t feature,
      size_t feature_split_value,
      const std::vector<size_t>& objects) const;

  std::vector<size_t> features_;
  std::vector<size_t> features_split_values_;
  size_t height_;
  std::vector<double> nodes_values_;
};

}  // namespace trees
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_TREES_GRADIENTBOOSTINGTREEOBLIVIOUS_H_
