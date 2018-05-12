#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include <iostream>

#include "gradient_boosting/trees/GradientBoostingTreeOblivious.h"

namespace gradient_boosting {
namespace trees {

using gradient_boosting::loss_functions::GradientBoostingLossFunction;
using gradient_boosting::loss_functions::GradientBoostingSplitInfo;
using std::pair;
using std::vector;

GradientBoostingTreeOblivious::GradientBoostingTreeOblivious(
    const GradientBoostingLossFunction& loss_function,
    size_t height)
    : GradientBoostingTree(loss_function)
    , height_(height) {
  features_.resize(height_);
  features_split_values_.resize(height_);
  nodes_values_.resize((1 << (height_ + 1)) - 1, 0);
}

inline pair<size_t, pair<double, vector<GradientBoostingSplitInfo>>>
GradientBoostingTreeOblivious::GetBestSplit(
    const vector<vector<size_t>>& features_objects,
    const vector<vector<size_t>>& objects_in_nodes,
    size_t train_feature) const {
  auto loss_function_ptr = loss_function_.Clone();
  size_t num_feature_values = 0;
  for (const auto& node : objects_in_nodes) {
    for (size_t object : node) {
      num_feature_values =
          std::max(
              num_feature_values,
              features_objects[train_feature][object] + 1);
    }
  }
  vector<vector<GradientBoostingSplitInfo>> feature_nodes_split_infos(
      num_feature_values,
      vector<GradientBoostingSplitInfo>(objects_in_nodes.size()));
  for (size_t node = 0; node < objects_in_nodes.size(); ++node) {
    loss_function_ptr->Configure(
        train_feature,
        num_feature_values,
        objects_in_nodes[node]);
    for (size_t feature_split_value = 0;
         feature_split_value < num_feature_values;
         ++feature_split_value) {
      feature_nodes_split_infos[feature_split_value][node] =
          loss_function_ptr->GetLoss(feature_split_value);
    }
  }
  auto loss_lambda =
      [&](size_t feature_split_value) {
        return
            std::accumulate(
                feature_nodes_split_infos[feature_split_value].begin(),
                feature_nodes_split_infos[feature_split_value].end(),
                0.0,
                [](double loss, const GradientBoostingSplitInfo& info) {
                    return loss + info.loss;
                });
      };

  size_t best_feature_split_value = 0;
  double best_feature_split_value_loss = loss_lambda(best_feature_split_value);

  for (size_t feature_split_value = 1;
       feature_split_value < num_feature_values;
       ++feature_split_value) {
    auto feature_split_value_loss = loss_lambda(feature_split_value);
    if (feature_split_value_loss < best_feature_split_value_loss) {
      best_feature_split_value = feature_split_value;
      best_feature_split_value_loss = feature_split_value_loss;
    }
  }

  return {
    best_feature_split_value,
        {
          best_feature_split_value_loss,
          feature_nodes_split_infos[best_feature_split_value]
        }
  };
}

void GradientBoostingTreeOblivious::Fit(
    const vector<vector<size_t>>& features_objects,
    const vector<vector<size_t>>&,
    const vector<double>&,
    const vector<size_t>& train_objects,
    const vector<size_t>& train_features,
    ctpl::thread_pool& thread_pool) {
  vector<vector<size_t>> objects_in_nodes{{train_objects}};

  const size_t num_threads = thread_pool.size();
  for (size_t height = 0; height < height_; ++height) {
    vector<
        pair<size_t, pair<double, vector<GradientBoostingSplitInfo>>>
    > features_best_splits(train_features.size());
    vector<std::future<
        pair<size_t, pair<double, vector<GradientBoostingSplitInfo>>>
    >> features_best_splits_futures(train_features.size());

    for (size_t train_feature_num = 0;
         train_feature_num < train_features.size();
         ++train_feature_num) {
      if (train_feature_num % (num_threads + 1) == 0) {
        continue;
      }

      features_best_splits_futures[train_feature_num] =
          thread_pool.push(
              [
                this,
                &features_objects,
                &objects_in_nodes,
                &train_features,
                train_feature_num
              ](int) {
                  return this->GetBestSplit(
                      features_objects,
                      objects_in_nodes,
                      train_features[train_feature_num]);
              });
    }
    for (size_t train_feature_num = 0;
         train_feature_num < train_features.size();
         train_feature_num += num_threads + 1) {
      const size_t train_feature = train_features[train_feature_num];
      features_best_splits[train_feature_num] = GetBestSplit(
          features_objects,
          objects_in_nodes,
          train_feature);
    }
    for (size_t train_feature_num = 0;
         train_feature_num < train_features.size();
         ++train_feature_num) {
      if (train_feature_num % (num_threads + 1) == 0) {
        continue;
      }

      features_best_splits[train_feature_num] =
          features_best_splits_futures[train_feature_num].get();
    }

    size_t best_feature = 0;
    for (size_t train_feature_num = 0;
         train_feature_num < train_features.size();
         ++train_feature_num) {
      if (features_best_splits[train_feature_num].second.first
          < features_best_splits[best_feature].second.first) {
        best_feature = train_feature_num;
      }
    }

    features_[height] = train_features[best_feature];
    features_split_values_[height] = features_best_splits[best_feature].first;
    const size_t first_height_num = (1 << height) - 1;
    const auto& best_feature_split_infos =
        features_best_splits[best_feature].second.second;
    for (size_t node = 0; node < best_feature_split_infos.size(); ++node) {
      nodes_values_[GetLeftChildNum(first_height_num + node)] =
          best_feature_split_infos[node].left_split_value;
      nodes_values_[GetRightChildNum(first_height_num + node)] =
          best_feature_split_infos[node].right_split_value;
    }

    vector<vector<size_t>> objects_in_nodes_next_level;
    for (size_t node = 0; node < objects_in_nodes.size(); ++node) {
      objects_in_nodes_next_level.push_back(
          GetLeftSplit(
              features_objects,
              features_[height],
              features_split_values_[height],
              objects_in_nodes[node]));
      objects_in_nodes_next_level.push_back(
          GetRightSplit(
              features_objects,
              features_[height],
              features_split_values_[height],
              objects_in_nodes[node]));
    }
    swap(objects_in_nodes, objects_in_nodes_next_level);
  }
}

inline double GradientBoostingTreeOblivious::Predict(
    const vector<size_t>& test_object_features) const {
  size_t node_num = 0;
  for (size_t h = 0; h < height_; ++h) {
    if (test_object_features[features_[h]] <= features_split_values_[h]) {
      node_num = GetLeftChildNum(node_num);
    } else {
      node_num = GetRightChildNum(node_num);
    }
  }

  return nodes_values_[node_num];
}

inline size_t GradientBoostingTreeOblivious::GetLeftChildNum(size_t node_num) const {
  return (node_num << 1) + 1;
}

inline size_t GradientBoostingTreeOblivious::GetRightChildNum(size_t node_num) const {
  return (node_num << 1) + 2;
}

inline vector<size_t> GradientBoostingTreeOblivious::GetLeftSplit(
    const vector<vector<size_t>>& features_objects,
    size_t feature,
    size_t feature_split_value,
    const vector<size_t>& objects) const {
  vector<size_t> left_split;
  for (size_t object : objects) {
    if (features_objects[feature][object] <= feature_split_value) {
      left_split.push_back(object);
    }
  }

  return left_split;
}

inline vector<size_t> GradientBoostingTreeOblivious::GetRightSplit(
    const vector<vector<size_t>>& features_objects,
    size_t feature,
    size_t feature_split_value,
    const vector<size_t>& objects) const {
  vector<size_t> right_split;
  for (size_t object : objects) {
    if (!(features_objects[feature][object] <= feature_split_value)) {
      right_split.push_back(object);
    }
  }

  return right_split;
}

}  // namespace trees
}  // namespace gradient_boosting
