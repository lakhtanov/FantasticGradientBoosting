#include <iostream>

#include "gradient_boosting/trees/GradientBoostingTree.h"

namespace gradient_boosting {
namespace trees {

using std::vector;

GradientBoostingTree::GradientBoostingTree(
    const gradient_boosting::loss_functions::GradientBoostingLossFunction&
    loss_function)
    : loss_function_(loss_function) {
}

double GradientBoostingTree::Predict(
    const vector<vector<size_t>>& objects_features,
    size_t test_object) const {
  return Predict(objects_features[test_object]);
}

vector<double> GradientBoostingTree::Predict(
    const vector<std::vector<size_t>>& objects_features,
    const vector<size_t>& test_objects,
    ctpl::thread_pool& thread_pool) const {
  vector<double> predictions(test_objects.size());

  auto predict_lambda =
      [
        this,
        &objects_features,
        &test_objects,
        &predictions
      ](size_t, size_t start_index, size_t finish_index) {
        for (size_t local_index = start_index;
             local_index < finish_index;
             ++local_index) {
          predictions[local_index] =
              this->Predict(
                  objects_features,
                  test_objects[local_index]);
        }
      };

  const size_t num_threads = thread_pool.size();
  const size_t task_size = test_objects.size() / (num_threads + 1);
  vector<std::future<void>> prediction_tasks;
  for (size_t test_object_num = task_size;
       test_object_num < test_objects.size();
       test_object_num += task_size) {
    prediction_tasks.push_back(
        thread_pool.push(
            predict_lambda,
            test_object_num,
            std::min(test_object_num + task_size, test_objects.size())));
  }

  for (size_t test_object_num = 0;
       test_object_num < task_size;
       ++test_object_num) {
    predictions[test_object_num] =
        Predict(
            objects_features,
            test_objects[test_object_num]);
  }

  for (auto& prediction_task : prediction_tasks) {
    prediction_task.get();
  }

  return predictions;
}

}  // namespace trees
}  // namespace gradient_boosting
