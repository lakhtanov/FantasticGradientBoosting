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
  vector<std::future<double>> predictions_futures(test_objects.size());

  const size_t num_threads = thread_pool.size();
  for (size_t test_object_num = 0;
       test_object_num < test_objects.size();
       ++test_object_num) {
    if (test_object_num % (num_threads + 1) == 0) {  // run in current thread
      continue;
    }
    predictions_futures[test_object_num] =
        thread_pool.push(
            [&](int) {
              return Predict(
                  objects_features,
                  test_objects[test_object_num]);
            });
  }

  for (size_t test_object_num = 0;
       test_object_num < test_objects.size();
       test_object_num += num_threads + 1) {
    predictions[test_object_num] =
        Predict(
            objects_features,
            test_objects[test_object_num]);
  }

  for (size_t test_object_num = 0;
       test_object_num < test_objects.size();
       ++test_object_num) {
    if (test_object_num % (num_threads + 1) == 0) {
      continue;
    }
    predictions[test_object_num] = predictions_futures[test_object_num].get();
  }

  return predictions;
}

}  // namespace trees
}  // namespace gradient_boosting
