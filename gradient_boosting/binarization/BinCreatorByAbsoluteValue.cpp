#include <algorithm>
#include <vector>

#include "gradient_boosting/binarization/BinCreatorByAbsoluteValue.h"

namespace gradient_boosting {
namespace binarization {

using std::vector;

vector<double> BinCreatorByAbsoluteValue::CreateBins_(
    const vector<double>& features,
    size_t num_bins) const {
  if (features.empty() || !num_bins) {
    return vector<double>();
  }

  const double min_element =
      *std::min_element(features.begin(), features.end());
  const double max_element =
      *std::max_element(features.begin(), features.end());
  vector<double> bins(num_bins);
  if (num_bins == 1) {
    bins[0] = {(min_element + max_element) / 2};
  } else {
    const double delta = (max_element - min_element) / (num_bins - 1);
    bins[0] = min_element;
    for (size_t i = 1; i < num_bins; ++i) {
      bins[i] = bins[i - 1] + delta;
    }
  }

  return bins;
}

}  // namespace binarization
}  // namespace gradient_boosting
