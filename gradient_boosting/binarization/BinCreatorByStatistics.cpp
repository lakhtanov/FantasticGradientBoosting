#include <algorithm>
#include <vector>

#include "gradient_boosting/binarization/BinCreatorByStatistics.h"

namespace gradient_boosting {
namespace binarization {

using std::vector;

vector<double> BinCreatorByStatistics::CreateBins_(
    const vector<double>& features,
    size_t num_bins) const {
  if (features.empty() || !num_bins) {
    return vector<double>();
  }

  vector<double> sorted_features = features;
  std::sort(sorted_features.begin(), sorted_features.end());

  vector<double> bins(std::min(num_bins, sorted_features.size()));
  size_t delta = sorted_features.size() / bins.size();
  for (size_t i = 0; i < bins.size(); ++i) {
    bins[i] = sorted_features[i * delta];
  }

  return bins;
}

}  // namespace binarization
}  // namespace gradient_boosting
