#include "gradient_boosting/binarization/BinCreator.h"

namespace gradient_boosting {
namespace binarization {

vector<double> BinCreator::CreateBins(
    BinCreator* bin_creator,
    const std::vector<double>& features,
    size_t num_bins) {
  return bin_creator->CreateBins_(features, num_bins);
}

}  // namespace binarization
}  // namespace gradient_boosting
