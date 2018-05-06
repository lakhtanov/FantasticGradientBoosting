#ifndef GRADIENT_BOOSTING_BINARIZATION_BINCREATOR_H_
#define GRADIENT_BOOSTING_BINARIZATION_BINCREATOR_H_

#include <vector>

namespace gradient_boosting {
namespace binarization {

using std::vector;

// TODO(lakhtanov): abstract class for creating bins.
class BinCreator {
 public:
  virtual ~BinCreator() = default;
  static vector<double> CreateBins(
      BinCreator* bin_creator,
      const vector<double>& features,
      size_t num_bins);
 protected:
  virtual vector<double> CreateBins_(
      const vector<double>& features,
      size_t num_bins) const = 0;
};

}  // namespace binarization
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_BINARIZATION_BINCREATOR_H_
