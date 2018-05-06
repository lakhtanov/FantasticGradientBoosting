#ifndef GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYABSOLUTEVALUE_H_
#define GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYABSOLUTEVALUE_H_

#include <vector>

#include "gradient_boosting/binarization/BinCreator.h"

namespace gradient_boosting {
namespace binarization {

using std::vector;

// TODO(lakhtanov): class for creating bins for learning based on absolute
// values of the learned data.
class BinCreatorByAbsoluteValue : public BinCreator {
 protected:
  vector<double> CreateBins_(
      const vector<double>& features,
      size_t num_bins) const override;
};

}  // namespace binarization
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYABSOLUTEVALUE_H_
