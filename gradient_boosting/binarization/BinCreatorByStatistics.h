#ifndef GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYSTATISTICS_H_
#define GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYSTATISTICS_H_

#include <vector>

#include "gradient_boosting/binarization/BinCreator.h"

namespace gradient_boosting {
namespace binarization {

using std::vector;

// TODO(lakhtanov): class for creating bins for learning based on relative order
// statistics of the learned data.
class BinCreatorByStatistics : public BinCreator {
 protected:
  vector<double> CreateBins_(
      const vector<double>& features,
      size_t num_bins) const override;
};

}  // namespace binarization
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYSTATISTICS_H_
