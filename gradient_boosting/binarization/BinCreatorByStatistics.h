#ifndef GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYSTATISTICS_H_
#define GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYSTATISTICS_H_

#include "gradient_boosting/binarization/BinCreator.h"

namespace gradient_boosting {
namespace binarization {

// TODO(lakhtanov): class for creating bins for learning based on relative order
// statistics of the learned data.
class BinCreatorByStatistics : public BinCreator {
 public:
 private:
};

}  // namespace binarization
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_BINARIZATION_BINCREATORBYSTATISTICS_H_
