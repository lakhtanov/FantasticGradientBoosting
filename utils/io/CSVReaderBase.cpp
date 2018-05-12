#include "utils/io/CSVReaderBase.h"

namespace utils {
namespace io {

CSVReaderBase::CSVReaderBase(const std::string& file_name, bool header)
    : file_name_(file_name)
    , header_(header) {
}

}  // namespace io
}  // namespace utils
