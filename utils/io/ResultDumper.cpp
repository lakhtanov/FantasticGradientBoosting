#include <fstream>
#include <string>
#include <vector>

#include "utils/io/ResultDumper.h"

namespace utils {
namespace io {

using std::string;
using std::vector;

ResultDumper::ResultDumper(
    const string& file_name,
    const string& id_name,
    const string& target_value_name)
    : out_(file_name)
    , file_name_(file_name)
    , id_name_(id_name)
    , target_value_name_(target_value_name) {
  out_ << id_name_ << "," << target_value_name << "\n";
  out_ << std::fixed;
  out_.precision(6);
}

void ResultDumper::AddResult(std::string id, double result) {
  out_ << id << "," << result << "\n";
}

void ResultDumper::AddResults(
    const vector<string>& ids, const vector<double>& results) {
  for (size_t index = 0; index < ids.size(); ++index) {
    AddResult(ids[index], results[index]);
  }
  out_.flush();
}

}  // namespace io
}  // namespace utils
