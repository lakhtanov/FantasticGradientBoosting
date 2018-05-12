#include "utils/io/SimpleCSVReader.h"

#include <vector>
#include <string>
#include <fstream>

namespace utils {
namespace io {

// TODO(rialeksandrov) move to utils/algorithms ?
std::vector<std::string> split(const std::string &str, char token = ',') {
  std::vector<std::string> result;
  std::string to_push;
  for (const auto& el : str) {
    if (token == el) {
      result.push_back(to_push);
      to_push = "";
    } else {
      to_push += el;
    }
  }
  result.push_back(to_push);
  return result;
}

SimpleCSVReader::SimpleCSVReader(const std::string &file_name)
    : CSVReaderBase(file_name, true) {
}

SimpleCSVReader::SimpleCSVReader(const std::string &file_name, bool header)
    : CSVReaderBase(file_name, header) {
}

data_containers::DataContainer SimpleCSVReader::ReadFile() const {
  std::ifstream in(file_name_);
  std::string str;
  if (header_) {
    std::getline(in, str);
    const std::vector<std::string> names = split(str);
    std::vector<std::vector<std::string>> raw_data;
    while (std::getline(in, str)) {
      raw_data.push_back(split(str));
    }
    return data_containers::DataContainer(names, raw_data);
  } else {
    std::vector<std::vector<std::string>> raw_data;
    while (std::getline(in, str)) {
      raw_data.push_back(split(str));
    }
    return data_containers::DataContainer(raw_data);
  }
}

}  // namespace io
}  // namespace utils
