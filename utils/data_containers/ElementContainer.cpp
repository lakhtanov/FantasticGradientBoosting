#include "ElementContainer.h"

#include <cassert>

namespace utils::data_containers {

    ElementContainer::ElementContainer(const std::string &str)
            : RawData(str), Type(DataType::String) {
        try {
            double el = std::stod(RawData);
            Type = DataType::Double;

        } catch (...) {
            // TODO(rialeksandrov) Debug Output here.
        }
    }

    bool ElementContainer::IsDouble() {
        return Type == DataType::Double;
    }

    bool ElementContainer::IsString() {
        return Type == DataType::String;
    }

    double ElementContainer::GetDouble() {
        assert(IsDouble());
        return std::stod(RawData);
    }

    std::string ElementContainer::GetString() {
        return RawData;
    }
}
