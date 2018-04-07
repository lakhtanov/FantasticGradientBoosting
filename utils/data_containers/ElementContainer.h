#ifndef FANTASTICGRADIENTBOOSTING_ELEMENTCONTAINER_H
#define FANTASTICGRADIENTBOOSTING_ELEMENTCONTAINER_H

#include <string>

namespace utils {
    namespace  data_containers {

        class ElementContainer {
        public:
            enum class DataType {Double, String, LongInteger};
            explicit ElementContainer (const std::string& str);
            bool IsDouble();
            bool IsString();
            double GetDouble();
            std::string GetString();

        private:
            std::string RawData;
            DataType Type;
        };
    }
}


#endif //FANTASTICGRADIENTBOOSTING_ELEMENTCONTAINER_H
