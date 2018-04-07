#include <iostream>

#include "utils/data_containers/ElementContainer.h"

int main() {
    utils::data_containers::ElementContainer el1("abcb 12123");
    utils::data_containers::ElementContainer el2("12123.12 abcd");
    std::cout << std::fixed;
    std::cout << el1.IsDouble() << " " << el2.IsDouble() << " "<< el2.GetDouble() << std::endl;
}