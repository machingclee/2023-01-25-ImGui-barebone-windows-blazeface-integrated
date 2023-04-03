#pragma once

#include <string>
#include <vector>
#include <iostream>

std::vector<std::string> split(std::string text, std::string space_delimiter = " ");

template <class F, class First, class... Rest>
void do_for(F f, First first, Rest... rest) {
    f(first);
    do_for(f, rest...);
}
template <class F>

void do_for(F f) {
    std::cout << "\n";
}

template <class... Args>
void print(Args... args) {
    do_for([](auto& arg) {
        std::cout << arg;
    },
           args...);
}