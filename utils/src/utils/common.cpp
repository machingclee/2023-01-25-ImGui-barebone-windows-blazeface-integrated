#include "utils/common.h"

std::vector<std::string> split(std::string text, std::string space_delimiter) {
    bool hasNext = false;
    std::vector<std::string> words{};
    do {
        size_t nextPos = text.find(space_delimiter);
        hasNext = (nextPos != std::string::npos);
        words.push_back(text.substr(0, nextPos));
        text.erase(0, nextPos + space_delimiter.length());
    } while (hasNext);
    return words;
}
