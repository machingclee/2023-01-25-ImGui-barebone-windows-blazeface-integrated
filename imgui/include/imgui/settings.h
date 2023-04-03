#pragma once
#include <iostream>
#include <string>
#include <vector>

namespace Settings {
// internal linkage, each translation unit get different copy, non-global scope
static int Tab = 0;
} // namespace Settings