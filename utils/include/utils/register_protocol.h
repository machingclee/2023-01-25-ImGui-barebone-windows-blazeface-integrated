#pragma once
#include "WinReg.h"
#include <string>

namespace RegisterProtocol {
inline std::wstring to_wide_string(const std::string& input);
inline std::string to_byte_string(const std::wstring& input);
void register_protocol(std::string app_path);
} // namespace RegisterProtocol