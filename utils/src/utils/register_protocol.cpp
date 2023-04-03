
#include "utils/win_reg.h"
#include <string>
#include <codecvt>

using namespace winreg;

namespace RegisterProtocol {

inline std::wstring to_wide_string(const std::string& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(input);
}

inline std::string to_byte_string(const std::wstring& input) {
    // std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(input);
}

void register_protocol(std::string exe_path) {
    std::wstring w_app_path = to_wide_string(exe_path);
    // RegCreateKeyEx(HKEY_CURRENT_USER, "Software\\Classes\\", 0, KEY_CREATE_SUB_KEY);
    RegKey key;
    key.Create(HKEY_CURRENT_USER, L"Software\\Classes\\eyecatcher", KEY_CREATE_SUB_KEY);
    key.Open(HKEY_CURRENT_USER, L"Software\\Classes\\eyecatcher", KEY_SET_VALUE);
    key.SetStringValue(L"URL Protocol", L"");

    key.Create(HKEY_CURRENT_USER, L"Software\\Classes\\eyecatcher\\shell\\open\\command", KEY_CREATE_SUB_KEY);
    key.Open(HKEY_CURRENT_USER, L"Software\\Classes\\eyecatcher\\shell\\open\\command", KEY_SET_VALUE);
    key.SetStringValue(L"", w_app_path);
}
} // namespace RegisterProtocol