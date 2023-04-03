cmake_minimum_required(VERSION 3.0.0)
project(EyeCatching VERSION 0.1.0)
set(CMAKE_BUILD_TYPE Debug)
set(WEB_SERVER_LIB web_server_lib)
set(UTILS_LIB utils)
set(IMGUI_LIB imgui_lib)
set(NPY_LIB libnpy)
set(CONFIG config)
set(MEDIAPIPE_LIBTORCH_LIB mediapipe_libtorch)
set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_COMPILER "-vc143")
set(BOOST_ROOT "C:\\Users\\user\\Repos\\C++Libraries\\boost_1_80_0")
set(BOOST_INCLUDEDIR "${BOOST_ROOT}\\boost")
set(BOOST_LIBRARYDIR "${BOOST_ROOT}\\stage\\lib")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
# list(APPEND CMAKE_C_STANDARD_LIBRARIES wsock32.lib ws2_32.lib)
list(APPEND CMAKE_C_STANDARD_LIBRARIES)

include(cmake/link_opencv_and_torch.cmake)

set(TORCH_LIB ${TORCH_LIBRARIES})

add_executable(EyeCatching main.cpp)

add_subdirectory(mongoose)
add_subdirectory(config)
add_subdirectory(WebServer)
add_subdirectory(utils)
add_subdirectory(imgui)
add_subdirectory(libnpy)
add_subdirectory(test)
add_subdirectory(mediapipe_libtorch)

if (MSVC)
  message("copying dll files")
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET EyeCatching
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:EyeCatching>)
endif (MSVC)

target_link_libraries(EyeCatching PUBLIC ${IMGUI_LIB} ${UTILS_LIB} ${CONFIG})
target_compile_features(${UTILS_LIB} PUBLIC cxx_std_20)
target_compile_features(${WEB_SERVER_LIB} PUBLIC cxx_std_20)