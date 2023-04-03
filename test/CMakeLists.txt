add_executable(test main.cpp)
target_link_libraries(test PUBLIC ${TORCH_LIB} ${NPY_LIB} ${UTILS_LIB} ${MEDIAPIPE_LIBTORCH_LIB})
if (MSVC)
  message("copying dll files")
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:test>)
endif (MSVC)