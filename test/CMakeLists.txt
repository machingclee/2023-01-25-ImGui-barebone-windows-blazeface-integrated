add_executable(${TEST} main.cpp)
target_link_libraries(test PUBLIC ${TORCH_LIB} ${NPY_LIB} ${UTILS_LIB} ${MEDIAPIPE_LIBTORCH_LIB})
if (MSVC)
  message("copying dll and pdb files")
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll" )
  file(GLOB TORCH_PDBS "${TORCH_INSTALL_PREFIX}/lib/*.pdb")
  set(TORCH_FILES ${TORCH_DLLS} ${TORCH_PDBS})
  add_custom_command(TARGET test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_FILES}
                     $<TARGET_FILE_DIR:test>)
endif (MSVC)
