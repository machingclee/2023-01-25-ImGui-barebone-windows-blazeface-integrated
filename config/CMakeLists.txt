set(
  SOURCES 
  "include/config/global.h"
)
add_library(${CONFIG} INTERFACE ${SOURCES})
target_include_directories(${CONFIG} INTERFACE include)