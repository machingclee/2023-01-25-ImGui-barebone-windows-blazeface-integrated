set(
    sources 
    include/npy.hpp
)

add_library(${NPY_LIB} INTERFACE ${sources})
target_include_directories(${NPY_LIB} INTERFACE include)