set(CMAKE_CXX_STANDARD 17)
set(
    sources 
    src/mediapipe_libtorch/visualize.h
    src/mediapipe_libtorch/visualize.cpp
    src/mediapipe_libtorch/facial_landmark.h
    src/mediapipe_libtorch/facial_landmark.cpp
    src/mediapipe_libtorch/blazeface_landmark.h
    src/mediapipe_libtorch/blazeface_landmark.cpp
    src/mediapipe_libtorch/blazeface.h
    src/mediapipe_libtorch/blazeface.cpp
    src/mediapipe_libtorch/blazebase.h
    src/mediapipe_libtorch/blazebase.cpp
)

add_library(${MEDIAPIPE_LIBTORCH_LIB} ${sources})
target_link_libraries(${MEDIAPIPE_LIBTORCH_LIB} PUBLIC ${TORCH_LIB} ${OpenCV_LIBS} ${UTILS_LIB})
target_include_directories(${MEDIAPIPE_LIBTORCH_LIB} PUBLIC src)
target_precompile_headers(${MEDIAPIPE_LIBTORCH_LIB} PUBLIC src/mediapipe_libtorch/pch.h)