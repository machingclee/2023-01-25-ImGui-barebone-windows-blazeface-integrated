set(
    SOURCE
    "include/utils/capture_utils.h"
    "include/utils/common.h"
    "include/utils/web_utils.h"
    "include/utils/win_reg.h" 
    "include/utils/npy_utils.h" 
    src/utils/capture_utils.cpp
    src/utils/common.cpp
    src/utils/web_utils.cpp
    src/utils/register_protocol.cpp
    src/utils/npy_utils.cpp
)

add_library(${UTILS_LIB} ${SOURCE})
target_link_libraries(${UTILS_LIB} PUBLIC mongoose ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${WEB_SERVER_LIB} ${CONFIG} ${TORCH_LIB} ${NPY_LIB})
message("mongoose_FOUND ${mongoose_FOUND}")
target_include_directories(${UTILS_LIB} PUBLIC include)