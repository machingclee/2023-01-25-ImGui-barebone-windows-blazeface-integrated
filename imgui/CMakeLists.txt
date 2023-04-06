set(
  SOURCES
  "include/imgui/font.h"
  "include/imgui/global.h"
  "include/imgui/iconcpp.h"
  "include/imgui/icons.h"
  "include/imgui/imconfig.h"
  "include/imgui/imgui_impl_dx9.h"
  "include/imgui/imgui_impl_win32.h"
  "include/imgui/imgui_internal.h"
  "include/imgui/imgui.h"
  "include/imgui/imguipp.h"
  "include/imgui/imstb_rectpack.h"
  "include/imgui/imstb_textedit.h"
  "include/imgui/imstb_truetype.h"
  "include/imgui/menu.h"
  "include/imgui/settings.h"
  "include/imgui/text_editor.h"
  "include/imgui/loader.h"
  src/imgui_demo.cpp
  src/imgui_draw.cpp
  src/imgui_impl_dx9.cpp
  src/imgui_impl_win32.cpp
  src/imgui_widgets.cpp
  src/imgui.cpp
  src/imguipp.cpp
  src/menu.cpp
  src/text_editor.cpp
)


add_library(${IMGUI_LIB} ${SOURCES})
target_link_libraries(${IMGUI_LIB} PRIVATE ${TORCH_LIBRARIES} ${OPENCV_AND_TORCH_LIB} ${WEB_SERVER_LIB} ${CONFIG} ${UTILS_LIB})
target_include_directories(${IMGUI_LIB} PUBLIC include)