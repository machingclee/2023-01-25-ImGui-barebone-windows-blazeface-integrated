set(
  SOURCE_FILES 
  TcpListener.cpp 
  MultiClientChat.cpp  
  WebServer.cpp
)

add_library(${WEB_SERVER_LIB} ${SOURCE_FILES})

if(WIN32)
  target_link_libraries(${WEB_SERVER_LIB} PUBLIC wsock32 ws2_32)
endif()

target_include_directories(${WEB_SERVER_LIB} PUBLIC ../include)
target_link_libraries(${WEB_SERVER_LIB} PRIVATE ${UTILS_LIB})