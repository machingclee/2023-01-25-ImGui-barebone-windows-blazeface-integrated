#include "web_server/WebServer.h"

int main() {
    WebServer webServer("0.0.0.0", 8080);
    int webServerInitResult = webServer.init();
    if (webServerInitResult != 0) {
        return webServerInitResult;
    }
    webServer.run();
    system("pause");
}
