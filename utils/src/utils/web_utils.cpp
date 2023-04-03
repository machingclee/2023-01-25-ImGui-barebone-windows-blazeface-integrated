#include "utils/web_utils.h"
#include "config/global.h"
#include <iostream>
#include <stdlib.h>
#include <signal.h>
#include "mongoose/Server.h"
#include "mongoose/WebController.h"

using namespace Mongoose;

namespace WebUtils {

class MLController : public WebController {
public:
    void start_calibration(Request& request, StreamResponse& response) {
        response << "Hello " << htmlEntities(request.get("name", "... what's your name ?")) << endl;
    }
    void collect_data(Request& request, StreamResponse& response) {
        response << "Hello " << htmlEntities(request.get("name", "... what's your name ?")) << endl;
    }

    void setup() {
        addRoute("GET", "/start-calibration", MLController, start_calibration);
        addRoute("POST", "collection-click-data", MLController, collect_data)
    }
};

static std::atomic<bool> atomic_stop_web_service_flag = false;
static std::thread web_server_thread;
int start_web_server_thread() {
    auto start_web_server = []() {
        MLController myController;
        Server server(8080);
        server.registerController(&myController);
        server.start();
        while (!atomic_stop_web_service_flag) {
            Sleep(10);
        }
    };
    web_server_thread = std::thread(start_web_server);
    return 0;
};

int stop_web_server_thread() {
    atomic_stop_web_service_flag = true;
    web_server_thread.join();
    atomic_stop_web_service_flag = false;
    return 0;
}
} // namespace WebUtils