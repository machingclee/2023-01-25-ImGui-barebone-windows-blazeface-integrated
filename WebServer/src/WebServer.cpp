#include "web_server/WebServer.h"
#include "utils/common.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void WebServer::onClientConnected(int clientSocket){

};
void WebServer::onClientDisconnected(int clientSocket){

};
void WebServer::onMessageReceived(int currSock, char* buffer, int bytesReceived) {
    std::string clientMessage{buffer};
    std::vector<std::string> parsed = split(clientMessage, " ");

    std::string method = parsed[0];
    std::string route = parsed[1];

    if (method == "POST") {
        if (route == "/collection_for_calibration") {
            std::string content{"I will store data for calibration ML work"};
            std::ostringstream ss;
            ss << "HTTP/1.1 200 OK\r\n"
               << "Cache-Control: no-cache, private\r\n"
               << "Content-Type: text/html\r\n"
               << "Content-Length: "
               << content.size()
               << "\r\n"
               << "\r\n"
               << content;

            std::string res = ss.str();
            size_t bytesToSend = res.size() + 1;
            sendToClient(currSock, res.c_str(), bytesToSend);
        }
    }

    if (method == "GET") {
        if (route == "/start_calibration") {
            // std::string fileRootLocation = "C:\\Users\\user\\Repos\\Javascript\\2021-02-18-machingclee.github.io";
            // int htmlPos = targetHtml.find(".html");
            // if (htmlPos != std::string::npos) {
            //     targetHtml.erase(htmlPos, htmlPos + 5);
            // }
            // targetHtml.erase(0, 1);
            // std::string filePath = fileRootLocation + "\\" + (targetHtml == "" ? "" : (targetHtml + "\\")) + "index.html";
            // std::ifstream file{filePath};
            // std::string content{"404 Not Found"};

            // if (file.good()) {
            //     std::ostringstream ss;
            //     ss << file.rdbuf();
            //     content = ss.str();
            //     // std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            //     // content = str;
            // }
            // file.close();

            std::string content{"I will do ML work"};
            std::ostringstream ss;
            ss << "HTTP/1.0 200 OK\r\n"
               << "Cache-Control: no-cache, private\r\n"
               << "Content-Type: text/html\r\n"
               << "Content-Length: "
               << content.size()
               << "\r\n"
               << "\r\n"
               << content;

            std::string res = ss.str();
            size_t bytesToSend = res.size() + 1;
            sendToClient(currSock, res.c_str(), bytesToSend);
        }
    }
};