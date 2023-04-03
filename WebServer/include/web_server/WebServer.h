#pragma once
#include "TcpListener.h"

class WebServer : public TcpListener {
public:
    WebServer(const char* ipAddress, int port) : TcpListener(ipAddress, port){};

protected:
    void onClientConnected(int clientSocket);
    void onClientDisconnected(int clientSocket);
    void onMessageReceived(int currSock, char* buffer, int bytesReceived);
};