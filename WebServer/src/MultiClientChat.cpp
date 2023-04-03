#include "web_server/MultiClientChat.h"

void MultiClientChat::onClientConnected(int clientSocket) {
    // send a welcome message to the connected client
    std::string welcomeMsg = "Welcome to the chat server";
    sendToClient(clientSocket, welcomeMsg.c_str(), welcomeMsg.size() + 1);
};

void MultiClientChat::onClientDisconnected(int clientSocket) {
    std::ostringstream ss;
    ss << "Client #" << clientSocket << " has disconnected";
    std::string msg = ss.str();
    broadcastToClients(clientSocket, msg.c_str(), msg.size() + 1);
};

void MultiClientChat::onMessageReceived(int currSock, char* buffer, int bytesReceived) {
    broadcastToClients(currSock, buffer, bytesReceived);
}