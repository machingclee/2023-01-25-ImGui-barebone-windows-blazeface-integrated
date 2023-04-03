#include "web_server/TcpListener.h"

TcpListener::TcpListener(const char* ipAddress, int port) : _ipAddress(ipAddress), _port(port){};

int TcpListener::init() {
    // initialize winsock
    WSADATA wsData;
    WORD ver = MAKEWORD(2, 2);

    int wsOk = WSAStartup(ver, &wsData);
    if (wsOk != 0) {
        std::cerr << "Can't Initialize winsock! Quitting" << std::endl;
        return wsOk;
    }

    // create a socket
    _socket = socket(AF_INET, SOCK_STREAM, 0);

    if (_socket == INVALID_SOCKET) {
        std::cerr << "can't create a socket, quitting" << std::endl;
        return WSAGetLastError();
    }

    // bind the ip address and port to a socket
    sockaddr_in hint;
    hint.sin_family = AF_INET;
    hint.sin_port = htons(_port);
    // hint.sin_addr.S_un.S_addr = INADDR_ANY;
    // hint.sin_addr.S_un.S_addr = inet_addr(_ipAddress);
    inet_pton(AF_INET, _ipAddress, &hint.sin_addr);
    int bindResult = bind(_socket, (sockaddr*)&hint, sizeof(hint));
    if (bindResult == SOCKET_ERROR) {
        return WSAGetLastError();
    }

    // tell winsock the socket is for listening
    int listeningResult = listen(_socket, SOMAXCONN);
    if (listeningResult == SOCKET_ERROR) {
        return WSAGetLastError();
    }

    FD_ZERO(&_fd_set);
    FD_SET(_socket, &_fd_set);

    return 0;
};
int TcpListener::run(std::atomic<bool>& stop_flag) {
    while (!stop_flag) {
        fd_set copy = _fd_set;
        int socketCount = select(0, &copy, nullptr, nullptr, nullptr);
        for (int i = 0; i < socketCount; i++) {
            SOCKET currSock = copy.fd_array[i];
            if (currSock == _socket) {
                // accept a new connection

                // sockaddr_in client;
                // int clientSize = sizeof(client);
                // SOCKET clientSocket = accept(listening, (sockaddr *)&client, &clientSize);

                SOCKET clientSocket = accept(_socket, nullptr, nullptr);

                // add the new connection the list of connected clients
                // socket <-> fd <-> u_int
                FD_SET(clientSocket, &_fd_set);
                onClientConnected(clientSocket);
            } else {
                char buffer[4096];
                ZeroMemory(buffer, 4096);

                //  receive message
                int bytesReceived = recv(currSock, buffer, 4096, 0);
                if (bytesReceived <= 0) {
                    // drop the client
                    // TODO: client disconnected;
                    onClientDisconnected(currSock);
                    closesocket(currSock);
                    FD_CLR(currSock, &_fd_set);

                } else {
                    // send message toimage.png other clients, excluding the listening socket
                    onMessageReceived(currSock, buffer, bytesReceived);
                    bool isNullSentence = buffer[0] == 13 && buffer[1] == 10;
                    if (!isNullSentence) {
                        for (int i = 0; i < _fd_set.fd_count; i++) {
                            SOCKET outSock = _fd_set.fd_array[i];

                            if (outSock != _socket && outSock != currSock) {
                                // ostringstream ss;
                                // ss << "SOCKET #" << sock << ": " << buffer << "\r\n";
                                // string strOut = ss.str();
                                // send(outSock, strOut.c_str(), strOut.size() + 1, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    //  remove the listening socket from the master fd set and close it
    // to prevent anyone else trying to connect.

    FD_CLR(_socket, &_fd_set);
    closesocket(_socket);

    while (_fd_set.fd_count > 0) {
        SOCKET sock = _fd_set.fd_array[0];
        FD_CLR(sock, &_fd_set);
        closesocket(sock);
    }

    WSACleanup();
    return 0;
};

void TcpListener::sendToClient(int clientSocket, const char* msg, size_t length) {
    send(clientSocket, msg, length, 0);
}

void TcpListener::broadcastToClients(int sendingClientToExlcude, const char* msg, size_t length) {
    for (int i = 0; i < _fd_set.fd_count; i++) {
        int outSock = _fd_set.fd_array[i];
        if (sendingClientToExlcude != _socket && sendingClientToExlcude != outSock) {
            sendToClient(outSock, msg, length);
        }
    }
}
