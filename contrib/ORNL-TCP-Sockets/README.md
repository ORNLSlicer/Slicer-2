# ORNL TCP Sockets
A static library that provides a TCP server and client using Qt. This library also provides a mechanism for establishing a text console over TCP sockets.

## Integrating with a project using CMake:

 1. Open a terminal window at the root of your project
 2. Add a git submodule by running: `git submodule add https://github.com/mdfbaam/ORNL-TCP-Sockets "contrib/ORNL-TCP-Sockets"`
> Note: this example assumes the repository should be cloned into a "contrib" folder. If you project uses a different folder for libraries, use that path instead.
3. In your CMakeLists.txt add `add_subdirectory("contrib/ORNL-TCP-Sockets")` anywhere before `add_executable()` is called. This will create two variables you can use: `${ORNL_TCP_SOCKETS_INCLUDE}` and `${ORNL_TCP_SOCKETS_LIB}`.
4. Add `include_directories(${ORNL_TCP_SOCKETS_INCLUDE})` after the `add_subdirectory()` call.
5. Add `${ORNL_TCP_SOCKETS_LIB}` to the end of your `target_link_libraries()` call
6. You should now be able to import classes from the library.

## Usage
### Server
```
#include  <tcp_server.h>
#include  <data_stream.h>

TCPServer* server = new TCPServer();
connect(server, &TCPServer::newClient, this, [this](ORNL::TCPConnection* connection)
{
    auto data_stream = new DataStream(connection);
    connect(data_stream, &DataStream::newData, this, [this, data_stream]()
    {
        QString msg = data_stream->getNextMessage();

        // Do something with message

        data_stream->send(msg); // Send back to host
    });
});

connect(server, &TCPServer::newClient, this, [this](TCPConnection* new_connection)
{
    // Do something with new client connection
});

connect(server, &TCPServer::clientDisconnected, this, [this](TCPConnection* connection)
{
    // Do something with new client disconnection
});

server->startAsync(12345); // 12345 is the port number
```

### Client
```
#include  <tcp_connection.h>
TCPConnection* client = new TCPConnection();

connect(client, &TCPConnection::connected, this, []()
{
    qDebug() << "Connected";
});

connect(client, &TCPConnection::disconnected, this, []()
{
    qDebug() << "Disconnected";
});

client->setupNewAsync("localhost", 12345);
```
