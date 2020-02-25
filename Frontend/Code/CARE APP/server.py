import socket
import time
import sys
import json


HEADERSIZE=10

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('localhost', 1236)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# s.bind((socket.gethostname(),1235))
sock.listen(1)

pred = {"6 months before":3882.80, "6 months after":13112.54, "1 year after":2230.19, 
    "2 year after":1736.58, "5 years after":11800.33, "10 years after":14917.57}
pred_string = json.dumps(pred)
pred_json = bytes(pred_string, encoding='utf-8')

while True:
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('connection from', client_address)
        while True:
            data = connection.recv(1024)

            print('received {!r}'.format(data))
            if data:
                print('sending data back to the client')  
                connection.sendall(pred_json)
            else:
                print('no data from', client_address)
                break
    finally:
        connection.close()



