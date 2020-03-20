import socket
import time
import sys
import json
import pandas as pd
import pickle

from somefunction import haha

# Create a TCP/ IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind the socket to a port. Port 1236 can be changed to any port if required.
server_address = ('localhost', 1236)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

# hardcoded prediction results
pred = {"cost":{"6 months before":3882.80, "6 months after":13112.54, "1 year after":2230.19, 
    "2 year after":1736.58, "5 years after":11800.33, "10 years after":14917.57},
    "survival": {"6 months before":100.0, "6 months after":96.20, "1 year after":90.10, 
    "2 years after":86.90, "5 years after":80.09, "10 years after":71.22}}
    
#takes json object and returns a string
pred_string = json.dumps(pred)

#takes string and returns bytes object
pred_json = bytes(pred_string, encoding='utf-8')

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('connection from', client_address)
        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(1024)

            print('received {!r}'.format(data))
            if data:
                # call machine learning model here. 
                # if required, use json.dumps(var) to convert data to string. bytes() function requires string argument to encode.
                # if required, use bytes(var, encoding='utf-8') function to convert string to bytes object. Connection.sendall accepts bytes object.

                pred = haha(data)
                stringData = pred.to_json()

                
                pred_json2 = bytes(stringData, encoding='utf-8')

                #sends data back to client  

                print('sending data back to the client')
                connection.sendall(pred_json2)

            else:
                print('no data from', client_address)
                break
    finally:
        # Clean up the connection
        connection.close()



