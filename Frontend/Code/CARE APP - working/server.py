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

#takes json object and returns a string
# pred_string = json.dumps(pred)

# #takes string and returns bytes object
# pred_json = bytes(pred_string, encoding='utf-8')

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

                surv_df, pred, groupj, bills= haha(data)
#`````````````````` Update this whole file and give then the model files ```````````
                fileLocation = "..\\middleWomen\\patient_new.csv"
                fileLocation_my = "..\\middleWomen\\bills_new.csv"
                fileLocation_surv = "..\\middleWomen\\survival.csv"
                pred.to_csv(fileLocation)
                bills.to_csv(fileLocation_my)
                surv_df.to_csv(fileLocation_surv)
#`````````````````` Update this whole file and give then the model files (END) ```````````                
                pred_string = json.dumps(groupj)

                #takes string and returns bytes object
                pred_json = bytes(pred_string, encoding='utf-8')
                connection.sendall(pred_json)
                # sends data back to client 
                 
                print('sending data back to the client')
                break

            else:
                print('no data from', client_address)
                break
    finally:
        # Clean up the connection
        connection.close()



