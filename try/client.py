from pickle import PickleError
import socket,ssl
import json
import pickle

SERVER_ADDRESS_PORT = ('127.0.0.1', 8700)
# 如果开多个客户端，这个client_type设置不同的值，比如客户端1为linxinfa，客户端2为linxinfa2
client_type ='linxinfa'

def send_data(client, cmd, **kv):
    global client_type
    jd = {}
    jd['COMMAND'] = cmd
    jd['client_type'] = client_type
    jd['data'] = kv
    pick = pickle.dumps(jd)
    print(b"send: "+pick)
    client.sendall(pick)

    
def input_client_type():
    return input("注册客户端，请输入名字 :")
    
if '__main__' == __name__:
    client_type = input_client_type()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations("assets/ca.crt")
    client = socket.socket()
    client.connect(SERVER_ADDRESS_PORT)
    client = context.wrap_socket(client, server_hostname="Arch")
    
    print(client.recv(1024).decode(encoding='utf8'))
    # print(client.recv(1024).decode(encoding='utf8'))
    send_data(client, 'CONNECT')

    while True:
        a=input("请输入要发送的信息:")
        send_data(client, 'SEND_DATA', data=a)
        # send_data(client, 'SEND_DATA', data=a)
