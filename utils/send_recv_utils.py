import struct

def send_msg(sock, msg):
    """
    在发送消息的头部加上4字节的长度
    """
    assert len(msg) < 0x100000000
    msg = struct.pack(">I", len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    """
    实现sock接收消息功能
    """
    def recvall(sock, n):
        """
        按照规定字节数接收消息，以实现按规定字节接收消息
        """
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    return recvall(sock, msglen)