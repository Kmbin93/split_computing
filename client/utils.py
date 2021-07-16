import time
import struct

payload_size = struct.calcsize(">L")
data = b""

def recv_frame(conn):
    global data
    while len(data) < payload_size:
        data += conn.recv(4096)
    start = time.time()
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    # print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    end = time.time()
    return (end-start) * 1000.0, frame_data