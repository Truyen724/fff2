import socket

# Khởi tạo socket cho client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Địa chỉ và cổng của server
server_address = ('127.0.0.1', 12345)

# Kết nối đến server

print("Đã kết nối đến server.")
client_socket.connect(server_address)
def send_message(message):
    client_socket.send(message.encode('utf-8'))
send_message("hellllllllllloooooooooooo")
        

