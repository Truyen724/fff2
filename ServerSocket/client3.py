import socket

# Khởi tạo socket cho client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Địa chỉ và cổng của server
server_address = ('127.0.0.1', 12345)

# Kết nối đến server

print("Đã kết nối đến server.")
client_socket.connect(server_address)
while True:
    # Nhận và hiển thị phản hồi từ server
    data = client_socket.recv(1024)
    print(f"Phản hồi từ server: {data.decode('utf-8')}")
client_socket.close()
# Đóng kết nối

