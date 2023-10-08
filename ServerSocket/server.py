import socket
import threading
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Xác định địa chỉ và cổng mà server sẽ lắng nghe
host = '127.0.0.1'  # Địa chỉ IP của máy local
port = 12345  # Cổng lắng nghe

# Gắn socket với địa chỉ và cổng
server_socket.bind((host, port))

# Lắng nghe kết nối từ client
server_socket.listen(10)  # Số lượng kết nối đồng thời tối đa là 5

print(f"Server đang lắng nghe trên {host}:{port}")

# Danh sách lưu trữ các client
clients = []
def handle_client(client_socket):
    while True:
        try:
            data = client_socket.recv(1024)  # Nhận dữ liệu từ client
            print(f"Phản hồi từ server: {data.decode('utf-8')}")
            if not data:
                break
            for client in clients:
                if client != client_socket:
                    client.send(data)  # Gửi dữ liệu đến tất cả các client kháC
        except:
            continue
while True:
    # Chấp nhận kết nối từ client
    client_socket, client_address = server_socket.accept()
    print(f"Kết nối từ {client_address} được chấp nhận.")
    # Thêm client vào danh sách
    clients.append(client_socket)
    print(len(clients))
    # Bắt đầu một luồng xử lý client
    client_thread = threading.Thread(target=handle_client, args=(client_socket,))
    client_thread.start()
    # client_socket.sendall("Xinchao".encode('utf-8'))