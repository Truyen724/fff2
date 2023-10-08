import socket
import threading
# Khởi tạo socket cho client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
global_variable = "0"
# Địa chỉ và cổng của server
server_address = ('127.0.0.1', 12345)

# Kết nối đến server
client_socket.connect(server_address)
print("Đã kết nối đến server.")
def client():
    locals_variable = global_variable
    while True:
        if locals_variable!= global_variable:
            locals_variable = global_variable
            # Gửi tin nhắn đến server
            client_socket.send(global_variable.encode('utf-8'))
    client_socket.close()
def device():
    global global_variable
    while True:
        message = input("Nhập vào message")
        global_variable = message

thread1 = threading.Thread(target=device)
thread2 = threading.Thread(target=client)

thread1.start()
thread2.start()

# thread1.join()
# thread2.join()


        
# send_message("xin chao")