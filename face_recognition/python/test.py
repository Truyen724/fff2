import uuid

def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0, 12, 2)])

mac_address = get_mac_address()
print("Địa chỉ MAC của thiết bị là:", mac_address)