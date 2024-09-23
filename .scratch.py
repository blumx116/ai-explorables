import struct

with open("smallest/model.safetensors", "rb") as f:
    result = struct.unpack("<Q", f.read()[:8])
    print(result)
