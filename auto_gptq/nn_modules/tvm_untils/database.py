from .pycuda_warpper import TVMHandler

handler_database = {}
def get_handler(bits:int, n:int, k:int):
    key = f"b{bits}n{n}k{k}"
    if key not in handler_database:
        handler_database[key] = TVMHandler(bits=bits, n=n, k=k)
    return handler_database[key]