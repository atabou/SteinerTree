import cffi

print_banner("Building CFFI Module")
ffi = cffi.FFI()

include = "../include"
headers = ["graph.hpp", "query.hpp", "table.hpp", ]
