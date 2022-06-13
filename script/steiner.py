
import ctypes

if __name__ == "__main__":
    libname = "./lib/libsteiner.so"
    print(libname)
    steiner = ctypes.CDLL(libname)

    ctypes
