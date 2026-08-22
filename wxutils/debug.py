# -*- coding: utf-8 -*-


def check_array(arr, prefix="Variable Check"):
    arr_type = type(arr).__name__
    # CuPy arrays have a '.device' attribute; NumPy arrays do not
    device = getattr(arr, 'device', 'CPU (host)')
    dtype = getattr(arr, 'dtype', type(arr).__name__)
    print('\n################################################')
    print(f"[{prefix}] -> Object: {arr_type} | Location: {device} | Element Type: {dtype}\n")