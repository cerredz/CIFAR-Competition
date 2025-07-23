import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def create_label_array(n: int):
    return n

def convert_conv_2d_input(row: list):
# convert a 3072 length array into a 3 x 32 x 32 tensor
    np_array = np.array(row, dtype=np.float32) / 255.0  # Convert to float and normalize to [0, 1]
    tensor = np_array.reshape((3, 32, 32))
    return tensor


