
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def create_label_array(n: int):
    arr = [0] * 10
    arr[n] = 1
    return arr
