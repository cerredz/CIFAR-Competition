import numpy as np
import random
import time

def augment(feature_input: list):
    print("Adding aut")
    res = []
    res.append(feature_input)
    #res.append(vertical_flip(feature_input))
    #res.append(horizontal_flip(feature_input))
    res.extend(noise(feature_input))

    return res

def vertical_flip(feature_input: list):
    return feature_input

def horizontal_flip(feature_input: list):
    return feature_input

def noise(feature_input: list):
    n = len(feature_input)
    num_iterations = 8 # augment data by factor of 128
    bucket_size = 256 # add noise to 1 out of every 16 elements in features
    res = []

    for i in range(num_iterations):
        copy = feature_input.copy()
        for j in range(0, n // bucket_size):
            rand_index = random.randint(j * bucket_size, (j + 1) * bucket_size - 1)
            copy[rand_index] = copy[rand_index] + 1
        
        res.append(copy)

    return res
    






