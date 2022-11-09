import numpy as np
import matplotlib.pyplot as plt

# EncodeRLE converts from binary array to RLE
def EncodeRLE(arr):
    # Flatten array column wise
    flat = arr.ravel('F') # column-wise
    switches = np.nonzero(np.append(flat,0) - np.append(0,flat))[0]
    counts = np.append(switches,switches[-1]) - np.append(0,switches)

    counts = counts[:-1]
    rle = np.hstack((switches[::2][:,None],counts[1::2][:,None]))
    
    return rle.ravel()

# DecodeRLE converts from RLE to binary array
def DecodeRLE(rle_arr,img_shape):
    mask = np.zeros(img_shape).ravel()
    for row in range(len(rle_arr)):
        start, count = rle_arr[row,0], rle_arr[row,1]
        mask[start:start+count] = 1
    
    mask = mask.reshape(img_shape)
    return mask.T

def DiceCoefficient(mask1, mask2):
    intersection = np.sum(mask1*mask2,axis=[1,2,3])
    tot = np.sum(mask1,axis=[1,2,3]) + np.sum(mask2,axis=[1,2,3])
    return np.mean((2 * intersection) / tot,axis=0)

