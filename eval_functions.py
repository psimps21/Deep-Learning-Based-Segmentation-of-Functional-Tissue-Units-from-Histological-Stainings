import numpy as np
import matplotlib.pyplot as plt
from custom_torch_dataset import SegDataset


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

    
if __name__ == '__main__':
    # Test encoding and decoding
    # Paths to directory with training images and the annotation csv file
    train_data_dir = "/Users/parkersimpson/CMU/02740/FinalProject/hubmap-organ-segmentation/train_images"
    annot_path = '/Users/parkersimpson/CMU/02740/FinalProject/hubmap-organ-segmentation/train.csv'

    # Initialize custom dataset object
    train_dataset = SegDataset(root_dir=train_data_dir,annot_csv=annot_path)

    # Test on our data
    for i in range(2):
        sample = train_dataset[i]
        print('Sample ID:',sample['id'])

        mask = DecodeRLE(sample['rle'],(sample['img_height'],sample['img_width']))
        encoded_rle = EncodeRLE(mask)
        print(sample['rle'].ravel()[:12])
        print(encoded_rle[:12])
        print('Encoded RLE is same as annotations:',np.array_equal(sample['rle'].ravel(),encoded_rle))
        print()

        fig, axs = plt.subplots(1,2)
        axs[0].imshow(sample['image'])
        axs[1].imshow(mask)
        plt.show()
