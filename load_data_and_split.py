import boto3
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from botocore.client import ClientError


# all below for loading imgs into a np array with shape (351, 3000, 3000, 3)
s3 = boto3.resource('s3')

bucket = s3.Bucket('dl-wiki-dataset')

try:
    s3.meta.client.head_bucket(Bucket=bucket.name)
except ClientError:
    print("Error. Bucket not found.")

def load_train_img():
    count = 0
    img_id = []
    for obj in bucket.objects.filter(Prefix='train_images/'):
        file_stream = io.BytesIO()

        key = obj.key #dir
        img_id.append(int(key[13:-5]))

        obj2 = bucket.Object(key)
        obj2.download_fileobj(file_stream)
        img = Image.open(file_stream)
        imarr = np.array(img)
        if count == 0:
            all_img = [imarr]
        else:
            all_img = np.append(all_img, [imarr], 0)
        count += 1
        if count == 5: # for test purposes
            break
    return img_id, all_img

# get labels
def load_labels():
    labels = bucket.Object('train.csv')
    labels.download_file('train.csv')
    labels_df = pd.read_csv('train.csv').iloc[:5] #slicing for test purposes
    return labels_df

# all_img same as before; img_id: list of ids in the same order as all_img,; labels: train.csv
# uses img_id to get labels so that if files are not loaded in the same order as csv that's no problem

def train_val_test_split(all_img, img_id, labels, split_rat = [0.6, 0.2, 0.2], seed=123):
    assert all_img.shape[0] == len(labels), 'Too many images or too many labels'

    n = len(labels)
    img_id = np.array(img_id)

    rng = np.random.default_rng(seed)
    shuffled_indices = list(range(n))
    rng.shuffle(shuffled_indices)

    train_idx = int(n * split_rat[0])
    val_idx = int(n * split_rat[1]) + train_idx

    train_img = all_img[shuffled_indices[:train_idx],:,:,:]
    val_img = all_img[shuffled_indices[train_idx:val_idx],:,:,:]
    test_img = all_img[shuffled_indices[val_idx:],:,:,:]

    train_id = img_id[shuffled_indices[:train_idx]]
    val_id = img_id[shuffled_indices[train_idx:val_idx]]
    test_id = img_id[shuffled_indices[val_idx:]]

    train_labels = labels.loc[labels.id.isin(train_id)]
    val_labels = labels.loc[labels.id.isin(val_id)]
    test_labels = labels.loc[labels.id.isin(test_id)]

    return [train_img, val_img, test_img], [train_labels, val_labels, test_labels]

img_id, all_img = load_train_img()
labels_df = load_labels()
X, Y = train_val_test_split(all_img, img_id, labels_df)

# result = s3.list_objects(Bucket = 'dl-wiki-dataset', Prefix='train_images/').get('Contents')

# json_contents = []
# count = 0
# for o in result:
#     data = s3.get_object(Bucket='dl-wiki-dataset', Key=o.get('Key'))
#     contents = data['Body'].read()
#     json_contents.append(contents)
#     if count == 0:
#         break

# print(json_contents[0])