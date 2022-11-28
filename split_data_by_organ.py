import pandas as pd

train_path = '/Users/parkersimpson/CMU/02740/FinalProject/hubmap-organ-segmentation/train.csv'

data = pd.read_csv(train_path)

# Group data by organ
val_counts = data['organ'].value_counts()
grouped = data.groupby('organ')
# print(grouped, val_counts,val_counts.index)

# Iterate over organs and split data
train_inxs, test_inxs = [],[]
for name, group in grouped:
    for row in range(len(group)):
        if row > len(group)*0.7:
            test_inxs.append(group.index[row])
        else:
            train_inxs.append(group.index[row])

print(len(data),len(train_inxs),len(test_inxs))

# Save files for test data
data.iloc[train_inxs,:].to_csv('train_70-30.csv',index=False)
data.iloc[test_inxs,:].to_csv('test_70-30.csv',index=False)
