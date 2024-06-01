import autoreload as autoreload
import matplotlib
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import Load_Data
import  numpy as np

# Load_Data.load_dataset()

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = Load_Data.load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


train_set_x_faltten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_faltten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("test_set_x_falyen shape" + str(test_set_x_faltten.shape))
print("train_set_x_faltten shape:" + str(train_set_x_faltten.shape))


train_set_x = train_set_x_faltten / 255.
test_set_x = test_set_x_faltten / 255.


print(str(train_set_x_orig.shape))
# print(m_train.shape)
# print(str(m_test.shape))
# print(str(num_px.shape))

# print(classes)
# print(train_set_x_orig)
# print(train_set_y)
# print(test_set_x_orig)
# print(test_set_y)

# # Example of a picture
# index = 10
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" +
#        classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")