import autoreload as autoreload
import matplotlib
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import skimage
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import Load_Data as ld
import  numpy as np

import Sigmoid_formula as sf
import All_Functions as af
import Run_Load_data as rl

# Load_Data.load_dataset()

# w, b = af.initialize_with_zero(2)
# print(str(af.initialize_with_zero(2)))
# print(sf.sigmoid(np.array([0.5, 0, 2.0])))
d = af.model(rl.train_set_x, rl.train_set_y, rl.test_set_x, rl.test_set_y, num_iterations=2000, learning_rate=0.0005, print_cost=True)# w =  np.array([[1.], [2]])
# print(w.T)

# learning_rates = [0.01, 0.001, 0.0001]
# models = {}
#
# for lr in learning_rates:
#     print ("Training a model with learning rate: " + str(lr))
#     models[str(lr)] = af.model(rl.train_set_x, rl.train_set_y, rl.test_set_x, rl.test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
#     print ('\n' + "-------------------------------------------------------" + '\n')
#
# for lr in learning_rates:
#     plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))
#
# plt.ylabel('cost')
# plt.xlabel('iterations (hundreds)')
#
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = ld.load_dataset()


## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "enhanced-image.png"

fname = "image/" + my_image
image = np.array(skimage.io.imread(fname))
my_image = skimage.transform.resize(image, (rl.num_px, rl.num_px)).reshape((1, rl.num_px*rl.num_px*3)).T
my_predicted_image = af.predict(d["w"], d["b"], my_image)

plt.imshow(image)
plt.show()

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
