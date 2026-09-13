import numpy as np
from tensorflow.keras.utils import to_categorical
from load_and_inspect_dataset import x_train, y_train, x_test, y_test

# Normalize pixel values from [0,255] to [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels: 3 -> [0,0,0,1,0,0,0,0,0,0]
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)