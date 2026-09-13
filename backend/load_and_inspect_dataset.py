import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

x_train = idx2numpy.convert_from_file(
    "data/archive/train-images-idx3-ubyte/train-images-idx3-ubyte"
)
y_train = idx2numpy.convert_from_file(
    "data/archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
x_test = idx2numpy.convert_from_file(
    "data/archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
)
y_test = idx2numpy.convert_from_file(
    "data/archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
)

print("Training images: ", x_train.shape)
print("Training labels: ", y_train.shape)

print("Testing images: ", x_test.shape)
print("Testing labels: ", y_test.shape)

plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title("Label: "+str(y_train[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()


print(x_train[0].shape)
print(x_train[0])
print(y_train[0])