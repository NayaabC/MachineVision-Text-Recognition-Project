import tensorflow as tf
from tensorflow.python.client import device_lib
print('NUM GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

print(device_lib.list_local_devices())