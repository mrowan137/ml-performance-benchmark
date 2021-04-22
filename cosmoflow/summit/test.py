import tensorflow as tf


gpu_devices = tf.config.list_physical_devices('GPU')
print('the len of gpu_devices=', len(gpu_devices))
