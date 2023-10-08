# imports related to keras framework
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras

from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import (Dense, Dropout, SpatialDropout2D, Activation,
    Flatten, BatchNormalization, LeakyReLU, GaussianNoise, ZeroPadding2D, Reshape,
    Lambda, Layer, Input, concatenate,  Conv2D, MaxPooling2D,
    Conv2DTranspose, UpSampling2D)
# may exist in stable version in tensorflow>2.4
from tensorflow.keras.layers.experimental.preprocessing import (
    Rescaling, Resizing, Normalization)
from tensorflow.keras.optimizers import *
# from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model, to_categorical
# np_utils

# from tensorflow.keras.utils.data_utils import get_file

from tensorflow.keras.applications import resnet50, vgg19, inception_v3
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

# from tensorflow.keras.datasets import mnist, cifar10

# import onnx
# import tf2onnx
# from onnx_tf.backend import prepare

# onnxruntime is not at the moment supported for M1 macos
import sys
if sys.platform != 'darwin':
    import onnxruntime
    import onnx
    import tf2onnx
    from onnx_tf.backend import prepare
    import nvidia_smi

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        continue

# from keras_segmentation.models.all_models import model_from_name
# from keras_segmentation.pretrained import pspnet_50_ADE_20K # semantic segmentation model trained in indoor scene dataset ADE20K
# import segmentation_models as sm # Segmentation Models: using `keras` framework.
# sm.set_framework('tf.keras')
# sm.framework()
smooth = 1.
