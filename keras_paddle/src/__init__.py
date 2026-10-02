from keras.src.backend.common.name_scope import name_scope
from keras_paddle.src import ops
from keras_paddle.src import random
from keras_paddle.src import rnn
from keras_paddle.src.ops.core import compute_output_spec
from keras_paddle.src.ops.core import device_scope
from keras_paddle.src.variable import Variable

IS_THREAD_SAFE = True
SUPPORTS_COMPLEX_DTYPES = True
SUPPORTS_GRADIENT = True
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_SPARSE_TENSORS = False

distribution_lib = None
