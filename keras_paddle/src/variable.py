import paddle

from keras.src.backend.common import KerasVariable
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras_paddle.src.ops.core import convert_to_numpy
from keras_paddle.src.ops.core import convert_to_tensor


class Variable(KerasVariable):
    def __init__(
        self,
        initializer,
        shape=None,
        dtype=None,
        trainable=True,
        autocast=True,
        name=None,
        aggregation="none",
    ):
        if dtype is None:
            if not callable(initializer) and not isinstance(initializer, str):
                t = convert_to_tensor(initializer)
                dtype = getattr(t, "dtype", None)
                if hasattr(dtype, "name"):
                    dtype = dtype.name.lower()
        elif hasattr(dtype, "name"):
            dtype = dtype.name.lower()
        elif isinstance(dtype, str):
            dtype = dtype.lower()

        super().__init__(
            initializer=initializer,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            autocast=autocast,
            name=name,
            aggregation=aggregation,
        )

    def _initialize(self, value):
        if isinstance(value, paddle.base.framework.EagerParamBase):
            # Reuse same parameter.
            self._value = value
        else:
            value = convert_to_tensor(value, dtype=self._dtype)
            if not value.stop_gradient:
                # Detach tensors computed from other tensors so that the
                # variable does not stay attached to the autograd graph.
                value = value.detach()
            self._value = value
            self._value.stop_gradient = not self.trainable

    def _direct_assign(self, value):
        self._value.set_value(value)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def __array__(self, dtype=None):
        value = convert_to_numpy(self.value)
        if dtype:
            return value.astype(dtype)
        return value

    @property
    def value(self):
        def maybe_use_symbolic_tensor(value):
            return value

        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                value = self._maybe_autocast(value)
                return maybe_use_symbolic_tensor(value)
        if self._value is None:
            value = self._maybe_autocast(
                self._initializer(self._shape, dtype=self._dtype)
            )
        else:
            value = self._maybe_autocast(self._value)
        return maybe_use_symbolic_tensor(value)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._value is not None:
            self._value.stop_gradient = not value
