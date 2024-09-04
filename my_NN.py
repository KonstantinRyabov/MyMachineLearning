import numpy as np
from numpy import ndarray
from typing import List

def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape
    return None

class Operation(object):

    def __init__(self):
        pass

    def forward(self, input_: ndarray):

        self.input_ = input_
        self.output = self._output()

        return self.output


    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad


    def _output(self) -> ndarray:
        raise NotImplementedError()


    def _input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()
    

class ParamOperation(Operation):

    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)
        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()