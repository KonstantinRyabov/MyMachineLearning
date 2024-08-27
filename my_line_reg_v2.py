import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List

# прямой проход
def forward_linear_regression(X_batch: ndarray,
                              y_batch: ndarray,
                              weights: Dict[str, ndarray]
                              )-> Tuple[float, Dict[str, ndarray]]:

    # проверка размерностей
    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights['W'].shape[0]
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    # вычисление предсказания и потерь
    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']
    loss = np.mean(np.power(y_batch - P, 2))

    # словарь с информацией
    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info

# обратный проход
def loss_gradients(forward_info: Dict[str, ndarray],
                   weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    batch_size = forward_info['X'].shape[0]

    # расчет градиента
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dLdN = dLdP * dPdN
    
    dNdW = np.transpose(forward_info['X'], (1, 0))
    dLdW = np.dot(dNdW, dLdN)
    
    dPdB = np.ones_like(weights['B'])
    dLdB = (dLdP * dPdB).sum(axis=0)
    

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB

    return loss_gradients