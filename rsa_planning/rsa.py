"""RSA speaker/listener helpers."""

import warnings

import numpy as np
import pandas as pd

class RSA_speaker_model_v1:

    ''' An implementation of RSA Speaker Model based on the mathematical formulae proposed by:
    CURRENTLY IN USE
    Monroe, Will, and Christopher Potts. "Learning in the rational speech acts model." arXiv preprint arXiv:1510.06807 (2015).'''

    def __init__(self, lexicon, prior, costs, alpha=1):

        self.lexicon = lexicon  # P(message | world_state)
        # self.prior = np.array(prior).reshape(-1,1)  # P(world_state) [column vector]
        # self.costs = np.array(costs).reshape(-1,1)  # C(message) [column vector]
        self.prior = column_vector(prior)
        self.costs = column_vector(costs)
        self.alpha = alpha  # rationality parameter

    def literal_speaker(self):
        utilities = self.alpha * (safelog(self.lexicon) - self.costs)
        # print(f"\nLiteral Speaker Utilities: \n {utilities}")
        literal_speaker_matrix = np.exp(utilities)
        # print(f"\nLiteral Speaker Norm: \n {colnorm(literal_speaker_matrix)}")
        return colnorm(literal_speaker_matrix)

    def pragmatic_listener(self):
        lit_spk = self.literal_speaker().T
        # print(f"\nLiteral Speaker Transposed: \n {lit_spk}")
        pragmatic_listener_matrix = lit_spk * self.prior
        # print(f"\nPragmatic Listener Non-Norm: \n {pragmatic_listener_matrix}")
        return colnorm(pragmatic_listener_matrix)

    def pragmatic_speaker(self):
        prag_lst = self.pragmatic_listener().T
        utilities = self.alpha * (safelog(prag_lst) - self.costs)
        pragmatic_speaker_matrix = np.exp(utilities)
        return colnorm(pragmatic_speaker_matrix)

def rownorm(mat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if isinstance(mat, np.ndarray):
            return mat / mat.sum(axis=1, keepdims=True)
        elif isinstance(mat, pd.DataFrame):
            row_sums = mat.sum(axis=1)
            return mat.div(row_sums, axis=0)
        else:
            raise TypeError("Input should be a numpy array or pandas DataFrame")

def column_vector(arr):
        arr = np.array(arr)
        return arr.reshape(-1, 1) if arr.ndim <= 1 else arr.T if arr.shape[1] != 1 else arr

def colnorm(mat):
    """Column normalization of np.array or pd.DataFrame"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if isinstance(mat, np.ndarray):
            return mat / mat.sum(axis=0, keepdims=True)
        elif isinstance(mat, pd.DataFrame):
            return mat.div(mat.sum(axis=0), axis=1)
        else:
            raise TypeError("Input should be a numpy array or pandas DataFrame")

def safelog(vals):
    """Silence distracting warnings about log(0)."""
    with np.errstate(divide='ignore'):
        return np.log(vals)
