"""
Data utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

# Modifications made by Yuan Feng et al., 2026:
# Retain partial functional functions.

import math
from typing import Union

import numpy as np
import torch

def turn_data_into_specified_dtype(
    data: Union[np.ndarray, torch.Tensor, list],
    dtype: str = "tensor",
) -> Union[np.ndarray, torch.Tensor]:
    """Turn the given data into the specified data type."""

    if isinstance(data, torch.Tensor):
        data = data if dtype == "tensor" else data.numpy()
    elif isinstance(data, list):
        data = torch.tensor(data) if dtype == "tensor" else np.asarray(data)
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data) if dtype == "tensor" else data
    else:
        raise TypeError(
            f"data should be an instance of list/np.ndarray/torch.Tensor, but got {type(data)}"
        )
    return data

def _parse_delta_torch(missing_mask: torch.Tensor) -> torch.Tensor:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def cal_delta_for_single_sample(mask: torch.Tensor) -> torch.Tensor:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        # the first step in the delta matrix is all 0
        d = [torch.zeros(1, n_features, device=device)]

        for step in range(1, n_steps):
            d.append(
                torch.ones(1, n_features, device=device) + (1 - mask[step - 1]) * d[-1]
            )
        d = torch.concat(d, dim=0)
        return d

    device = missing_mask.device
    if len(missing_mask.shape) == 2:
        n_steps, n_features = missing_mask.shape
        delta = cal_delta_for_single_sample(missing_mask)
    else:
        n_samples, n_steps, n_features = missing_mask.shape
        delta_collector = []
        for m_mask in missing_mask:
            delta = cal_delta_for_single_sample(m_mask)
            delta_collector.append(delta.unsqueeze(0))
        delta = torch.concat(delta_collector, dim=0)

    return delta


def _parse_delta_numpy(missing_mask: np.ndarray) -> np.ndarray:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def cal_delta_for_single_sample(mask: np.ndarray) -> np.ndarray:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        # the first step in the delta matrix is all 0
        d = [np.zeros(n_features)]

        for step in range(1, seq_len):
            d.append(np.ones(n_features) + (1 - mask[step - 1]) * d[-1])
        d = np.asarray(d)
        return d

    if len(missing_mask.shape) == 2:
        seq_len, n_features = missing_mask.shape
        delta = cal_delta_for_single_sample(missing_mask)
    else:
        n_samples, seq_len, n_features = missing_mask.shape
        delta_collector = []
        for m_mask in missing_mask:
            delta = cal_delta_for_single_sample(m_mask)
            delta_collector.append(delta)
        delta = np.asarray(delta_collector)
    return delta


def parse_delta(
    missing_mask: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask :
        Binary masks indicate missing data (0 means missing values, 1 means observed values).
        Shape of [n_steps, n_features] or [n_samples, n_steps, n_features].

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """
    if isinstance(missing_mask, np.ndarray):
        delta = _parse_delta_numpy(missing_mask)
    elif isinstance(missing_mask, torch.Tensor):
        delta = _parse_delta_torch(missing_mask)
    else:
        raise RuntimeError
    return delta

