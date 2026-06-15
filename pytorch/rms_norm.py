import numpy as np
from typing import List

def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
    # RMS Normalization

    x, gamma = np.array(x), np.array(gamma)
    rms = np.sqrt(1 / x.size * np.sum(x ** 2) + eps)
    x /= rms

    return list(np.round(x * gamma, 4))
