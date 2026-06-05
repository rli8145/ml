import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        e = np.exp(z - z.max())
        return np.round(e / e.sum(), 4)