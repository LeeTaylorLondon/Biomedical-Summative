import numpy as np
import pandas as pd

_   = np.ones((208, 176), dtype=float)
arr = [_ for x in range(179)]
ar2 = [_ for y in range(12)]
ar3 = [_ for z in range(640)]
ar4 = [_ for i in range(448)]

dict__ = {'mild': arr, 'moderate': ar2, 'none': ar3, 'very_mild': ar4}

npa = np.array(arr.copy())
print(npa.shape)

## Automate this
# npm = np.empty((0, 208, 176))
# npm = np.append(npm, arr, axis=0)
# npm = np.append(npm, ar2, axis=0)
# print(npm.shape)

# npm = np.empty((0, 208, 176))
# for vec in dict_.values():
#     vec = np.array(vec)
#     npm = np.append(npm, vec, axis=0)
# print(npm.shape)

def f(dict_, debug=False):
    npm = np.empty((0, 208, 176))
    for vec in dict_.values():
        vec = np.array(vec)
        npm = np.append(npm, vec, axis=0)
    if debug: print(npm.shape)
    return npm

print(f(dict__).shape)

ones = np.ones(10)
print(ones)
twos = np.ones(10) * (1 + 1)
print(twos)