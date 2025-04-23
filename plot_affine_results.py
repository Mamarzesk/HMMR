import os
import re

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Scheherazade'
plt.rcParams.update({'font.size': 13})


directory = os.getenv('directory')
title = os.getenv('title')

indices = [
    case for case in os.listdir(directory)
    if case.endswith('results.csv')
]
indices = sorted(indices, key=lambda x: int(re.search(r'\d+', x).group()))
ncols = 6
fig, axes = plt.subplots(
    ncols=ncols, nrows=1 + len(indices) // ncols, figsize=(12, 9)
)
flat_axes = axes.flatten()
for i in range(len(indices), flat_axes.shape[0]):
    flat_axes[i].axis('off')

for i in range(len(indices)):
    array = np.genfromtxt(directory + indices[i], delimiter=',').T
    flat_axes[i].scatter(
        -array[:, 0], array[:, 1], c=np.arange(array.shape[0]), cmap='rainbow', s=3
    )
    flat_axes[i].scatter(
        -array[:1, 0], array[:1, 1], c='k', s=20
    )
    flat_axes[i].set_ylim(bottom=0)
    flat_axes[i].set_ylabel('mTRE (mm)')
    flat_axes[i].set_xlabel('similarity metric')
    flat_axes[i].set_title(re.search(r'\d+', indices[i]).group())
    flat_axes[i].axhline(y=2, linestyle='--', color='k')
plt.tight_layout(pad=2.0, w_pad=0.7, h_pad=1)
plt.suptitle(title)
plt.show()
