# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:45:41 2015

@author: Ilia
"""

import numpy as np

from matplotlib import pyplot as plt

for colorind in range(3):
    this_color = np.zeros((1,1,3))
    this_color[0,0,colorind]=1
    indices = np.where(np.all(next_board == this_color,axis=2))
    N,M,ignore = next_board.shape
    xs = indices[0] - N/2
    ys = indices[1] - M/2
    distances = np.sqrt(xs**2 + ys**2)
    
    plt.figure()
    plt.hist(distances,bins=100)

