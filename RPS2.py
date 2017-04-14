# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 18:09:10 2015

@author: Ilia
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy as sp
from scipy import ndimage


#def eight_neighbors(N,M):
#    for dx in [-1,0,1]:
#        xi = np.arange(M) + dx
#        if dx==1:
#            xi[-1] = 0
#        for dy in [-1,0,1]:
#            yi = np.arange(N) + dy
#            if dy==1:
#                yi[-1] = 0
##            yield (xi[np.newaxis,:],yi)
#            yield (xi,yi)

def eight_neighbors(N,M):
    for dx in [-1,0,1]:
        xi = np.arange(1,M-1) + dx
        for dy in [-1,0,1]:
            yi = np.arange(1,N-1) + dy
            yield (xi,yi[:,np.newaxis])

def GetPopFromBMP(bmp,colors):
    pop = np.zeros((bmp.shape[0],bmp.shape[1]),dtype=int)
    
    for colorind,color in enumerate(colors):
        this_color = np.array(color).reshape((1,1,3))
        pop[np.all(bmp == this_color,axis=2)] = colorind+1
    return pop


def RPSTensor_AllPlay(L):
    A = np.zeros((L+1,L+1))
    for i in range(1,L+1):
        for j in range(i+1,L+1):
            sgn = ((-1)**(i-j))
            A[i,j] = sgn
            A[j,i] = -sgn
    return A

def RPSTensor_Degrees(L):
    A = np.zeros((L+1,L+1))
    for i in range(1,L+1):
        for j in range(i+1,L+1):
            
            sgn = ((-1)**(i-j))
            A[i,j] = sgn
            A[j,i] = -sgn
    return A

def MakeGames(pop,rps_tensor,neighbor_iterator):
    N,M = pop.shape
#    tensor = rps_tensor[np.newaxis,np.newaxis,...]
#    results=[]
#    games_total = np.zeros( (N-2,M-2,3,3) )
#    leader_board_results = np.zeros( (N-2,M-2) )
#    next_board = np.zeros(N,M)
#    results_total = np.zeros( (N-2,M-2) );
    results_total = np.zeros((N,M))
    xi0 = np.arange(1,M-1)
    yi0 = np.arange(1,N-1).reshape((N-2,1))
    for i,(xi,yi) in enumerate(neighbor_iterator(N,M)):
#        return xi,yi
#        this_result = (pop[yi,xi,np.newaxis,:] * tensor * pop[1:-1,1:-1,:,np.newaxis]).sum((2,3))
        this_result = rps_tensor[pop[yi,xi],pop[1:-1,1:-1]]
#        return this_result
        results_total[1:-1,1:-1] += this_result
#        results.append( pop[yi,xi,np.newaxis,:] * tensor * pop[yi0,xi0,:,np.newaxis])
#        games_total += np.sum(this_result,(2,3))
    leader_board_results = np.zeros( (N,M) )
    next_board = np.copy(pop)
    for i,(xi,yi) in enumerate(neighbor_iterator(N,M)):
        swap_ind = np.where( \
            leader_board_results[1:-1,1:-1] \
            < 1.01 * results_total[yi,xi])
        
        leader_board_results[1:-1,1:-1][swap_ind[0],swap_ind[1]] = results_total[yi,xi][swap_ind]
        next_board[1:-1,1:-1] [swap_ind[0],swap_ind[1]] = pop[yi,xi] [swap_ind[0],swap_ind[1]]
    return next_board

def GetRGB_For_POP(pop,colors):
    rgb = np.zeros(pop.shape + (3,),dtype=np.uint8)
    for colorind,color in enumerate(colors):
        this_color = np.array(color).reshape((1,3))
        rgb[pop == colorind+1,:] = this_color
    return rgb

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)]
#colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
L = len(colors)

games_func = MakeGames
rps_func = RPSTensor_AllPlay

num_gen = 200

#sample_fname = 'sample1.bmp'
sample_fname = 'RGB_CY2.bmp'
bmp = ndimage.imread(sample_fname)
pop_0 = GetPopFromBMP(bmp,colors)
rps_tensor = rps_func(L)
next_board = np.copy(pop_0)
#for i in range(10):
#    next_board = games_func(next_board,rps_tensor,eight_neighbors)
#
###
#plt.figure()
#plt.imshow(GetRGB_For_POP(pop_0,colors))
#
#plt.figure()
#plt.imshow(GetRGB_For_POP(next_board,colors))


##


from matplotlib import animation

anim_writer = animation.writers[u'ffmpeg'] (fps=25)

fig = plt.figure()
img_h = plt.imshow(GetRGB_For_POP( pop_0,colors))

##

with anim_writer.saving(fig,'result_%s_%d_%s_%s_%d.mp4' % \
        (rps_func.__name__,L,games_func.__name__,sample_fname,num_gen),250):
    anim_writer.grab_frame()
    
    next_board = pop_0
    for i in xrange(num_gen):
        print i
        next_board = games_func(next_board,rps_tensor,eight_neighbors)
        img_h.set_data(GetRGB_For_POP(next_board,colors))
        anim_writer.grab_frame()
