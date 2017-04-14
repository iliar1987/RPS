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

def four_neighbors(N,M):
    for dx,dy in [(-1,0),(0,-1),(1,0),(0,1),(0,0)]:
        xi = np.arange(1,M-1) + dx
        yi = np.arange(1,N-1) + dy
        yield (xi,yi[:,np.newaxis])


def GetComposition(pop):
    composition=[]
    for i in range(3):
        composition.append(pop[...,i]==255)
    return composition
    
def GetPopFromBMP(bmp):
    return np.array(bmp,dtype=float)/255.0


def GetRPS_Tensor():
    A = np.zeros((3,3))
    for i in range(3):
        A[i,i-1] = -1
        A[i,i-2] = 1
    return A

def MG_Basic(pop,rps_tensor,neighbor_iterator):
    N,M,ignore = pop.shape
    tensor = rps_tensor[np.newaxis,np.newaxis,...]
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
        this_result = (pop[yi,xi,np.newaxis,:] * tensor * pop[1:-1,1:-1,:,np.newaxis]).sum((2,3))
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
        next_board[1:-1,1:-1,...] [swap_ind[0],swap_ind[1],...] = pop[yi,xi,...] [swap_ind[0],swap_ind[1],...]
    return next_board
#    return results

def Normalize(board):
    board /= (((board**normalization_power).sum(2)) ** (1.0/normalization_power)) [...,np.newaxis]
    board[np.isnan(board)]=0.0


def MakeGames2(pop,rps_tensor,neighbor_iterator):
    N,M,ignore = pop.shape
    tensor = rps_tensor[np.newaxis,np.newaxis,...]
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
        this_result = (pop[yi,xi,np.newaxis,:] * tensor * pop[1:-1,1:-1,:,np.newaxis]).sum((2,3))
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
        next_board[1:-1,1:-1,...] [swap_ind[0],swap_ind[1],...] = pop[yi,xi,...] [swap_ind[0],swap_ind[1],...]
    next_board2 = np.copy(pop)
    next_board2 += next_board*diffusion_coefficient
#    next_board2[next_board2>1.0] = 1.0
    Normalize(next_board2)
    return next_board2


def Games2ReduDiffu(pop,rps_tensor,neighbor_iterator):
    N,M,ignore = pop.shape
    tensor = rps_tensor[np.newaxis,np.newaxis,...]
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
        this_result = (pop[yi,xi,np.newaxis,:] * tensor * pop[1:-1,1:-1,:,np.newaxis]).sum((2,3))
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
        next_board[1:-1,1:-1,...] [swap_ind[0],swap_ind[1],...] = pop[yi,xi,...] [swap_ind[0],swap_ind[1],...]
    next_board2 = np.copy(pop)
    next_board2 += next_board*diffusion_coefficient
#    next_board2[next_board2>1.0] = 1.0
    next_board2 **= diffusion_power
    Normalize(next_board2)
    return next_board2
    

def MakeGames3(pop,rps_tensor,neighbor_iterator):
    N,M,ignore = pop.shape
    tensor = rps_tensor[np.newaxis,np.newaxis,...]
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
        this_result = (pop[yi,xi,np.newaxis,:] * tensor * pop[1:-1,1:-1,:,np.newaxis]).sum((2,3))
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
        next_board[1:-1,1:-1,...] [swap_ind[0],swap_ind[1],...] = pop[yi,xi,...] [swap_ind[0],swap_ind[1],...]
    next_board /= next_board.sum(2)[...,np.newaxis]
    next_board[np.isnan(next_board)]=0.0
    return next_board
#    return results

diffusion_power = 1.0
diffusion_coefficient = 1.0
normalization_power = 1.0

Games2ReduDiffu.__name__ = Games2ReduDiffu.__name__ + '_%.2g_N%.1f_%.1f' \
    %(diffusion_power-1.0,normalization_power,diffusion_coefficient)

MakeGames2.__name__ = MakeGames2.__name__ + '_%.1f_N%.1f' \
    %(diffusion_coefficient,normalization_power)


neighbors = four_neighbors
#games_func = Games2ReduDiffu
games_func = MG_Basic

num_gen = 2000

#sample_fname = 'sample1.bmp'
sample_fname = 'sample1.bmp'
bmp = ndimage.imread(sample_fname)
pop_0 = GetPopFromBMP(bmp)
rps_tensor = GetRPS_Tensor()
next_board = games_func(pop_0,rps_tensor,neighbors)
##
#plt.figure()
#plt.imshow(pop_0)
#
#plt.figure()
#plt.imshow(next_board)

##


from matplotlib import animation

anim_writer = animation.writers[u'ffmpeg'] (fps=25)

fig = plt.figure()
img_h = plt.imshow(pop_0)

##

out_fname = 'test2_%s_%s_%s_%d.mp4' % \
        (neighbors.__name__,games_func.__name__,sample_fname,num_gen)

print out_fname

with anim_writer.saving(fig,out_fname,250):
    anim_writer.grab_frame()
    
    next_board = pop_0
    for i in xrange(num_gen):
        print i
        next_board = games_func(next_board,rps_tensor,eight_neighbors)
        img_h.set_data(next_board)
        anim_writer.grab_frame()
