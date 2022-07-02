import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import scipy
import random
from scipy import ndimage

N = 500 #linear size of lattice
beta = 3.0 #inverse temperature
J = 1.0 #ising coupling. J>0 means ferromagnetic
H = 0.0 #external magnetic field
FRAMES = 500
FRAME_INTERVAL = 10000 #how many timesteps per frame of animation
TIMESTEPS = FRAMES*FRAME_INTERVAL
BATCH_SIZE = 1000 #shouldn't have to worry about collisions so long as (batch size)*(coordination number) << N^2

#randomly initialize state
state = np.zeros((N,N), dtype = 'int8')
for i in range(N):
    for j in range(N):
        state[i][j] = random.randint(0,1)

#this list of ndarrays will store the animation
data = [np.ndarray.copy(state)]

#computing the evolution
for i in range(1,TIMESTEPS):

    #randomly drawing the batch
    batch_x = np.random.randint(0,N-1, BATCH_SIZE)
    batch_y = np.random.randint(0,N-1, BATCH_SIZE)
    seed = random.random()

    #computing the updated spins
    for j in range(BATCH_SIZE):
        k = batch_x[j]
        l = batch_y[j]
    energy_difference = -4*J*(state[(k-1)%N][l] + state[(k+1)%N][l] + state[k][(l-1)%N] + state[k][(l+1)%N] - 2) + 2*H
    state[k][l] = int(seed < 1./(1. + math.exp(beta*energy_difference)))

    #saving current frame of animation (once every FRAME_INTERVAL iterations)
    if i%FRAME_INTERVAL == 0:
        print("writing frame "+str(i/FRAME_INTERVAL)+" of "+str(FRAMES))
        data.append(np.ndarray.copy(state))

#initializing graphics
fig = plt.figure()
ax = plt.axes(xlim=(0, N-1), ylim=(0, N-1))
imgplot = plt.imshow(data[0], vmin = 0, vmax = 1)

#defining animation function called by matplotlib.animation
def animate(k):
    imgplot.set_data(data[k])
    return [imgplot]

#displaying and saving the animation
anim = animation.FuncAnimation(fig, animate, frames=FRAMES-1, interval=1, blit=True)
anim.save('2d_ising_metropolis_beta='+str(beta)+'_H='+str(H)+'.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
plt.show()