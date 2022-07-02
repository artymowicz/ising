
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import scipy
import random
from scipy import ndimage

N = 10
beta = 0.1
J = 1.0
H = 0
FRAMES = 1000
FRAME_INTERVAL = 1
TIMESTEPS = FRAMES*FRAME_INTERVAL


fig = plt.figure()
ax = plt.axes(xlim=(0, 0), ylim=(0, N-1))

state = np.zeros((N,1), dtype = 'int8')

#this will store the animation
data = [np.ndarray.copy(state)]

#computing the evolution
for i in range(1,TIMESTEPS):
    k = random.randint(0,N-1)
    print(beta*(-4*J*(state[(k-1)%N] + state[(k+1)%N] + 2*H)))
    print(1/(1 + math.exp(beta*(-4*J*(state[(k-1)%N] + state[(k+1)%N] + 2*H)))))
    if random.random() < 1/(1 + math.exp(beta*(-4*J*(state[(k-1)%N] + state[(k+1)%N] + 1) + 2*H) )):
        state[k] = 1
    else:
        state[k] = 0

    if i%FRAME_INTERVAL == 0:
        print("writing frame "+str(i/FRAME_INTERVAL)+" of "+str(FRAMES))
        data.append(np.ndarray.copy(state))


imgplot = plt.imshow(data[0], vmin = 0, vmax = 1)

def animate(k):
    imgplot.set_data(data[k])
    return [imgplot]


'''

def init():
    imgplot.set_data(np.random.random((N,N)))
    return [imgplot]

def animate(ik):
    imgplot.set_data(np.random.random((N,N)))
    return [imgplot]
'''

#print(data)

anim = animation.FuncAnimation(fig, animate,
                               frames=FRAMES-1, interval=1, blit=True)

anim.save('basic_animation.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

plt.show()
