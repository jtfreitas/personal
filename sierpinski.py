import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations as combo

def triangle_gen():
    o, a, b = ((0,0), 
               (random.random(), random.random()), 
               (random.random(), random.random()))
    return np.array([o, a, b])

triangle = triangle_gen()
#triangle = np.array([(0,2), 
#                     (5,0), 
#                     (3,5)])

plt.figure(dpi = 1800)

for segment in zip(combo(triangle[:,0], 2), 
                   combo(triangle[:,1], 2)):
    plt.plot(segment[0], segment[1], c = 'r', lw = 0.5)


def gen_point(p0, p1, p2):
    
    r1, r2 = sorted([random.random(), 
                     random.random()])
    
    point = (r1*p0[0] + (r2 - r1)*p1[0] + (1 - r2)*p2[0],
             r1*p0[1] + (r2 - r1)*p1[1] + (1 - r2)*p2[1])          
    
    return point

def move_pt(point, triangle, scale):
    
    choice_vertex = random.choice(triangle)
    motion_vector = choice_vertex - point
    moved_pt = point + motion_vector/scale
    return moved_pt

iterations = 100000

init_point = gen_point(*triangle)
track_point = init_point
scale = 2

plot_pts = np.zeros((iterations, 2))
for i in range(iterations):
#    point = gen_point(*triangle)
#    plt.scatter(point[0], point[1], marker = '.', c = 'k')
    plot_pts[i] = track_point    
    track_point = move_pt(track_point, triangle, scale)


plt.scatter(plot_pts[:,0], plot_pts[:,1], marker ='.', linewidths = 0, s=0.2, c = 'k')
plt.axis('off')
plt.savefig('sierpinski.png')
plt.show()