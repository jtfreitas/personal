"""
The game of life
"""
import numpy as np
from itertools import product

class game_of_life:
    
    def __init__(self, grid_x, grid_y, seed_val):
        self.the_grid = np.zeros((grid_x, grid_y))
        self.neighbor_set = np.array([[-1,-1], [-1,0],[-1, 1],
                                      [0, -1],         [0, 1],
                                      [1, -1], [1, 0], [1, 1]])
        self.seed = seed_val
        
    def start(self, no_inits):
        
        if type(self.seed) == int:
            np.random.seed(self.seed)
        else:
            print('Use an integer as the seed')
            return
        
        for i in range(no_inits):
            while True:
                position_init = np.array(np.random.randint(0, len(self.the_grid), size = 2))
                if 0 not in position_init:
                    break
                
            init_cluster = np.random.randint(0, len(self.neighbor_set))
            
            for i in range(init_cluster+1):
                neighbor = self.neighbor_set[np.random.choice(len(self.neighbor_set), replace = False)]
                try:
                    self.the_grid[tuple(position_init + neighbor)] = 1
                except:
                    continue
        return self
    
    def count_neighbors(self, x, y):
        count = 0
        cell_arr = np.array([x, y])
        for neighbor in self.neighbor_set:
            try:
                if self.the_grid[tuple(cell_arr + neighbor)] == 1:
                    count += 1
            except:
                continue
        return count
    
    def fate(self, x, y):
        if self.count_neighbors(x, y) in [2, 3] and (self.the_grid[x, y] == 1):
            return 1
        elif self.count_neighbors(x,y) == 3 and (self.the_grid[x, y] == 0):
            return 1
        else:
            return 0
        
        
    def update(self):
        for i in product(range(np.shape(self.the_grid)[0]), range(np.shape(self.the_grid)[1])):
            self.the_grid[i] = self.fate(*i)
        return self