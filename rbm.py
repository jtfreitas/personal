import numpy as np
import matplotlib.pyplot as plt
import random

def spinify(data):
    vmin = -1
    data = np.where(data == True, data, vmin)
    gap = 10
        
    return data, vmin, gap

def digitize(data):
    vmin = 0
    data = np.where(data == True, data, vmin)
    gap = 1

    return data, vmin, gap

class Boltzmann:
    
    def __init__(self, n_visible, n_hidden):
        self.L = n_visible
        self.M = n_hidden
        
        def create_coords(n_p, x0):
            x = [i/(n_p - 1) - 0.5 for i in range(n_p)]
            y = [-x0] * n_p
            return x, y
        
    
        
        def initialize():
            random.seed(12345)
    
            sigma = np.sqrt(4./(self.L + self.M))
            init_weights = sigma * (2*np.random.rand(self.L, self.M) - 1)
            init_visible_units = sigma * (2*np.random.rand(self.L) - 1)
            init_hidden_units = np.zeros(self.M)
            
            weights = init_weights
            visible_units = init_visible_units
            hidden_units = init_hidden_units
            return weights, visible_units, hidden_units
        
        self.x1, self.y1 = create_coords(self.L, 0)
        self.x2, self.y2 = create_coords(self.M, 1)
        self.w, self.a, self.b = initialize()
        
    def activate(self, v_in, vmin, ws, bias, DE, info = False):
        act_level = np.dot(v_in, ws) + bias
        prob = 1/ (1 + np.exp(-DE*act_level))
        n = len(act_level)
        v_out = np.full(n, vmin)
        v_out[np.random.random_sample(n) < prob] = 1
        if info:
            print(f'input = {v_in} \n act = {act_level} \n prob = {prob} \n output = {v_out}')
        return v_out
        
    def train(self, data, convert = 'spin', epochs = 50, minibatch_size = 500, learning_rate = 1.0, viz = False):
        if convert == 'spin':
            data, vmin, gap = spinify(data)
        elif convert == 'digital':
            data, vmin, gap = digitize(data)
        N = len(data)
        
        for epoch in range(1,1+epochs):
            m = 0
            for n in range(N):
                if m == 0:
                    # initialise
                    v_data, v_model = np.zeros(self.L), np.zeros(self.L)
                    h_data, h_model = np.zeros(self.M), np.zeros(self.M)
                    vh_data,vh_model = np.zeros((self.L,self.M)), np.zeros((self.L,self.M))
            
                # positive CD (contrasted divergence) phase
                h = self.activate(data[n], vmin, self.w, self.b, gap)
                # negative CD phase
                vf = self.activate(h, vmin, self.w.T, self.a, gap) #generates first fantasy particle
                # positive CD phase nr  2 : generate second hidden units
                hf = self.activate(vf, vmin,  self.w, self.b, gap)
                
                v_data += data[n]
                v_model += vf #fantasy data 
                h_data += h # first hidden units generated from data
                h_model += hf
                vh_data += np.outer(data[n].T,h) #generates matrix of elements of v[i] h[j] that we use to update them
                vh_model += np.outer(vf.T, hf) #same for fantasy data
                
                m += 1 #index of minibatch 
                if m == minibatch_size: #at the end of the minibatch you do updates
                    #generate incriments prop to gradient 
                    #divide by num of data in minibatch and * learning rate
                    c = learning_rate/minibatch_size
                    dw = c*(vh_data - vh_model) #diff between data and fantasy times by constant
                    da = c*(v_data - v_model)
                    db = c*(h_data - h_model)
                    
                    if epoch <= 2 and n<=minibatch_size:
                        print("--- epoch = ", epoch, "n = ",n, "minibatch no ", m)
                        print("dw=",dw,"da=",da,"db=",db)
                    
                    self.w += dw
                    self.a += da
                    self.b += db
                    m=0

            # randomise order
            np.random.shuffle(data) #shuffle rows so in next epoch all minibatches will be different
            learning_rate = learning_rate / (0.05*learning_rate + 1) #decaying learning rate
            if viz:
                if epoch%10 == 0:
                    self.plotgraph(epoch_no = epoch)
                    print("learning_rate=",learning_rate)

            
### -------------------------------------------------      ------------------------------------------------- ###       
            

    def mycolor(self, val):
        if val > 0:
            return 'red'
        elif val < 0:
            return 'blue'
        else:
            return 'grey'
    

    def plotgraph(self, epoch_no = 0, *args):
    
        A = 2./self.w.max()
        for i in range(self.L):
            for j in range(self.M):
                ex, ey, col = (self.x1[i], self.x2[j]), (self.y1[i], self.y2[j]), self.mycolor(self.w[i][j])
                plt.plot(ex, ey, col, zorder = 1, lw = A*np.abs(self.w[i][j]))
        A = 300/(self.a.max() + self.b.max())
        for i in range(self.L):
            plt.scatter(self.x1[i], self.y1[i], s = A*np.abs(self.a[i]), zorder = 2, c = self.mycolor(self.a[i]))
        for j in range(self.M):
            plt.scatter(self.x2[j], self.y2[j],  s = A*np.abs(self.b[j]), zorder = 2, c = self.mycolor(self.b[j]))
        
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        plt.title(f'Boltzmann machine visualized, epoch = {epoch_no}')
        plt.xticks([])
        plt.yticks([])
        plt.text(max(ex) + 0.1, max(ey)-0.01, 'Visible units', fontsize = 'large')
        plt.text(max(ex)+0.1, min(ey)-0.01, 'Hidden units', fontsize = 'large')
        plt.show()