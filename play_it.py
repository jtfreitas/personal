from conway import game_of_life
import matplotlib.pyplot as plt
from matplotlib import animation
g = game_of_life(50, 50, 8585)

g.start(3)
def play_the_game(rounds):
    
    def init():
        im.set_data(g.the_grid)
        return im,

    def animate(i):
        g.update()
        im.set_data(g.the_grid)
        return im,

    g.start(5)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 50), ylim=(0, 50))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = plt.imshow(g.the_grid, cmap = 'gray')
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = rounds, blit = True)
    anim.save('game_of_life.gif', fps=10)#, extra_args=['-vcodec', 'libx264'])
    
    
play_the_game(200)


#good seed: 7970