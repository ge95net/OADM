import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/yy/research/CrowdNav-master1_copy/CrowdNav-master')
from crowd_sim.envs.utils.robot import Robot


def plot(danger_map):
    # ax.set_xlabel('orientation')
    # ax.set_ylabel('distance')
    '''
    ax.set_thetagrids(np.arange(0.0,360.0,10.0))
    ax.set_thetamin(0.0)
    ax.set_thetamax(180.0)
    ax.grid(True,linestyle='-',color='k',linewidth=0.5,alpha=0.5)
    ax.pcolormesh(danger_map,cmap = 'gray_r')
    '''
    print("plot===",danger_map)


    theta = np.linspace(-120, 120, 24) / 180 * np.pi
    r = np.linspace(0, 5, 5)

    theta, r = np.meshgrid(theta, r)
    figure = plt.figure()
    ax = figure.gca(projection='polar')
    ax.set_title('danger_map')
    ax.set_rgrids([100, 200, 300, 400])
    im = ax.contourf(theta, r, danger_map, cmap='gray_r')
    figure.colorbar(im, ax=ax)
    # sns.heatmap(danger_map,cmap='gray_r')
    ani = FuncAnimation(figure,update, interval=50)
    # label_y = ax.get_yticklabels()
    # plt.xticks(range(1,25,1),range(-120,120,10))
    # plt.setp(label_y, rotation=0, horizontalalignment='right')
    # label_x = ax.get_xticklabels()
    # plt.setp(label_x, rotation=90, horizontalalignment='right')
    plt.show()
    plt.close()


def update(self, i):
    danger_map = self.robot.get_lidar_reading()
    theta = np.linspace(-120, 120, 24) / 180 * np.pi
    r = np.linspace(0, 5, 5)
    theta, r = np.meshgrid(theta, r)
    im = self.ax.contourf(theta, r, danger_map, cmap='gray_r')
    return im


if __name__ == "__main__":
    danger_map = sys.argv[1]
    plot(danger_map)
