from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np



def sph2cart(r,theta,phi):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return(x,y,z)


plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z,  rstride=4, cstride=4,linewidth=0.5, color='#b5651d')
scats = []

def animate(fps):
    global scats
    for scat in scats:
        scat.remove()
    scats=[]
    theta = np.random.randint(-180,180)
    phi = np.random.randint(-90,90)
    x,y,z = sph2cart(r=1, theta=theta, phi=phi)
    scats.append(ax.scatter(x,y,z,color='b'))
    ax.set_title(u"Azimuth = {}°       Elevation: {}°".format(theta, phi))
    plt.draw()

    

anim = matplotlib.animation.FuncAnimation(fig, animate, 50, 
                                interval=200, blit=False)


plt.show(block=True)

print("exitedfoo")


