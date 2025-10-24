#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def dipole_field(x, y, a, m): # https://en.wikipedia.org/wiki/Magnetic_dipole
    r2 = (x-a)**2 + y**2
    r3 = r2*np.sqrt(r2)
    r5 = r3*r2
    Bx = 3*m*y*(x-a)/r5
    By = 3*m*y**2/r5 - m/r3
    return Bx, By

def total_field(x, y, a1, a2, m1, m2):
    B1 = dipole_field(x, y, a1, m1)
    B2 = dipole_field(x, y, a2, m2)
    return B1[0] + B2[0], B1[1] + B2[1]

if __name__ == '__main__':
    # make two separate plot windows
    fig1 = plt.figure(figsize=(8, 8))
    ax1  = fig1.add_subplot(1, 1, 1)
    fig2 = plt.figure(figsize=(8, 8))
    ax2  = fig2.add_subplot(1, 1, 1)

    # plotting domain and mash
    xmin, xmax = -5, 5
    ymin, ymax =  0, 5
    xx, yy = np.linspace(xmin, xmax, 150), np.linspace(ymin, ymax, 150)
    xx_grid, yy_grid = np.meshgrid(xx, yy)

    # magnetic dipoles in plane x=0
    a1, a2 = -2, 2  # dipoles' x positions
    m1, m2 = 1, -10 # dipoles' strength
    Bx, By = total_field(xx_grid, yy_grid, a1, a2, m1, m2)

    if 1: # field
        ax1.streamplot(xx_grid, yy_grid, Bx, By, density=2.5, color='b',
                       linewidth=1, cmap=plt.cm.viridis, arrowsize=0.5)
        ax1.set_title('Vector B')
    if 1: # abs value and its gradient
        B = np.sqrt(Bx**2 + By**2)
        grad_y, grad_x = np.gradient(B, yy, xx)
        ax2.streamplot(xx_grid, yy_grid, grad_x, grad_y, density=2.5, color='b',
                       linewidth=1, cmap=plt.cm.viridis, arrowsize=0.5)
        ax2.set_title('grad(|B|)')

    for ax in [ax1, ax2]:
        # Add circles representing the dipole sources
        ax.add_artist(plt.Circle((a1, 0), 0.05, color='r', zorder=2))
        ax.add_artist(plt.Circle((a2, 0), 0.1, color='r', zorder=2))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    plt.show()
