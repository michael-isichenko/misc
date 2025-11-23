#!/usr/bin/env python3

'''
Optimization of magnetic configuration created by two spirals.
Saving resuts to file, no plotting.
'''
import os
import sys
from dataclasses import dataclass, fields
from scipy.optimize import minimize
import scipy.integrate as integrate
from itertools import combinations
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

@dataclass
class Spirals:
    rho_max: float = 25
    xc1: float = -0.5     # center of inner circle
    dx: float =  1.0    # xc2 - xc1
    z1: float = -0.25   # z-height of the spiral
    z2: float = -0.35
    a1: float =  0.2    # min_radius
    a2: float =  0.2
    d1: float =  2.0    # max_radius - min_radius
    d2: float =  2.0
    Ir: float = -1.0    # I2/I1
    ex1: float = 0      # excentricity, must be less than 1 by abs value
    ex2: float = 0 
    def n1(self):  return int(np.round(self.rho_max*self.d1*(1-np.abs(self.ex1)))) # number of revolutions (circles)
    def n2(self):  return int(np.round(self.rho_max*self.d2*(1-np.abs(self.ex2))))
    def dr1(self): return 1/(self.rho_max*(1-np.abs(self.ex1))) # radius increment per revolution
    def dr2(self): return 1/(self.rho_max*(1-np.abs(self.ex2)))
    def x1(self, i):      return self.xc1 + self.ex1*i*self.dr1() # center of ith circle
    def x2(self, i):      return self.xc1 + self.ex2*i*self.dr2() + self.dx
    def r1(self, i: int): return self.a1 + i*self.dr1()           # radius of ith circle
    def r2(self, i: int): return self.a2 + i*self.dr2()
    def set(self, rho_max,xc1,dx,z1,z2,a1,a2,d1,d2,Ir,ex1,ex2):
        self.rho_max = float(rho_max)
        self.xc1 = float(xc1)
        self.dx  = float(dx)
        self.z1  = float(z1)
        self.z2  = float(z2)
        self.a1  = float(a1)
        self.a2  = float(a2)
        self.d1  = float(d1)
        self.d2  = float(d2)
        self.Ir  = float(Ir)
        self.ex1 = float(ex1)
        self.ex2 = float(ex2)
    def randomize(self, seed, bounds):
        random.seed(seed)
        self.xc1 = random.uniform(bounds.xc1[0], bounds.xc1[1])
        self.dx  = random.uniform(bounds.dx[0], bounds.dx[1])
        self.a1  = random.uniform(bounds.a1[0], bounds.a1[1])
        self.a2  = random.uniform(bounds.a2[0], bounds.a2[1])
        self.d1  = random.uniform(bounds.d1[0], bounds.d1[1])
        self.d2  = random.uniform(bounds.d2[0], bounds.d2[1])
        self.Ir  = random.uniform(bounds.Ir[0], bounds.Ir[1])
        if 0:
            self.ex1 = random.uniform(bounds.ex1[0], bounds.ex1[1])
            self.ex2 = random.uniform(bounds.ex2[0], bounds.ex2[1])
    def __str__(self): # make Params object printable
        n = 4 # precision
        return \
        f'x=({round(self.x1(0),n)},{round(self.x2(0),n)}), ' + \
        f'z({round(self.z1,n)},{round(self.z2,n)}), ' + \
        f'a({round(self.a1,n)},{round(self.a2,n)}), ' + \
        f'd({round(self.d1,n)},{round(self.d2,n)}), ' + \
        f'ex({round(self.ex1,n)},{round(self.ex2,n)}), ' + \
        f'n({self.n1()},{self.n2()}), ' + \
        f'Ir={round(self.Ir,n)}'

@dataclass
class SpiralBounds:
    xc1 = (-3, 2)
    dx = (0.1, 3)
    a1 = (0.2, 1)
    a2 = (0.2, 1)
    d1 = (1, 3)
    d2 = (1, 3)
    ex1 = (-0.5, 0.5)
    ex2 = (-0.5, 0.5)
    Ir = (-1, -0.1) # second spiral current opposite and smaller than first
    # z1,z2 fixed

def dB(x, y, z, a, t):
    cos, sin = np.cos(t), np.sin(t)
    D2 = (x - a*cos)**2 + (y - a*sin)**2 + z**2
    D3 = D2*np.sqrt(D2)
    bx = a*z*cos/D3
    by = a*z*sin/D3
    bz = a*(a - x*cos - y*sin)/D3
    return np.array([bx, by, bz])

def dB_derivs(x, y, z, a, t):
    cos, sin = np.cos(t), np.sin(t)
    D2 = (x - a*cos)**2 + (y - a*sin)**2 + z**2
    D1 = np.sqrt(D2)
    D3 = D2*D1
    D5 = D2*D3
    bxx = -3*a*z*cos*(x - a*cos)/D5
    bxz = a*cos/D3*(1 - 3*z*z/D2)
    bzx = -a*cos/D3 - 3*a*(x - a*cos)*(a - x*cos - y*sin)/D5
    bzz = 3*a*z*(x*cos + y*sin - a)/D5
    return np.array([bxx, bxz, bzx, bzz]) # only some of them

def loop_B(x, y, z, a):
    return integrate.quad_vec(lambda t: dB(x, y, z, a, t), 0, 2*np.pi)[0]

def loop_B_derivs(x, y, z, a):
    return integrate.quad_vec(lambda t: dB_derivs(x, y, z, a, t), 0, 2*np.pi)[0]

def add_to(lhs, rhs):
    return rhs if lhs is None else lhs + rhs

def total_B(x, y, z, spirals): # (Bx, By, Bz)
    B1, B2 = None, None
    for i in range(spirals.n1()): B1=add_to(B1, loop_B(x - spirals.x1(i), y, z - spirals.z1, spirals.r1(i)))
    for i in range(spirals.n2()): B2=add_to(B2, loop_B(x - spirals.x2(i), y, z - spirals.z2, spirals.r2(i)))
    Bx = B1[0] + spirals.Ir*B2[0]
    By = B1[1] + spirals.Ir*B2[1]
    Bz = B1[2] + spirals.Ir*B2[2]
    return np.array([Bx, By, Bz])

def total_B_derivs(x, y, z, spirals):
    D1, D2 = None, None
    for i in range(spirals.n1()): D1=add_to(D1, loop_B_derivs(x - spirals.x1(i), y, z - spirals.z1, spirals.r1(i)))
    for i in range(spirals.n2()): D2=add_to(D2, loop_B_derivs(x - spirals.x2(i), y, z - spirals.z2, spirals.r2(i)))
    return D1 + spirals.Ir*D2 # (Bxx, Bxz, Bzx, Bzz)

def compute_jac_eigenvector(Bxx, Bxz, Bzx, Bzz):
    C = Bzz - Bxx
    D = np.sqrt(C*C + 4*Bxz*Bzx)
    evec = (C + D, 2*Bxz) # the other one is (C - D, 2*Bxz)
    return evec

def grads_B(Bxx, Bxz, Bzx, Bzz):
    # instead of compute_jac_eigenvector:
    assert np.abs(Bxz - Bzx) <= 1e-6*(np.abs(Bxz) + np.abs(Bzx))
    C = Bzz - Bxx
    D = np.sqrt(C*C + 4*Bxz*Bzx)
    vx1, vx2, vz = C + D, C - D, 2*Bxz
    grad1 = (vx1**2 + 2*vx1*vz*Bxz + vz**2*Bzz)/(vx1**2 + vz**2)
    grad2 = (vx2**2 + 2*vx2*vz*Bxz + vz**2*Bzz)/(vx2**2 + vz**2)
    return grad1[0], grad2[0]

def get_angle(evec): # in [-pi, pi]
    angle = np.atan2(evec[0], evec[1]) # in [-pi, pi]
    if angle < 0:
        angle += np.pi
    if angle > np.pi/2:
        angle -= np.pi/2
    return angle

def get_relevant_angle_slope_and_grad(Bxx, Bxz, Bzx, Bzz):
    C = Bzz - Bxx
    D = np.sqrt(C*C + 4*Bxz*Bzx)
    vx1, vx2, vz = C + D, C - D, 2*Bxz
    slope1 = np.abs(vz)/(np.abs(vx1) + 1e-10)
    slope2 = np.abs(vz)/(np.abs(vx2) + 1e-10)
    if slope1 < slope2:
        angle = np.atan2(vx1, vz)
        slope = slope1
        grad = (vx1**2 + 2*vx1*vz*Bxz + vz**2*Bzz)/(vx1**2 + vz**2)
    else:
        angle = np.atan2(vx2, vz)
        slope = slope2
        grad = (vx2**2 + 2*vx2*vz*Bxz + vz**2*Bzz)/(vx2**2 + vz**2)
    return slope, np.abs(grad)

def search_params(null_p, vars, pp):
    # pp is default/initial params
    bb = SpiralBounds()
    null_x, null_y, null_z = null_p
    angle = None
    grads = None
    def angle_utility(evec):
        nonlocal angle
        angle = get_angle(evec)
        deg = angle*180/np.pi # in [0, 90]
        return ((deg - 36)*(deg - 56)/8100)**2
    def utility(values):
        nonlocal grads
        for var, value in zip(vars, values):
            setattr(pp, var, value) # set pp.<var> = value
        B = total_B(null_x, null_y, null_z, pp)
        field_utility = np.sum(np.square(B))
        Bxx, Bxz, Bzx, Bzz = total_B_derivs(null_x, null_y, null_z, pp)
        evec = compute_jac_eigenvector(Bxx, Bxz, Bzx, Bzz)
        grads = grads_B(Bxx, Bxz, Bzx, Bzz)
        return field_utility + angle_utility(evec)
    initial_values = [getattr(pp, var) for var in vars]
    bounds         = [getattr(bb, var) for var in vars]
    result = minimize(utility, initial_values, bounds=bounds)
    return result, angle, grads

def search_params_with_grad(null_p, vars, pp, tau, gamma):
    T = np.tan(36.0*np.pi/180)
    # pp is default/initial params
    bb = SpiralBounds()
    null_x, null_y, null_z = null_p
    angle = None
    grad = None
    def utility(values):
        nonlocal angle
        nonlocal grad
        for var, value in zip(vars, values):
            setattr(pp, var, value) # set pp.<var> = value
        B = total_B(null_x, null_y, null_z, pp)
        Bxx, Bxz, Bzx, Bzz = total_B_derivs(null_x, null_y, null_z, pp)
        angle, slope, grad = get_relevant_angle_slope_and_grad(Bxx, Bxz, Bzx, Bzz)
        return np.sum(np.square(B)) + tau*(slope - T)**2 - gamma*grad**2
    initial_values = [getattr(pp, var) for var in vars]
    bounds         = [getattr(bb, var) for var in vars]
    result = minimize(utility, initial_values, bounds=bounds)
    return result, angle, grad

def write_result(fname, idx, seed, vars, result, angle, grads):
    # print(f'XXX {grads}')
    pp = Spirals()
    ff = [field.name for field in fields(pp)]
    if not os.path.isfile(fname):
        fp = open(fname, 'w')
        ff_csv = ','.join(ff)
        print(f'idx,seed,dim,vars,{ff_csv},n1,n2,U,angle,grad1,grad2', file=fp)
    else:
        fp = open(fname, 'a')
    for var, value in zip(vars, result.x):
        setattr(pp, var, value)
    values = [str(getattr(pp, var)) for var in ff]
    print(f'{idx},{seed},{len(vars)},{" ".join(vars)},{",".join(values)},{pp.n1()},{pp.n2()},{result.fun},{angle[0]},{grads[0]},{grads[1]}', file=fp)
    fp.close()

def write_result2(fname, idx, seed, vars, result, angle, grad, tau, gamma):
    # print(f'XXX {grads}')
    pp = Spirals()
    ff = [field.name for field in fields(pp)]
    if not os.path.isfile(fname):
        fp = open(fname, 'w')
        ff_csv = ','.join(ff)
        print(f'idx,seed,tau,gamma,{ff_csv},n1,n2,U,angle,grad', file=fp)
    else:
        fp = open(fname, 'a')
    for var, value in zip(vars, result.x):
        setattr(pp, var, value)
    values = [str(getattr(pp, var)) for var in ff]
    print(f'{idx},{seed},{tau},{gamma},{",".join(values)},{pp.n1()},{pp.n2()},{result.fun},{angle[0]},{grads[0]},{grads[1]}', file=fp)
    fp.close()

def optimize_over_spiral_combinations(null_p, play_vars, dim, key, beg_seed, nrand):
    vars_tuples = list(combinations(play_vars, dim))
    for idx, vars in enumerate(vars_tuples):
        pp = Spirals()
        for rand in range(nrand):
            seed = beg_seed + rand
            pp.randomize(seed, SpiralBounds())
            print(f'{idx}:{seed} starting with {pp}')
            result, angle, grads = search_params(null_p, vars, pp)
            fname = f'{key}.multiloop.{os.getpid()}.csv'
            write_result(fname, idx, seed, vars, result, angle*180/np.pi, grads)

def optimize_over_spirals(null_p, key, beg_seed, nrand):
    vars = ['xc1', 'dx', 'a1', 'a2', 'd1', 'd2', 'ex1', 'ex2', 'Ir'] # search space
    pp = Spirals()
    tau = 50
    gamma = 0.001
    for rand in range(nrand):
        seed = beg_seed + rand
        pp.randomize(seed, SpiralBounds())
        print(f'{idx}:{seed} starting with {pp}')
        result, angle, grad = search_params_with_grad(null_p, vars, pp, tau, gamma)
        fname = f'{key}.spirals.{os.getpid()}.csv'
        write_result2(fname, idx, seed, vars, result, angle*180/np.pi, grad, tau, gamma)

def run_seed(seed, null_p, play_vars, dim, key):
    # optimize_over_spiral_combinations(null_p, play_vars, dim, key, seed, nrand=1000)
    optimize_over_spirals(null_p, key, seed, nrand=100)
                
def do_runs(noun, null_p, play_vars, dim, key):
    if noun == 'multi':
        seeds = [1000*i for i in range(8)]
        for seed in seeds:
            print(f'{sys.argv[0]} run {seed}')
    else:
        run_seed(int(noun), null_p, play_vars, dim, key)

'''
plotting
'''
def spiral_demo():
    '''
    A spiral is modeled by a family of concentric circles.  This
    looks okay, because the mean radial current due to the converging
    spiral is compensated by the outgoing straight wire in the opposite
    direction.
    '''
    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(1, 1, 1)
    for i in range(10):
        ax.add_patch(Circle((-1.5, 0), 0.5+0.1*i, lw=1, alpha=0.7,
                            facecolor='none', edgecolor='blue'))
    tt = np.linspace(0, 20*np.pi, 1000, endpoint=False)
    rr = 0.5 + 1.0*tt/(20*np.pi)
    xx = 1.5 + rr*np.cos(tt)
    yy = 0.0 + rr*np.sin(tt)
    ax.plot(xx, yy, color='red', lw=1)
    ax.add_artist(lines.Line2D([xx[0], xx[-1]+0.1], [yy[0], 0], linewidth=1, color='red'))
    ax.add_artist(lines.Line2D([xx[-1], xx[-1]+0.1], [yy[-1], yy[-1]], linewidth=1, color='red'))
    ax.set_aspect('equal')
    #plt.show()
    plt.savefig('tmp.pdf')

def ex_spiral_demo():
    '''
    Excentric spiral.  Both radius and x-center increase linearly with polar angle.
    '''
    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(1, 1, 1)
    a, d, ex = 0.5, 2.5, 0.8
    n = 50
    tmax = 2*np.pi*n
    tt = np.linspace(0, tmax, n*100, endpoint=True)
    rr = a + d*tt/(2*np.pi*n)
    xx = d*ex*tt/tmax + rr*np.cos(tt)
    yy = rr*np.sin(tt)
    ax.plot(xx, yy, color='green', lw=0.5)
    ax.set_aspect('equal')

'''
Display (better) results of optimization.
'''
def mark_spiral(ax, x, z, a, d, ex, n, color):
    dr = d/(n - 1) 
    # dr = d/n XXX
    for i in range(n):
        xi = x + i*dr*ex
        ri = a + i*dr
        ax.add_patch(Circle((xi - ri, z), 0.01, color=color, alpha=0.7))
        ax.add_patch(Circle((xi + ri, z), 0.01, color=color, alpha=0.7))
    ax.add_artist(lines.Line2D([xi - ri, xi + ri], [z, z], linewidth=1, color=color))

def mark_x(ax, evec, null_p):
    null_x, null_y, null_z = null_p
    if 0: # small marker
        ax.scatter(null_x, null_z, s=50, color='brown', marker='X')
    R = 0.5 # wheel radius
    lw = 0.3
    theta = np.atan2(evec[0], evec[1])
    for i in range(4):
        ax.add_artist(lines.Line2D([null_x, null_x + R*np.cos(theta+i*np.pi/2)],
                                   [null_z, null_z + R*np.sin(theta+i*np.pi/2)],
                                   linewidth=lw, color='blue'))
    ax.add_patch(Circle((null_x, null_z), R, fill=False, color='blue', linewidth=lw))
    #print(f'theta={theta}') # theta=[0.22812388]
    theta = theta[0]
    if theta > np.pi/2:
        theta -= np.pi/2
    return theta*180/np.pi

def plot_spirals(ax, pp):
    for i in range(pp.n1()): ax.add_patch(Circle((pp.x1(i), 0), pp.r1(i), facecolor='none', edgecolor='blue', alpha=0.7))
    for i in range(pp.n2()): ax.add_patch(Circle((pp.x2(i), 0), pp.r2(i), facecolor='none', edgecolor='green', alpha=0.7))

def plot_field(ax, pp, null_p, title):
    focus_on_x = False # draw only in the vicinity of the X-point
    if focus_on_x:
        xmin, xmax = -1, 1
        zmin, zmax =  0, 1.5
    else:
        xmin, xmax = -5, 5
        zmin, zmax = 0, 5
    xx, zz = np.linspace(xmin, xmax, 150), np.linspace(zmin, zmax, 150)
    xx_grid, zz_grid = np.meshgrid(xx, zz)
    B = total_B(xx_grid, 0, zz_grid, pp)
    # print(B)
    if 1:
        mark_spiral(ax, pp.xc1,         pp.z1, pp.a1, pp.d1, pp.ex1, pp.n1(), 'blue')
        mark_spiral(ax, pp.xc1 + pp.dx, pp.z2, pp.a2, pp.d2, pp.ex2, pp.n2(), 'red')
        null_x, null_y, null_z = null_p
        Bxx, Bxz, Bzx, Bzz = total_B_derivs(null_x, null_y, null_z, pp)
        evec = compute_jac_eigenvector(Bxx, Bxz, Bzx, Bzz)
        deg = mark_x(ax, evec, null_p)
        grads = grads_B(Bxx, Bxz, Bzx, Bzz)
    ax.streamplot(xx_grid, zz_grid, B[0], B[2], density=2, color='g',
                    linewidth=0.5, cmap=plt.cm.viridis, arrowsize=0.8)
    ax.set_aspect('equal')
    ax.set_title(f'{title}: ang={round(deg, 2)} grads={round(grads[0],2)},{round(grads[1],2)}', fontsize=6)

def plot_field_and_spirals(pdf, idx, seed, pp, null_p):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 12))
    ax1.set_ylim(-1, 5)
    ax2.set_ylim(-5, 5)
    for ax in (ax1, ax2):
        ax.set_xlim(-5, 5)
        ax.set_aspect('equal')
    plt.tight_layout()
    ax1.hlines(y=0, xmin=-5, xmax=5, colors='black', linestyles='--', lw=1)
    title = f'Run {idx}:{seed}: {pp}'
    plot_field(ax1, pp, null_p, title)
    plot_spirals(ax2, pp)
    print(f'# {{{pdf}}}')
    plt.savefig(pdf)

def do_plots(noun, null_p):
    if 0: spiral_demo()
    if 0: ex_spiral_demo()
    fname = noun
    with open(fname) as file:
        for line in file:
            if line.startswith('idx,seed,dim,vars,rho_max,xc1,dx,z1,z2,a1,a2,d1,d2,Ir,ex1,ex2,n1,n2,U,angle'):
                continue
            idx,seed,dim,vars,rho_max,xc1,dx,z1,z2,a1,a2,d1,d2,Ir,ex1,ex2,n1,n2,U,angle,grad1,grad2 = line.rstrip().split(',')
            if float(U) > 1e-6:
                continue
            pp = Spirals()
            pp.set(rho_max,xc1,dx,z1,z2,a1,a2,d1,d2,Ir,ex1,ex2)
            pdf = fname.replace('.csv', '')
            pdf = f'{pdf}.{idx}.{seed}.pdf'
            plot_field_and_spirals(pdf, idx, seed, pp, null_p)

def do_join(fnames):
    header = False
    for fname in fnames:
        with open(fname) as file:
            for line in file:
                if line.startswith('idx'):
                    if not header:
                        print(line.rstrip())
                        header = True
                    else:
                        continue
                print(line.rstrip())
                
if __name__ == '__main__':
    null_p = np.array([0.0]), np.array([0.0]), np.array([0.98])
    assert len(sys.argv) >= 3
    verb, noun = sys.argv[1], sys.argv[2]
    if verb == 'plot':
        do_plots(noun, null_p)
    elif verb == 'run':
        play_vars = ['xc1', 'dx', 'a1', 'a2', 'd1', 'd2', 'ex1', 'ex2', 'Ir']
        # play_vars = ['xc1', 'dx', 'a1', 'a2', 'd1', 'd2', 'Ir'] # no ex
        dim = len(play_vars) # how many parameters (out ot 9) to play with at a time
        key = datetime.now().strftime("%Y%m%d.%H%M%S") # when started
        do_runs(noun, null_p, play_vars, dim, key)
    elif verb == 'join':
        assert len(sys.argv) > 3
        do_join(sys.argv[2:])
    else:
        assert False
