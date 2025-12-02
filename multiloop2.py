#!/usr/bin/env python3

'''
Optimization of magnetic configuration created by two spirals.
Saving resuts to file, no plotting.
'''
import os
import sys
from dataclasses import dataclass, fields
from scipy.optimize import minimize, NonlinearConstraint
import scipy.integrate as integrate
from itertools import combinations
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

NL:  int   = 2     # number of layers
Z1:  float = -0.25 # z-height of 1st spiral - blue
Z2:  float = -0.35 # z-height of 2nd spiral - green
RHO: float = 25    # max loop density
X:   float = 5     # half-size of the board
assert NL == 2
USE_NONLINEAR_CONSTRAINT = False # it can't

def ZZ(k):
    if 0 == k: return Z1
    else:      return Z2

def colors(k):
    if 0 == k: return 'blue'
    else:      return 'green'

# helper geometry functions:
def help_n(a, b, c, dx):
    if False:
        # number of circles, eq (21)
        dxp = dx if dx > 0 else 0.0
        dxm = dx if dx < 0 else 0.0
        return int(min((X-a-c)/(1.0/RHO + 2.0*dxp),
                       (X-a+c)/(1.0/RHO - 2.0*dxm)))
    else:
        if RHO*(b - a - np.abs(dx)) < 0:
            print(f'a={a} b={b} dx={dx} n={RHO*(b - a - np.abs(dx))}')
        return int(RHO*(b - a - np.abs(dx)))

def help_dr(dx):   # radius increment per revolution, eq (18)
    return 1.0/RHO + np.abs(dx)

@dataclass
class SpiralBounds:
    a1  = (0.2, 0.5)
    a2  = (0.2, 0.4)
    b1  = (1.0, 9.0)
    b2  = (1.0, 9.0)
    c1  = (-4.0, -1.0)
    c2  = (1.0, 4.0)
    dx1 = (-0.038, 0)
    dx2 = (0, 0.038)
    Ir  = (-1, -0.5) # lower 2nd spiral current
    #Ir  = (0.5, 1) # same direction

'''
20251127:182841 Ir > 0 and more separated centers
'''

@dataclass
class Spirals:
    a1:  float =  0.2  # min_radius
    a2:  float =  0.2
    b1:  float =  1.5  # max_radius
    b2:  float =  1.0
    c1:  float = -2.0  # x-position of center of inner circle
    c2:  float =  1.5
    dx1: float =  0    # optional center x-shift per loop (excentricity)
    dx2: float =  0 
    Ir:  float =  1.0  # I2/I1
    # dependent properties:
    def n(self, k): # number of circles
        if 0 == k: return help_n(self.a1, self.b1, self.c1, self.dx1)
        else:      return help_n(self.a2, self.b2, self.c2, self.dx2)
    def xx(self, k): # centers circles
        if 0 == k: return [self.c1 + self.dx1*i for i in range(self.n(k))]
        else:      return [self.c2 + self.dx2*i for i in range(self.n(k))]
    def rr(self, k): # radii of circles
        if 0 == k: return [self.a1 + i*help_dr(self.dx1) for i in range(self.n(k))]
        else:      return [self.a2 + i*help_dr(self.dx2) for i in range(self.n(k))]
    def geom_penalty(self):
        pen = 0
        for k in range(NL):
            n = self.n(k)
            if n < 10:
                pen += (10 - n)
            x = self.xx(n)[-1]
            r = self.rr(n)[-1]
            xleft = x - r
            xright = x + r
            if xleft < -X: pen += (-X - xleft)
            if xright > X: pen += (xright - X)
        return pen
    def set(self, a1,a2,b1,b2,c1,c2,dx1,dx2,Ir):
        self.a1  = float(a1)
        self.a2  = float(a2)
        self.b1  = float(b1)
        self.b2  = float(b2)
        self.c1  = float(c1)
        self.c2  = float(c2)
        self.dx1 = float(dx1)
        self.dx2 = float(dx2)
        self.Ir  = float(Ir)
    def randomize(self, vars, seed, bounds):
        # print(f'seed={seed}', file=sys.stderr)
        random.seed(seed)
        if 'a1 ' in vars: self.a1  = random.uniform(bounds.a1[0],  bounds.a1[1])
        if 'a2 ' in vars: self.a2  = random.uniform(bounds.a2[0],  bounds.a2[1])
        if 'b1'  in vars: self.b1  = random.uniform(bounds.b1[0],  bounds.b1[1])
        if 'b2 ' in vars: self.b2  = random.uniform(bounds.b2[0],  bounds.b2[1])
        if 'c1'  in vars: self.c1  = random.uniform(bounds.c1[0],  bounds.c1[1])
        if 'c2 ' in vars: self.c2  = random.uniform(bounds.c2[0],  bounds.c2[1])
        if 'dx1' in vars: self.dx1 = random.uniform(bounds.dx1[0], bounds.dx1[1])
        if 'dx2' in vars: self.dx2 = random.uniform(bounds.dx2[0], bounds.dx2[1])
        if 'Ir'  in vars: self.Ir  = random.uniform(bounds.Ir[0],  bounds.Ir[1])
        
    def __str__(self): # make Params object printable
        prec = 4 # precision
        return \
        f'a({round(self.a1,prec)},{round(self.a2,prec)}), ' + \
        f'b=({round(self.b1,prec)},{round(self.b2,prec)}), ' + \
        f'c=({round(self.c1,prec)},{round(self.c2,prec)}), ' + \
        f'dx({round(self.dx1,prec)},{round(self.dx2,prec)}), ' + \
        f'n({self.n(0)},{self.n(1)}), ' + \
        f'Ir={round(self.Ir,prec)}'

def dB(x, y, z, r, t): # infinitesimal field of loop with radius r, eq (3)
    cos, sin = np.cos(t), np.sin(t)
    D2 = (x - r*cos)**2 + (y - r*sin)**2 + z**2
    D3 = D2*np.sqrt(D2)
    bx = r*z*cos/D3
    by = r*z*sin/D3
    bz = r*(r - x*cos - y*sin)/D3
    return np.array([bx, by, bz])

def dB_derivs(x, y, z, r, t):
    cos, sin = np.cos(t), np.sin(t)
    D2 = (x - r*cos)**2 + (y - r*sin)**2 + z**2
    D1 = np.sqrt(D2)
    D3 = D2*D1
    D5 = D2*D3
    bxx = -3*r*z*cos*(x - r*cos)/D5
    bxz = r*cos/D3*(1 - 3*z*z/D2)
    bzx = -r*cos/D3 - 3*r*(x - r*cos)*(r - x*cos - y*sin)/D5
    bzz = 3*r*z*(x*cos + y*sin - r)/D5
    return np.array([bxx, bxz, bzx, bzz]) # only some of them

def loop_B(x, y, z, r):
    return integrate.quad_vec(lambda t: dB(x, y, z, r, t), 0, 2*np.pi)[0]

def loop_B_derivs(x, y, z, r):
    return integrate.quad_vec(lambda t: dB_derivs(x, y, z, r, t), 0, 2*np.pi)[0]

def add_to(lhs, rhs):
    return rhs if lhs is None else lhs + rhs

def single_B(x, y, z, spirals, k): # (Bx, By, Bz)
    B = None
    for xi, ri in zip(spirals.xx(k), spirals.rr(k)):
        B = add_to(B, loop_B(x - xi, y, z - ZZ(k), ri))
    return B if 0 == k else spirals.Ir*B
    
def total_B(x, y, z, spirals): # (Bx, By, Bz)
    for k in range(NL):
        assert spirals.n(k) > 0, f'{spirals}'
    B = [None, None]
    for k in range(NL):
        for xi, ri in zip(spirals.xx(k), spirals.rr(k)):
            B[k] = add_to(B[k], loop_B(x - xi, y, z - ZZ(k), ri))
    B = B[0] + spirals.Ir*B[1]
    # print(B, file=sys.stderr)
    return B

def total_B_derivs(x, y, z, spirals):
    D = [None, None]
    for k in range(NL):
        for xi, ri in zip(spirals.xx(k), spirals.rr(k)):
            D[k] = add_to(D[k], loop_B_derivs(x - xi, y, z - ZZ(k), ri))
    return D[0] + spirals.Ir*D[1] # (Bxx, Bxz, Bzx, Bzz)

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
    return angle, slope, np.abs(grad)

def search_params(null_p, vars, pp, tau, gamma):
    T = np.tan(36.0*np.pi/180)
    # pp is default/initial params
    bb = SpiralBounds()
    null_x, null_y, null_z = null_p
    B_at_null = None
    angle = None
    grad = None
    def utility(values):
        nonlocal B_at_null
        nonlocal angle
        nonlocal grad
        for var, value in zip(vars, values):
            setattr(pp, var, value) # set pp.<var> = value
        if False and not pp.is_valid():
            B_at_null = [1e6]
            angle = [0.0]
            grad = [0.0]
            return 1e6
        B = total_B(null_x, null_y, null_z, pp)
        B2 = np.sum(np.square(B))
        B_at_null = np.sqrt(B2)
        Bxx, Bxz, Bzx, Bzz = total_B_derivs(null_x, null_y, null_z, pp)
        angle, slope, grad = get_relevant_angle_slope_and_grad(Bxx, Bxz, Bzx, Bzz)
        return B2 + tau*(slope - T)**2 - gamma*grad**2 + pp.geom_penalty()
        # return np.sum(np.abs(B)) + tau*np.abs(slope - T) - gamma*grad**2
    initial_values = [getattr(pp, var) for var in vars]
    bounds         = [getattr(bb, var) for var in vars]
    def callback(x):
        #print(f'U({x}) = {utility(x)}', file=sys.stderr, flush=True)
        print(f'U({", ".join([str(v) for v in x])}) = {utility(x)}', file=sys.stderr, flush=True)
    result = minimize(utility, initial_values, bounds=bounds) # , callback=callback)
    return result, B_at_null, angle, grad

def search_params_constraint(null_p, vars, pp, tau, gamma):
    T = np.tan(36.0*np.pi/180)
    bb = SpiralBounds()
    null_x, null_y, null_z = null_p
    B_at_null = [1e6]
    angle = [0.0]
    grad = [0.0]

    def B2_fun(values):
        nonlocal B_at_null
        for var, value in zip(vars, values):
            setattr(pp, var, value) # set pp.<var> = value
        B = total_B(null_x, null_y, null_z, pp)
        B2 = np.sum(np.square(B))
        B_at_null = np.sqrt(B2)
        return B2

    def slope_fun(values):
        for var, value in zip(vars, values):
            setattr(pp, var, value) # set pp.<var> = value
        Bxx, Bxz, Bzx, Bzz = total_B_derivs(null_x, null_y, null_z, pp)
        angle, slope, grad = get_relevant_angle_slope_and_grad(Bxx, Bxz, Bzx, Bzz)
        return slope - T

    B2_con    = NonlinearConstraint(B2_fun, 0.0, 0.0)
    slope_con = NonlinearConstraint(slope_fun, 0.0, 0.0)
        
    def utility(values):
        nonlocal B_at_null
        nonlocal angle
        nonlocal grad
        for var, value in zip(vars, values):
            setattr(pp, var, value) # set pp.<var> = value
        Bxx, Bxz, Bzx, Bzz = total_B_derivs(null_x, null_y, null_z, pp)
        angle, slope, grad = get_relevant_angle_slope_and_grad(Bxx, Bxz, Bzx, Bzz)
        return -grad**2
    initial_values = [getattr(pp, var) for var in vars]
    bounds         = [getattr(bb, var) for var in vars]
    def callback(x):
        #print(f'U({x}) = {utility(x)}', file=sys.stderr, flush=True)
        print(f'U({", ".join([str(v) for v in x])}) = {utility(x)}', file=sys.stderr, flush=True)
    result = minimize(utility, initial_values, bounds=bounds, constraints=[B2_con, slope_con]) # , callback=callback)
    return result, B_at_null, angle, grad

def write_result(fname, seed, vars, result, B_at_null, angle, grad, tau, gamma):
    # print(f'XXX {grads}')
    pp = Spirals()
    ff = [field.name for field in fields(pp)]
    if not os.path.isfile(fname):
        fp = open(fname, 'w')
        ff_csv = ','.join(ff)
        print(f'seed,tau,gamma,{ff_csv},n1,n2,U,B_at_null,angle,grad', file=fp)
    else:
        fp = open(fname, 'a')
    for var, value in zip(vars, result.x):
        setattr(pp, var, value)
    values = [str(getattr(pp, var)) for var in ff]
    if not isinstance(B_at_null, float): B_at_null = B_at_null[0]
    if not isinstance(angle, float): angle = angle[0]
    if not isinstance(grad, float): grad = grad[0]
    print(f'{seed},{tau},{gamma},{",".join(values)},{pp.n(0)},{pp.n(1)},{result.fun},{B_at_null},{angle},{grad}', file=fp)
    fp.close()

def optimize_over_spirals(null_p, key, vars, beg_seed, nrand):
    pp = Spirals()
    tau = 50
    gamma = 0 # 1e-6
    for rand in range(nrand):
        seed = beg_seed + rand
        pp.randomize(vars, seed, SpiralBounds())
        print(f'seed={seed} starting with {pp}')
        if USE_NONLINEAR_CONSTRAINT:
            result, B_at_null, angle, grad = search_params_constraint(null_p, vars, pp, tau, gamma)
        else:
            result, B_at_null, angle, grad = search_params(null_p, vars, pp, tau, gamma)
        fname = f'{key}.spirals.{os.getpid()}.csv'
        if not isinstance(B_at_null, float): B_at_null = B_at_null[0]
        if not isinstance(angle, float): angle = angle[0]
        if not isinstance(grad, float): grad = grad[0]
        write_result(fname, seed, vars, result, B_at_null, angle*180/np.pi, grad, tau, gamma)

def run_seed(seed, null_p, vars, dim, key):
    optimize_over_spirals(null_p, key, vars, seed, nrand=200)
                
def do_runs(noun, null_p, vars, dim, key):
    if noun == 'multi':
        seeds = [1000*i for i in range(8)]
        for seed in seeds:
            print(f'{sys.argv[0]} run {seed}')
    else:
        seed = int(noun)
        run_seed(int(noun), null_p, vars, dim, key)

def do_demo(noun):
    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(1, 1, 1)
    if noun == 'circles':
        '''
        A spiral is modeled by a family of concentric circles.  This
        looks okay, because the mean radial current due to the converging
        spiral is compensated by the outgoing straight wire in the opposite
        direction.
        '''
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
    elif noun == 'excentric':
        '''
        Excentric spiral.  Both radius and x-center increase linearly with polar angle.
        '''
        a, d, ex = 0.5, 2.5, 0.8
        n = 50
        tmax = 2*np.pi*n
        tt = np.linspace(0, tmax, n*100, endpoint=True)
        rr = a + d*tt/(2*np.pi*n)
        xx = d*ex*tt/tmax + rr*np.cos(tt)
        yy = rr*np.sin(tt)
        ax.plot(xx, yy, color='green', lw=0.5)
        ax.set_aspect('equal')
    elif noun == '2loops':
        xmin, xmax = -3, 2
        zmin, zmax = 0, 2
        xx, zz = np.linspace(xmin, xmax, 150), np.linspace(zmin, zmax, 150)
        xx_grid, zz_grid = np.meshgrid(xx, zz)
        B = [None, None]
        xx = [-2, 1.5]
        rr = [1, 0.5]
        zz = [-0.05, -0.10]
        II = [2, 1]
        cc = ['blue', 'green']
        density = 0.75
        for i, (xi, ri) in enumerate(zip(xx, rr)):
            B[i] = loop_B(xx_grid - xi, 0, zz_grid -zz[i], ri)
            ax.add_patch(Circle((xi - ri, zz[i]), 0.03, color=cc[i], alpha=0.7))
            ax.add_patch(Circle((xi + ri, zz[i]), 0.03, color=cc[i], alpha=0.7))
            ax.add_artist(lines.Line2D([xi - ri,  xi + ri], [zz[i], zz[i]], linewidth=II[i], color=cc[i]))
            ax.streamplot(xx_grid, zz_grid, B[i][0], B[i][2], density=density, color=cc[i],
                          linewidth=0.2, cmap=plt.cm.viridis, arrowsize=0.8)
        total_B = B[0] + B[1]
        ax.streamplot(xx_grid, zz_grid, total_B[0], total_B[2], density=density*2, color='k',
                      linewidth=0.5, cmap=plt.cm.viridis, arrowsize=0.8)
        ax.set_aspect('equal')
        # ax.set_title('')
    else:
        assert False
    plt.show()
    #plt.savefig('tmp.pdf')

def mark_coils(ax, pp): # side view
    for k in range(NL):
        xx = pp.xx(k)
        rr = pp.rr(k)
        for x, r in zip(xx, rr):
            ax.add_patch(Circle((x - r, ZZ(k)), 0.01, color=colors(k), alpha=0.7))
            ax.add_patch(Circle((x + r, ZZ(k)), 0.01, color=colors(k), alpha=0.7))
        xin, xout = xx[0], xx[-1]
        rin, rout = rr[0], rr[-1]
        for x in [xin - rin, xin + rin, xout - rout, xout + rout]:
            ax.add_artist(lines.Line2D([x,  x], [ZZ(k), -6], linewidth=0.2, color=colors(k)))

def plot_spirals(ax, pp, yoff): # top view
    for k in range(NL):
        for x, r in zip(pp.xx(k), pp.rr(k)):
            ax.add_patch(Circle((x, yoff), r, facecolor='none', edgecolor=colors(k), alpha=0.7))

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
    tmp_single_check = False
    if tmp_single_check:
        single_k = 1
        Bsingle = single_B(xx_grid, 0, zz_grid, pp, single_k)
    else:
        B = total_B(xx_grid, 0, zz_grid, pp)
    # print(B)
    if 1:
        null_x, null_y, null_z = null_p
        Bxx, Bxz, Bzx, Bzz = total_B_derivs(null_x, null_y, null_z, pp)
        evec = compute_jac_eigenvector(Bxx, Bxz, Bzx, Bzz)
        deg = mark_x(ax, evec, null_p)
        grads = grads_B(Bxx, Bxz, Bzx, Bzz)
    if tmp_single_check:
        ax.streamplot(xx_grid, zz_grid, Bsingle[0], Bsingle[2], density=2, color=colors(single_k),
                      linewidth=0.5, cmap=plt.cm.viridis, arrowsize=0.8)
    else:
        ax.streamplot(xx_grid, zz_grid, B[0], B[2], density=2, color='g',
                      linewidth=0.5, cmap=plt.cm.viridis, arrowsize=0.8)
    ax.set_aspect('equal')
    ax.set_title(f'{title}: ang={round(deg, 2)} grads={round(grads[0],2)},{round(grads[1],2)}', fontsize=6)

def plot_field_and_spirals(pdf, tau, gamma, seed, pp, null_p, U, B_at_null, angle, grad):
    yoffset = -6
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.hlines(y=0, xmin=-5, xmax=5, colors='black', linestyles='--', lw=1)
    ax.hlines(y=-1, xmin=-5, xmax=5, colors='black', linestyles='-', lw=1)
    ax.hlines(y=yoffset, xmin=-5, xmax=5, colors='black', linestyles='--', lw=1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-11, 3)
    ax.set_aspect('equal')
    if tau is None:
        title = f'seed={seed}: {pp}'
    else:
        title = f'tau={round(tau,3)} gamma={round(gamma,3)} seed={seed}: {pp}'
    plot_field(ax, pp, null_p, title)
    mark_coils(ax, pp)
    plot_spirals(ax, pp, yoffset)
    r1 = pp.rr(0)[-1]
    r2 = pp.rr(1)[-1]
    plt.text(-4.5, -0.8, f'r1={round(r1,5)}, r2={round(r2, 5)} U={round(U,5)}, B_at_null={round(B_at_null, 5)}, angle={round(angle, 5)}, grad={round(grad, 5)}',
             fontsize=8)
    print(f'# {{{pdf}}}')
    plt.savefig(pdf)

def do_plots(null_p, maxU, fnames):
    if 0: spiral_demo()
    if 0: ex_spiral_demo()
    for fname in fnames:
        with open(fname) as file:
            format = None
            tau = None
            gamma = None
            for line in file:
                seed,tau,gamma,a1,a2,b1,b2,c1,c2,dx1,dx2,Ir,n1,n2,U,B_at_null,angle,grad = line.rstrip().split(',')
                if seed == 'seed':
                    continue # header
                if float(U) > maxU:
                    continue
                pp = Spirals()
                pp.set(a1,a2,b1,b2,c1,c2,dx1,dx2,Ir)
                #print(f'XXX {a1}, {a2}, {b1}, {b2}, {c1}, {c2}, {dx1}, {dx2}, {Ir}')
                #print(f'XXX {pp}')
                pdf = fname.replace('.csv', '')
                pdf = f'{pdf}.{seed}.pdf'
                plot_field_and_spirals(pdf, float(tau), float(gamma), seed, pp, null_p,
                                       float(U), float(B_at_null), float(angle), float(grad))

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
        assert len(sys.argv) > 3
        maxU = float(noun)
        fnames = sys.argv[3:]
        do_plots(null_p, maxU, fnames)
    elif verb == 'run':
        vars = ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'Ir'] # no dx
        dim = len(vars) # how many parameters (out ot 9) to play with at a time
        key = datetime.now().strftime("%Y%m%d.%H%M%S") # when started
        do_runs(noun, null_p, vars, dim, key)
    elif verb == 'join':
        assert len(sys.argv) > 3
        do_join(sys.argv[2:])
    elif verb == 'demo':
        do_demo(noun)
    else:
        assert False
