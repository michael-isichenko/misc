$\newcommand{\bfL}{{\bf L}}$
$\newcommand{\bfA}{{\bf A}}$
$\newcommand{\bfb}{{\bf b}}$
$\newcommand{\bfB}{{\bf B}}$
$\newcommand{\bfx}{{\bf x}}$
$\newcommand{\bfv}{{\bf v}}$
$\newcommand{\bfna}{{\bf\nabla}}$

## Magnetic field, its derivatives, and the null point

### Biot-Savart field of a circular loop

Biot-Savart field equals
$$
\bfB(\bfx)=\oint\bfb(\bfx,t)\,dt,\quad
\bfb(\bfx,t) = \frac{\dot\bfL\times(\bfx-\bfL)}{D^3},\quad D\equiv|\bfx-\bfL|.
$$
Here the current loop is defined parametrically by $\bfL(t)=(a\cos(t),a\sin(t),0)$.

Component-wise, we have
$$
\begin{aligned}
b_x &= \frac{\dot L_y(z-L_z) - \dot L_z(x-L_x)}{D^3} = \frac{\dot L_yz}{D^3},\\
b_y &= \frac{\dot L_z(x-L_x) - \dot L_x(z-L_z)}{D^3} = -\frac{\dot L_xz}{D^3},\\
b_z &= \frac{\dot L_x(y-L_y) - \dot L_y(x-L_x)}{D^3}
= \frac{L_x\dot L_y-L_y\dot L_x + \dot L_xy - \dot L_y x}{D^3}
= \frac{a^2 + \dot L_xy - \dot L_y x}{D^3}.
\end{aligned}
$$
Here the condition $L_z\equiv0$ is applied for a flat $(x,y)$ loop.  In terms of the parameter $t$,
$$
\begin{aligned}
b_x &= \frac{az\cos(t)}{D^3},\\
b_y &= \frac{az\sin(t)}{D^3},\\
b_z &= a\frac{a - x\cos(t) - y\sin(t)}{D^3},\\
D   &= \sqrt{(x - a\cos(t))^2 + (y - a\sin(t))^2 + z^2}
\end{aligned}
$$
Note that, for $y=0$, $D$ is an even function of $t$, and $B_y=\int_{-\pi}^\pi b_y\,dt=0$.

Derivatives of the infinitesimal field:
$$
\begin{aligned}
b_{x,x} &= -3az\frac{\cos(t)(x-a\cos(t))}{D^5},\\
b_{y,y} &= -3az\frac{\sin(t)(y-a\sin(t))}{D^5},\\
b_{z,z} &=  3az\frac{x\cos(t) + y\sin(t) - a}{D^5},\\
b_{x,z} &=  \frac{a\cos(t)}{D^3}\left(1-3\frac{z^2}{D^2}\right),\\
b_{z,x} &= -\frac{a\cos(t)}{D^3} -3a\frac{(x-a\cos(t))(a-x\cos(t)-y\sin(t))}{D^5}.
\end{aligned}
$$
It is immediately verified that $\nabla\cdot\bfb=b_{x,x}+b_{y,y}+b_{z,z}=0$, and therefore the total field $\bfB$ is divergence free, as it should be.  Not so for the infinitesimal $b_{x,z}$ and $b_{z,x}$.  The current-free condition $B_{x,z}=B_{z,x}$ is valid only for the total field after $\oint dt$ over $[0,2\pi]$.  Proving this in general by integrating the derivatives above by $t$ is very difficult, as elliptic integrals are involved.  *Mathematica* consistenly evaluates $B_{x,z}-B_{x,z}$ at various random values of $(a,x,y,z)$ as exact zero, which cannot be a coincidence.
