import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def ot(il, r, omega):
    return il ** 2 / r ** 2 * omega


def dudt(r, mu, k, nl, o):
    return r * o ** 2 - k/mu * (r - nl)


def drdt(u):
    return u


def solve_ODE(omega, k, mu, nl, il, T, dt):
    n = int(T/dt)
    t = np.linspace(0, T, n+1)
    r = np.empty(n + 1)
    u = np.empty(n + 1)
    theta = np.empty(n + 1)
    o = np.empty(n + 1)
    KE = np.empty(n + 1)
    PE = np.empty(n + 1)
    r[0] = il
    u[0] = 0
    theta[0] = 0
    o[0] = omega
    for i in range(n):
        o1 = ot(il, r[i], omega) * dt
        u1 = dudt(r[i], mu, k, nl, o[i]) * dt
        r1 = drdt(u[i]) * dt
        o2 = ot(il, r[i] + r1 * 0.5, omega) * dt
        u2 = dudt(r[i] + r1 * 0.5, mu, k, nl, o[i] + o1 * 0.5) * dt
        r2 = drdt(u[i] + u1 * 0.5) * dt
        o3 = ot(il, r[i] + r2 * 0.5, omega) * dt
        u3 = dudt(r[i] + r2 * 0.5, mu, k, nl, o[i] + o2 * 0.5) * dt
        r3 = drdt(u[i] + u2 * 0.5) * dt
        o4 = ot(il, r[i] + r3, omega) * dt
        u4 = dudt(r[i] + r3, mu, k, nl, o[i] + o3) * dt
        r4 = drdt(u[i] + u3) * dt
        u[i + 1] = u[i] + 1.0/6.0 * (u1 + 2 * u2 + 2 * u3 + u4)
        r[i + 1] = r[i] + 1.0/6.0 * (r1 + 2 * r2 + 2 * r3 + r4)
        theta[i + 1] = theta[i] + 1.0/6.0 * (o1 + 2 * o2 + 2 * o3 + o4)
        o[i + 1] = 1.0/6.0 * (o1 + 2 * o2 + 2 * o3 + o4) / dt
    return r, theta


v = 40
phi = np.pi/3
omega = 2*np.pi
k = 5.0
m1 = 1.0
m2 = 5.0
nl = 0
il = 5.0
dt = 1e-4
T=10
mu = m1*m2/(m1+m2)
r, theta1 = solve_ODE(omega, k, mu, nl, il, T, dt) 
theta2 = theta1 + np.pi
r1 = mu/m1 * r
r2 = mu/m2 * r
x1 = r1 * np.cos(theta1)
y1 = r1 * np.sin(theta1)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=[-20, 20], ylim=[-20, 20])
ax.set_aspect('equal')
ax.grid()
line1, = ax.plot([], [], 'o-', lw=1)
line2, = ax.plot([], [], 'o-', lw=1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9,' ', transform=ax.transAxes)


def init():
    line1.set_data([], [])
    return line1, time_text


def animate(i):
    line1.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    time_text.set_text(time_template % (i * dt))
    return line1, time_text


ani = anim.FuncAnimation(fig, animate, range(0, len(x1), 100),interval=10, blit=True, init_func=init)
plt.show()