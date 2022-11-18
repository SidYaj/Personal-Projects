import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def dudt(r1, r2, m, k, l, o):
    return (r1 * o ** 2) - k / m * (r1 + r2 - l)


def drdt(u):
    return u


def solve_ODE_RK4(omega, k, m1, m2, l, T, dt):
    n = int(T/dt)
    t = np.linspace(0, T, n+1)
    r1 = np.empty(n + 1)
    r2 = np.empty(n + 1)
    u1 = np.empty(n + 1)
    u2 = np.empty(n + 1)
    theta = np.empty(n + 1)
    r1[0] = 2 * m2 * 1 / (m1 + m2)
    u1[0] = 0
    r2[0] = 2 * m1 * 1 / (m1 + m2)
    u2[0] = 0
    for i in range(n):
        o = m1 * m2 / (m1 + m2) * ((r1[0] + r2[0]) ** 2) / (m1 * (r1[i] ** 2) + m2 * (r2[i] ** 2)) * omega
        u1k1 = dudt(r1[i], r2[i], m1, k, l, o) * dt
        u2k1 = dudt(r2[i], r1[i], m2, k, l, o) * dt
        r1k1 = drdt(u1[i]) * dt
        r2k1 = drdt(u2[i]) * dt
        u1k2 = dudt(r1[i] + r1k1 * 0.5, r2[i] + r2k1 * 0.5, m1, k, l, o) * dt
        u2k2 = dudt(r2[i] + r2k1 * 0.5, r1[i] + r1k1 * 0.5, m2, k, l, o) * dt
        r1k2 = drdt(u1[i] + u1k1 * 0.5) * dt
        r2k2 = drdt(u2[i] + u2k1 * 0.5) * dt
        u1k3 = dudt(r1[i] + r1k2 * 0.5, r2[i] + r2k2 * 0.5, m1, k, l, o) * dt
        u2k3 = dudt(r2[i] + r2k2 * 0.5, r1[i] + r1k2 * 0.5, m2, k, l, o) * dt
        r1k3 = drdt(u1[i] + u1k2 * 0.5) * dt
        r2k3 = drdt(u2[i] + u2k2 * 0.5) * dt
        u1k4 = dudt(r1[i] + r1k3, r2[i] + r2k3, m1, k, l, o) * dt
        u2k4 = dudt(r2[i] + r2k3, r1[i] + r1k3, m2, k, l, o) * dt
        r1k4 = drdt(u1[i] + u1k3) * dt
        r2k4 = drdt(u2[i] + u2k3) * dt
        u1[i + 1] = u1[i] + 1.0 / 6.0 * (u1k1 + 2 * u1k2 + 2 * u1k3 + u1k4)
        u2[i + 1] = u2[i] + 1.0 / 6.0 * (u2k1 + 2 * u2k2 + 2 * u2k3 + u2k4)
        r1[i + 1] = r1[i] + 1.0 / 6.0 * (r1k1 + 2 * r1k2 + 2 * r1k3 + r1k4)
        r2[i + 1] = r2[i] + 1.0 / 6.0 * (r2k1 + 2 * r2k2 + 2 * r2k3 + r2k4)
        theta[i + 1] = theta[i] + o * dt
        if theta[i + 1] >= 2 * np.pi:
            theta[i + 1] -= 2 * np.pi
    return r1, r2, theta


def solve_COM(v, phi, T, dt, g=9.8):
    num_steps = int(T /dt)
    x = np.empty(num_steps+1)
    y = np.empty(num_steps+1)
    vy = np.empty(num_steps+1)
    x[0] = 0
    y[0] = 0
    vy[0] = v*np.sin(phi)
    for i in range(num_steps):
        x[i+1] = x[i] + v*np.cos(phi)*dt
        vyh = vy[i] - g*dt/2
        vy[i+1] = vy[i] - g*dt
        y[i+1] = y[i] + vyh*dt
    return x, y


omega = 2 * np.pi
k = 5
m1 = 1
m2 = 1
l = 0
dt = 1e-4
v = 40
phi = np.pi/3
T = int(2*v*np.sin(phi)/9.8)
r1, r2, theta1 = solve_ODE_RK4(omega, k, m1, m2, l, T, dt)
x, y = solve_COM(v, phi, T, dt)
theta2 = theta1 + np.pi
x1 = x + r1 * np.cos(theta1)
y1 = y + r1 * np.sin(theta1)
x2 = x + r2 * np.cos(theta2)
y2 = y + r2 * np.sin(theta2)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=[-10, 200], ylim=[0, 100])
ax.set_aspect('equal')
ax.grid()


line1, = ax.plot([], [], 'o-', lw=1)
line2, = ax.plot([], [], '-', lw=1, label="Mass 1")
line3, = ax.plot([], [], '-', lw=1, label="Mass 2")
line4, = ax.plot([], [], '--', lw=1, label="Centre of mass")
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, ' ', transform=ax.transAxes)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4


def animate(i):
    line1.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    line2.set_data(x1[:i], y1[:i])
    line3.set_data(x2[:i], y2[:i])
    line4.set_data(x[:i], y[:i])
    time_text.set_text(time_template % (i*dt))
    return line1, line2, line3, line4, time_text


ani = anim.FuncAnimation(fig, animate, range(0, len(x1), 100), interval=10, blit=True, init_func=init)
ax.legend()
plt.show()