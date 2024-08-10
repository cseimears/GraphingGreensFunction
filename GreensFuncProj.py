import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the differential equations
def dy1dt(y1, t):
    y2 = y1[1]
    y1dot = [y2, 2 - 2*y2 - 2*y1[0]]
    return y1dot

def dy2dt(y1, t):
    y2 = y1[1]
    y2dot = [y1[1], 4 - y1[0] - y1[1]**2]
    return y2dot

# Define the time points to solve the ODEs at
t1 = np.linspace(0, 10, 1000)
t2 = np.linspace(0, 5, 1000)

# Solve the homogeneous equations
y1_homo = odeint(dy1dt, [0, 0], t1)
y2_homo = odeint(dy2dt, [0, 0], t2)

# Plot the homogeneous solutions
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(t1, y1_homo[:, 0], label="y(t)")
ax[0].plot(t1, y1_homo[:, 1], label="y'(t)")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t), y'(t)")
ax[0].set_title("Homogeneous Solution for y'' + 2y' + 2 = 2x")
ax[0].legend()

ax[1].plot(t2, y2_homo[:, 0], label="y(t)")
ax[1].plot(t2, y2_homo[:, 1], label="y'(t)")
ax[1].set_xlabel("t")
ax[1].set_ylabel("y(t), y'(t)")
ax[1].set_title("Homogeneous Solution for y'' + y = 4")
ax[1].legend()

# Define the Green's function for each ODE
def green1(t, s, f):
    mask = (s <= t)
    result = np.zeros_like(t)
    result[mask] = f(s[mask]) * (np.exp(-(t-s[mask])) - np.exp(-2*(t-s[mask])))
    result[~mask] = f(s[~mask]) * (np.exp(-(s[~mask]-t)) - np.exp(-2*(s[~mask]-t)))
    return result

def green2(t, s):
    return np.sin(t-s) - np.sin(s-t)

# Define the particular solution function using the Green's function
s = np.linspace(0,10,1000)

def particular_solution(t, y0, f, green):
    result = np.zeros_like(t)
    for i in range(len(t)):
        integrand = lambda tau: np.squeeze(f(tau) * green(t[i], tau))
        if t[i] == t[0]:
            result[i] = np.trapz([integrand(t[0])], [t[0]])
        else:
            result[i] = np.trapz([integrand(tau) for tau in t[0:i+1]], t[0:i+1]) + y0*np.exp(-t[i])
    return result


    
# Solve for the particular solution using the Green's function
y1_p = particular_solution(t1, 0, lambda t: 2*t, lambda t, s: green1(t, s, lambda s: 2*s))
y2_p = particular_solution(t2, 0, lambda t: 4, lambda t, s: green2(t, s))

# Plot the particular solutions
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(t1, y1_homo[:, 0] + y1_p, label="y(t)")
ax[0].plot(t1, y1_homo[:, 1] + 2*y1_p, label="y'(t)")
ax[0].set_xlabel("t")
ax[0].set_ylabel("y(t), y'(t)")
ax[0].set_title("Particular Solution for y'' + 2y' + 2 = 2x")
ax[0].legend()

ax[1].plot(t2, y2_homo[:, 0] + y2_p, label="y(t)")
ax[1].plot(t2, y2_homo[:, 1] + y2_p, label="y'(t)")
ax[1].set_xlabel("t")
ax[1].set_ylabel("y(t), y'(t)")
ax[1].set_title("Particular Solution for y'' + y = 4")
ax[1].legend()

plt.show()
