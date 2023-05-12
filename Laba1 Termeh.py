import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
t = sp.Symbol('t')

x = (1+1.5 * sp.sin(12*t))
y = (1.25 * t + 0.2 * sp.cos(12*t))

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
F_x = sp.lambdify(t, x)
F_y = sp.lambdify(t, y)
F_Vx = sp.lambdify(t, Vx)
F_Vy = sp.lambdify(t, Vy)
F_Ax = sp.lambdify(t, Ax)
F_Ay = sp.lambdify(t, Ay)
T = np.linspace(0, 10, 1001)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
for i in np.arange(len(T)):
 X[i] = F_x(T[i]) 
 Y[i] = F_y(T[i]) 
 VX[i] = F_Vx(T[i])
 VY[i] = F_Vy(T[i])
 AX[i] = F_Ax(T[i])
 AY[i] = F_Ay(T[i])

Phi = np.arctan2(VY, VX)
fig = plt.figure(figsize=[13,8])
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-2, 2], ylim=[-2,2])
ax.grid(True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Движение точки')

ax.plot(X, Y)
P = ax.plot(X[0], Y[0], marker='o', label='Точка')[0]
V_Line = ax.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], color=[0, 0, 0], label='Скорость')[0]
def Rot2D(X, Y, Phi):
 RotX = X * np.cos(Phi) - Y * np.sin(Phi)
 RotY = X * np.sin(Phi) + Y * np.cos(Phi)
 return RotX, RotY 
XArrow = np.array([-0.15, 0, -0.15])
YArrow = np.array([0.1, 0, -0.1])
RArrowX, RarrowY = Rot2D(XArrow, YArrow, Phi[0])
V_Arrow = ax.plot(X[0]+VX[0]+RArrowX, Y[0]+VY[0]+RarrowY, color=[0, 0, 0])[0]

A_Line = ax.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], color=[1, 0, 0], label='Ускорение')[0]
XArrow_A = np.array([-0.1, 0, -0.1])
YArrow_A = np.array([0.15, 0, -0.15])
RArrowX_A, RarrowY_A = Rot2D(XArrow_A, YArrow_A, Phi[0])
A_Arrow = ax.plot(X[0]+AX[0]+RArrowX_A, Y[0]+AY[0]+RarrowY_A, color=[1, 0, 0])[0]

ax.legend()

def MagicOfTheMovement(i):
 P.set_data(X[i], Y[i])
 V_Line.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
 RArrowX, RarrowY = Rot2D(XArrow, YArrow, Phi[i])
 V_Arrow.set_data(X[i]+VX[i]+RArrowX, Y[i]+VY[i]+RarrowY)
 A_Line.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
 RArrowX_A, RarrowY_A = Rot2D(XArrow_A, YArrow_A, Phi[i])
 A_Arrow.set_data(X[i]+AX[i]+RArrowX_A, Y[i]+AY[i]+RarrowY_A)
 return [P, V_Line, V_Arrow, A_Line, A_Arrow]

animation = FuncAnimation(fig, MagicOfTheMovement, interval=20, frames=len(T))
plt.show()