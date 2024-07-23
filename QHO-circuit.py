# Q4: QHO. cetin ilhan kaya

from qutip import *
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib as mpl
import numpy as np
from math import *

#in natural units, hbar = 1, m=1
# for simplicity, choose freq w=10^10Hz
hbar=1.0
w = 2*pi
m = 1.0
N = 100
a = destroy(N) #annihilation op.

# Hamiltonian
H = hbar * w * (a.dag()*a + 0.5)

# # using a and a.dag(), x, n and p can be constructed as

x = sqrt(hbar/(2*m*w) ) * ( a + a.dag() )
p = 1j * sqrt(hbar*m*w/2) * (a.dag() - a)
n = a.dag()*a

#initially coherent state
coh = coherent(N,alpha = 5.0)

# list for solver should store the state vector
tlist = np.linspace(-10,10,N)
result= mesolve(H, coh, tlist, [], [])

#expectation values
exval_x, exval_p, exval_n, exval_x2, exval_p2, exval_n2 = [],[], [],[],[],[]
for i in range(N):
    exval_x.append(expect(x,result.states[i]))
    exval_p.append(expect(p,result.states[i]))
    exval_n.append(expect(n,result.states[i]))
    exval_x2.append(expect(x**2,result.states[i]))
    exval_p2.append(expect(p**2,result.states[i]))
    exval_n2.append(expect(n**2,result.states[i]))
    
    
stdx=variance(x, result.states)
stdp=variance(p, result.states)
stdn=variance(n, result.states)

fig, axis = plt.subplots(figsize=(15.0,8.0),nrows=3, ncols=1)
axis[0].plot(tlist, exval_x, color="k",label="<x>")
axis[0].fill_between(tlist,exval_x-stdx, exval_x+stdx, color="r")
axis[0].set_xlabel("time")
axis[0].set_title("Expectation value of x")
axis[0].grid()
axis[1].plot(tlist,exval_p, color="k",label="<p>")
axis[1].fill_between(tlist,exval_p-stdp, exval_p+stdp, color="r")
axis[1].set_xlabel("time")
axis[1].set_title("Expectation value of p")
axis[1].grid()
axis[2].plot(tlist,exval_n, color="k",label="<p>")
axis[2].fill_between(tlist,exval_n-stdn, exval_n+stdn, color="r")
axis[2].set_xlabel("time")
axis[2].set_title("Expectation value of n")
axis[2].grid()
plt.show()

cohdm = ket2dm(coh)
fig2, axis2 = plt.subplots(figsize=(15.0,8.0),nrows=1, ncols=2)

#function to call at each frame, func(frame)
def func(i):
    axis2[0].cla() #clear previous frame
    axis2[1].cla()
    W_coh = wigner(cohdm,tlist,result.states[i])
    wcoh, ylist = W_coh if type(W_coh) is tuple else (W_coh,tlist)
    fock = fock_dm(N,2)
    W_fock = wigner(fock,tlist,result.states[i])
    wmap = wigner_cmap(wcoh)
    wlim = abs(wcoh).max()
    norm = mpl.colors.Normalize(-wlim,wlim)
    axis2[0].contourf(tlist,ylist,wcoh,100,cmap=wmap, norm=norm)
    wfock = np.real(W_fock[i,:])
    axis2[1].bar(np.arange(0,N),wfock)
    axis2[0].set_title("Wigner quasiprobability distribution")
    axis2[1].set_title("Fock distribution")
    axis2[1].set_ylim(-.2,.4,auto=False)
    plt.show()

anime = anm.FuncAnimation(fig2,func,frames=N, interval=5000)
writervideo = anm.FFMpegWriter(fps=10)
anime.save("coherent.mp4", writer=writervideo)
plt.close(fig2)

# Squeezed coherent state => Squeeze * Displacement * |0>
s = displace(N,5.0)*basis(N,0)
squeezed = squeeze(N,1.0)*s
result2 = mesolve(H,squeezed,tlist,[],[])
exval_x.clear(), exval_p.clear(), exval_n.clear(), exval_x2.clear(), exval_p2.clear(), exval_n2.clear()

for i in range(N):
    exval_x.append(expect(x,result2.states[i]))
    exval_p.append(expect(p,result2.states[i]))
    exval_n.append(expect(n,result2.states[i]))
    exval_x2.append(expect(x**2,result2.states[i]))
    exval_p2.append(expect(p**2,result2.states[i]))
    exval_n2.append(expect(n**2,result2.states[i]))
    
    
stdx=variance(x, result2.states)
stdp=variance(p, result2.states)
stdn=variance(n, result2.states)

fig3, axis3 = plt.subplots(figsize=(15.0,8.0),nrows=3, ncols=1)
axis3[0].plot(tlist, exval_x, color="k",label="<x>")
axis3[0].fill_between(tlist,exval_x-stdx, exval_x+stdx, color="r")
axis3[0].set_xlabel("time")
axis3[0].set_title("Expectation value of x")
axis3[0].grid()
axis3[1].plot(tlist,exval_p, color="k",label="<p>")
axis3[1].fill_between(tlist,exval_p-stdp, exval_p+stdp, color="r")
axis3[1].set_xlabel("time")
axis3[1].set_title("Expectation value of p")
axis3[1].grid()
axis3[2].plot(tlist,exval_n, color="k",label="<p>")
axis3[2].fill_between(tlist,exval_n-stdn, exval_n+stdn, color="r")
axis3[2].set_xlabel("time")
axis3[2].set_title("Expectation value of n")
axis3[2].grid()
plt.show()

sqdm = ket2dm(squeezed)
fig4, axis4 = plt.subplots(figsize=(15.0,8.0),nrows=1, ncols=2)

#function to call at each frame, func(frame)
def func_sq(i):
    axis4[0].cla() #clear previous frame
    axis4[1].cla()
    W_sq = wigner(sqdm,tlist,result2.states[i])
    wsq, ylist = W_sq if type(W_sq) is tuple else (W_sq,tlist)
    fock = fock_dm(N,2)
    W_fock = wigner(fock,tlist,result2.states[i])
    wmap = wigner_cmap(W_sq)
    wlim = abs(wsq).max()
    norm = mpl.colors.Normalize(-wlim,wlim)
    axis4[0].contourf(tlist,ylist,wsq,100,cmap=wmap, norm=norm)
    wfock = np.real(W_fock[i,:])
    axis4[1].bar(np.arange(0,N),wfock)
    axis4[0].set_title("Wigner quasiprobability distribution")
    axis4[1].set_title("Fock distribution")
    axis4[1].set_ylim(-.2,.4,auto=False)
    plt.show()
    
anime2 = anm.FuncAnimation(fig4,func_sq,frames=N, interval=5000)
anime2.save("squeezed.mp4", writer=writervideo)
plt.close(fig4)

