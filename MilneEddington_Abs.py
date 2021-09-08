# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt, pi, tau
from matplotlib.animation import FuncAnimation
from numpy import exp, sin, cosh, abs
from matplotlib.ticker import ScalarFormatter
from scipy import integrate
import os
import csv

plt.rcParams["font.size"] = 18

####### parameter ############
# 0:plt 1:save
SAVE = 1

Save_fp = "./figs/MilneEddington/Abs"
SaveIter_fp = "./figs/MilneEddington/Abs/Iteration"

for fp in [Save_fp, SaveIter_fp]:
  os.makedirs(fp,exist_ok=True)

##############################


# グラフの体裁を整える
def set_ax(ax):
  ax.grid(linestyle="dotted")
  ax.xaxis.set_tick_params(direction='in')
  ax.yaxis.set_tick_params(direction='in')
  ax.tick_params(labelsize=21)
  # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  # ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
  ax.yaxis.offsetText.set_fontsize(20)
  # ax.set_ylim(-10,10)

def Intensity_Abs_(tau,mu,b,eps):
  alpha = sqrt(3.0*eps)
  A = (1.0-2.0/3.0*b)/(1.0+2.0/3.0*alpha)
  B = (1-eps)/(1.0+mu*alpha)*A
  if(mu > -1e-5): return 1.0+b*(tau+mu)-B*exp(-alpha*tau)
  else: return 1.0+b*(tau+mu)-B*exp(-alpha*tau)-(1.0+mu*b-B)*exp(tau/mu)

def Intensity_Abs(tau,mu,b,eps):
  # mu が iterable かどうかで場合分け
  if(not hasattr(mu, "__iter__")): return Intensity_Abs_(tau,mu,b,eps)
  else: return np.array([Intensity_Abs_(tau,mu_) for mu_ in mu])

def B(tau, b): return 1.0*b*tau
def J_Ana(tau,b,eps):
  return 1.0+b*tau-(1.0-2.0/3.0*b)/(1.0+2.0/3.0*sqrt(3.0*eps))*exp(-sqrt(3.0*eps)*tau)
def H_Ana(tau,b,eps):
  return b/3.0+sqrt(3.0*eps)/3.0*(1.0-2.0/3.0*b)/(1.0+2.0/3.0*sqrt(3.0*eps))*exp(-sqrt(3.0*eps)*tau)
def K_Ana(tau,b,eps):
  return J_Ana(tau,b,eps)/3.0
def S_Ana(tau,b,eps):
  return 1.0+b*tau-(1.0-eps)*(1.0-2.0/3.0*b)/(1.0+2.0/3.0*sqrt(3.0*eps))*exp(-sqrt(3.0*eps)*tau)

def JfromI_Ana_(tau,b,eps):
  # 被積分関数fを定義
  f = lambda mu : 0.5*Intensity_Abs(tau,mu,b,eps)
  return integrate.quad(f,-1.0,1.0)[0]
def JfromI_Ana(tau,b,eps):
  if(not hasattr(tau, "__iter__")): return JfromI_Ana_(tau,b,eps)
  else: return np.array([JfromI_Ana_(tau_,b,eps) for tau_ in tau])

def HfromI_Ana_(tau,b,eps):
  # 被積分関数fを定義
  f = lambda mu : 0.5*mu*Intensity_Abs(tau,mu,b,eps)
  return integrate.quad(f,-1.0,1.0)[0]
def HfromI_Ana(tau,b,eps):
  if(not hasattr(tau, "__iter__")): return HfromI_Ana_(tau,b,eps)
  else: return np.array([HfromI_Ana_(tau_,b,eps) for tau_ in tau])

def KfromI_Ana_(tau,b,eps):
  # 被積分関数fを定義
  f = lambda mu : 0.5*mu*mu*Intensity_Abs(tau,mu,b,eps)
  return integrate.quad(f,-1.0,1.0)[0]
def KfromI_Ana(tau,b,eps):
  if(not hasattr(tau, "__iter__")): return KfromI_Ana_(tau,b,eps)
  else: return np.array([KfromI_Ana_(tau_,b,eps) for tau_ in tau])

def JfromI_Num(I, mu_array, tau_array):
  return np.array([integrate.simps(0.5*I[:,itau],mu_array) for itau in range(len(tau_array))])

def HfromI_Num(I, mu_array, tau_array):
  return np.array([integrate.simps(0.5*I[:,itau]*mu_array,mu_array) for itau in range(len(tau_array))])

def KfromI_Num(I, mu_array, tau_array):
  return np.array([integrate.simps(0.5*I[:,itau]*mu_array*mu_array,mu_array) for itau in range(len(tau_array))])


def plot_Intensity():
  fig1 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Intensity")
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"$I/H_0$")
  tau = np.linspace(0,5,100)
  Ip_TSA = lambda tau,mubar : (tau+2.0*mubar)/(mubar*mubar)
  Im_TSA = lambda tau,mubar : (tau)/(mubar*mubar)
  for mu in [1.0, 0.7, 0.3, 0.0, -0.3, -0.7, -1.0]:
    ax1.plot(tau,Intensity_Abs(tau, mu), label=r"$\mu$=%.1f"%(mu))
  mubar = 1.0/sqrt(3)
  ax1.plot(tau,Ip_TSA(tau, mubar), "--", color="red", label=r"$I^{+}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  ax1.plot(tau,Im_TSA(tau, mubar), "-.", color="red", label=r"$I^{-}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  mubar = 1.0
  ax1.plot(tau,Ip_TSA(tau, mubar), "--", color="blue", label=r"$I^{+}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  ax1.plot(tau,Im_TSA(tau, mubar), "-.", color="blue", label=r"$I^{-}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  ax1.legend(fontsize=12)
  if(SAVE == 0): plt.show()
  else: fig1.savefig(Save_fp + "/Intensity_Abs.png")

def plot_Moment(b):
  fig1 = plt.figure(figsize=(8, 8))
  fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Moment (b/B(0) = %.2f)"%b)
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"(J,H,S)/B(0)")

  tau = np.linspace(0,2,100)

  cols = ["red", "blue", "green"]
  for i,eps in enumerate([1.0,0.5,0.1]):
    ax1.plot(tau,S_Ana(tau, b, eps),"-",color=cols[i])
    ax1.plot(tau,J_Ana(tau, b, eps),"--",color=cols[i])
    ax1.plot(tau,H_Ana(tau, b, eps),"-.",color=cols[i])
    # ax1.plot(tau,JfromI_Ana(tau),"-",color="red",label="J (from I)")
    # ax1.plot(tau,HfromI_Ana(tau),"-",color="blue",label="H (from I)")
    # ax1.plot(tau,KfromI_Ana(tau),"-",color="green",label="K (from I)")
    
  ax1.plot([],[],"-",color="black",label="S")
  ax1.plot([],[],"--",color="black",label="J")
  ax1.plot([],[],"-.",color="black",label="H")
  for i,eps in enumerate([1.0,0.5,0.1]):
    ax1.plot([],[],color=cols[i],label=r"$\varepsilon$=%.1f"%eps)
  

  ax1.legend()

  if(SAVE == 0): plt.show()
  else: fig1.savefig(Save_fp + "/MomentAbs_b=%.2f.png"%(b))

def plot_EddingtonTensor():
  fig1 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Eddington tensor f = K/J")
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"K/J")
  # ax1.set_ylim(ymin=0.0,ymax=0.5)
  tau = np.linspace(0,5,100)
  ax1.plot(tau,[1.0/3.0]*len(tau),"-",color="black",label="1/3")
  
  cols = ["blue","green","red"]
  lsts = ["--","-.","-"]
  eps_list = [0.02,0.5,1.0]
  b_list = [0.0,1.5,5.0]
  for j,eps in enumerate(eps_list):
    ax1.axvline(x=1.0/sqrt(3.0*eps),linestyle=lsts[j],color="black")
  
  for i,b in enumerate(b_list):
    for j,eps in enumerate(eps_list):
      ax1.plot(tau,KfromI_Ana(tau,b,eps)/JfromI_Ana(tau,b,eps),lsts[j],color=cols[i])
      # ax1.plot(tau,KfromI_Ana(tau,b,eps)/JfromI_Ana(tau,b,eps),lsts[j],color=cols[i],label="b=%.1f, $\epsilon$=%.1f"%(b,eps))
  
  for i,b in enumerate(b_list):
    ax1.plot([],[],"-",color=cols[i],label="b=%.1f"%(b))
  for j,eps in enumerate(eps_list):
    ax1.plot([],[],lsts[j],color="black",label="$\epsilon$=%.2f"%(eps))
  ax1.legend(fontsize=12)

  

  if(SAVE == 0): plt.show()
  else: fig1.savefig(Save_fp + "/EddingtonTensor.png")

def plot_Temperature():
  fig1 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Temperature")
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"$T/T_{\mathrm{eff}}$")
  # ax1.set_ylim(ymin=0.0,ymax=2)
  tau = np.linspace(0,5,100)
  T_TSA = lambda tau,mubar : ((tau+mubar)/(4.0*mubar*mubar))**(1.0/4.0)
  T_ME = lambda tau : ((tau+2.0/3.0)*3.0/4.0)**(1.0/4.0)

  ax1.axhline(y=1.0,color="black",linestyle=":")
  ax1.plot(tau,T_ME(tau), "-", color="black", label=r"$T_\mathrm{MilneEddington}$")
  for mubar in [0.2,0.4,1.0/sqrt(3.0),1.0]:
    ax1.plot(tau,T_TSA(tau, mubar), "--", label=r"$T_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  ax1.legend(fontsize=12)
  if(SAVE == 0): plt.show()
  else: fig1.savefig(Save_fp + "/Temperature_NoAbs.png")

def init_I(mu_array, tau_array):
  I = []
  for mu in mu_array:
    data = []
    for tau in tau_array:
      tmp = 3.0*(tau+mu+2.0/3.0)
      if(tau < 0.0): tmp -= 3.0*(mu+2.0/3.0)*exp(tau/mu)
      data.append(tmp)
    I.append(np.array(data))
  return np.array(I)

def FormalIntegral(J, mu_array, tau_array):
  S = J
  I = []
  for mu in mu_array:
    integrand = -S*exp(-tau_array/mu)/mu
    t = tau_array
    if(mu < 0.0):
      tmp = integrate.cumtrapz(integrand, t, initial=0)*exp(tau_array/mu)
    else:
      tmp = integrate.cumtrapz(integrand[::-1], t[::-1], initial=0)[::-1]*exp(tau_array/mu)
    I.append(tmp)
  return np.array(I)

def check_Intensity(I, mu_array, tau_array, it):
  fig1 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Intensity (N$_\mathrm{iter}$ = %d)"%it)
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"$I/H_0$")
  Nout = int(len(mu_array))//4
  for imu,mu in enumerate(mu_array):
    if(imu%Nout != 0): continue
    ax1.plot(tau_array, I[imu], label=r"num $\mu$=%.1f"%(mu))
    ax1.plot(tau_array, Intensity_Abs(tau_array, mu), "--", label=r"ana $\mu$=%.1f"%(mu))
  ax1.legend()
  if(SAVE == 0): plt.show()
  else:
    fig1.savefig(SaveIter_fp + "/Intensity_it=%d.png"%it)
    # plt.clf()
    plt.close()

def check_Moment(J, H ,K, tau, it):
  tau0 = 1
  fig1 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Moment (N$_\mathrm{iter}$ = %d)"%it)
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"moment")
  ax1.plot(tau[tau<tau0],JfromI_Ana(tau[tau<tau0]),"--",color="red",label="J (from analytic I)")
  ax1.plot(tau[tau<tau0],J[tau<tau0],"-",color="red",label="J (numeric)")
  ax1.plot(tau[tau<tau0],HfromI_Ana(tau[tau<tau0]),"--",color="blue",label="H (from analytic I)")
  ax1.plot(tau[tau<tau0],H[tau<tau0],"-",color="blue",label="H (numeric)")
  ax1.plot(tau[tau<tau0],KfromI_Ana(tau[tau<tau0]),"--",color="green",label="K (from analytic I)")
  ax1.plot(tau[tau<tau0],K[tau<tau0],"-",color="green",label="K (numeric)")
  ax1.legend()

  fig2 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax2 = fig2.add_subplot(111)
  set_ax(ax2)
  ax2.set_title("Eddington tensor f = K/J (N$_\mathrm{iter}$ = %d)"%it)
  ax2.set_xlabel(r"$\tau$")
  ax2.set_ylabel(r"K/J")
  ax2.set_ylim(ymin=0.0,ymax=0.5)
  ax2.plot(tau[tau<tau0],KfromI_Ana(tau[tau<tau0])/JfromI_Ana(tau[tau<tau0]),"--",color="black",label="from analytic I")
  ax2.plot(tau[tau<tau0],K[tau<tau0]/J[tau<tau0],"-",color="black",label="numeric")
  if(SAVE == 0): plt.show()
  else:
    fig1.savefig(SaveIter_fp + "/Moment_it=%d.png"%it)
    fig2.savefig(SaveIter_fp + "/EddingtonTensor_it=%d.png"%it)
    plt.close()

def LambdaIteration():
  Nmu, Ntau, Niter = 20, 1000, 1003
  taumin, taumax = 0.0, 30.0
  mu_array = np.linspace(-1.0,1.0,Nmu)
  tau_array = np.linspace(taumin,taumax,Ntau)
  J_ini = 3.0*(tau_array+2.0/3.0)

  J = J_ini

  for it in range(Niter):
    I = FormalIntegral(J, mu_array, tau_array)
    J = JfromI_Num(I, mu_array, tau_array)
    H = HfromI_Num(I, mu_array, tau_array)
    K = KfromI_Num(I, mu_array, tau_array)
    if(it%50 == 0):
      print("iteration = %d"%it)
      check_Intensity(I, mu_array, tau_array, it)
      check_Moment(J, H, K, tau_array, it)


# plot_Moment(0.0)
# plot_Moment(0.5)
# plot_Moment(1.0)
# plot_Moment(1.5)
# plot_Moment(2.0)
# plot_Moment(5.0)
# plot_Moment(10.0)

# plot_Temperature()
# plot_Intensity()
# plot_EddingtonTensor()

# LambdaIteration()

for a in [1,2,3,4]:
  for b in [4,5,6,7,8,9,10,11,12,13,14,15]:
    res = (a+b*0.8)/(a+b)
    # print("a=%f b=%f"%(a,b))
    # print(res)

for N in [2,3,4,5,6,7,8,9,10,100,1000,10000,1e9]:
  res = (1+6*0.8 +(1+5*0.8)*(N-2))/(7+6*(N-2))
  print("N=%d"%(N))
  print(res)