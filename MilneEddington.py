# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt, pi
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

Save_fp = "./figs/MilneEddington"

for fp in [Save_fp]:
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

def Intensity_NoAbs_(tau,mu):
  if(mu > -1e-5): return 3.0*(tau + mu + 2.0/3.0)
  else: return 3.0*(tau + mu + 2.0/3.0 - (mu + 2.0/3.0)*exp(tau/mu))

def Intensity_NoAbs(tau,mu):
  if(type(mu) == float): return Intensity_NoAbs_(tau,mu)
  else: return np.array([Intensity_NoAbs_(tau,mu_) for mu_ in mu])


def JfromI_(tau):
  # 被積分関数fを定義
  f = lambda mu : 0.5*Intensity_NoAbs(tau,mu)
  return integrate.quad(f,-1.0,1.0)[0]
def JfromI(tau):
  if(type(tau) == float): return JfromI_(tau)
  else: return np.array([JfromI_(tau_) for tau_ in tau])

def HfromI_(tau):
  # 被積分関数fを定義
  f = lambda mu : 0.5*mu*Intensity_NoAbs(tau,mu)
  return integrate.quad(f,-1.0,1.0)[0]
def HfromI(tau):
  if(type(tau) == float): return HfromI_(tau)
  else: return np.array([HfromI_(tau_) for tau_ in tau])

def KfromI_(tau):
  # 被積分関数fを定義
  f = lambda mu : 0.5*mu*mu*Intensity_NoAbs(tau,mu)
  return integrate.quad(f,-1.0,1.0)[0]
def KfromI(tau):
  if(type(tau) == float): return KfromI_(tau)
  else: return np.array([KfromI_(tau_) for tau_ in tau])

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
    ax1.plot(tau,Intensity_NoAbs(tau, mu), label=r"$\mu$=%.1f"%(mu))
  mubar = 1.0/sqrt(3)
  ax1.plot(tau,Ip_TSA(tau, mubar), "--", color="red", label=r"$I^{+}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  ax1.plot(tau,Im_TSA(tau, mubar), "-.", color="red", label=r"$I^{-}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  mubar = 1.0
  ax1.plot(tau,Ip_TSA(tau, mubar), "--", color="blue", label=r"$I^{+}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  ax1.plot(tau,Im_TSA(tau, mubar), "-.", color="blue", label=r"$I^{-}_\mathrm{TSA}$ ($\bar{\mu}$=%.2f)"%mubar)
  ax1.legend(fontsize=12)
  if(SAVE == 0): plt.show()
  else: fig1.savefig(Save_fp + "/Intensity_NoAbs.png")

def plot_Moment():
  fig1 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Moment")
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"(J,H,K)/H$_0$")
  tau = np.linspace(0,1,1000)
  J = 3.0*(tau+2.0/3.0)
  H = [1.0]*len(tau)
  K = 1.0*(tau+2.0/3.0)

  cols = ["red", "blue", "green"]
  
  ax1.plot(tau,J,"--",color="red",label="J (Analytic)")
  ax1.plot(tau,JfromI(tau),"-",color="red",label="J (from I)")
  ax1.plot(tau,H,"--",color="blue",label="H (Analytic)")
  ax1.plot(tau,HfromI(tau),"-",color="blue",label="H (from I)")
  ax1.plot(tau,K,"--",color="green",label="K (Analytic)")
  ax1.plot(tau,KfromI(tau),"-",color="green",label="K (from I)")

  ax1.legend()

  if(SAVE == 0): plt.show()
  else: fig1.savefig(Save_fp + "/Moment_NoAbs_tau1.png")

def plot_EddingtonTensor():
  fig1 = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Eddington tensor f = K/J")
  ax1.set_xlabel(r"$\tau$")
  ax1.set_ylabel(r"K/J")
  ax1.set_ylim(ymin=0.0,ymax=0.5)
  tau = np.linspace(0,3,100)
  ax1.plot(tau,[1.0/3.0]*len(tau),"--",color="black",label="1/3")
  ax1.plot(tau,KfromI(tau)/JfromI(tau),"-",color="black",label="K/J (from I)")

  ax1.legend()

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


plot_Temperature()
# plot_Intensity()
# plot_Moment()
# plot_EddingtonTensor()