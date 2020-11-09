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

####### parameter ############
# グラフ描画範囲
x = np.linspace(0,10)
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

def Spherical_Eddington():
  fig1 = plt.figure(figsize=(8, 8))
  fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  tau = np.linspace(0,1000,1000000)
  f_rr = lambda tau : (1+tau)/(1+3*tau)
  f_thth = lambda tau : 0.5*(1-f_rr(tau))
  ax1.plot(tau,f_rr(tau),label=r"$f_{rr}$")
  ax1.plot(tau,f_thth(tau),label=r"$f_{\theta\theta}$")
  ax1.set_title("Eddington factor (spherical)")
  ax1.set_xlabel(r"\tau")
  ax1.set_xscale("log")
  ax1.legend()
  plt.savefig("./figs/Spherical_f")

def FLD():
  fig1 = plt.figure(figsize=(8, 8))
  fig1.subplots_adjust(left=0.2)
  ax1 = fig1.add_subplot(221)
  ax2 = fig1.add_subplot(222)
  ax3 = fig1.add_subplot(223)
  set_ax(ax1)
  set_ax(ax2)
  set_ax(ax3)
  R = np.linspace(0,1000,1000000)
  lam1 = lambda R : (2+R)/(6+3*R+R*R)

  f = lambda R : lam1(R)+lam1(R)*lam1(R)*R*R
  f_rr = lambda R : 0.5*(1-f(R))

  ax1.plot(R,lam1(R),label=r"1")
  ax2.plot(R,f(R),label=r"1")
  ax3.plot(R,f_rr(R),label=r"1")
  # ax1.set_title("Eddington factor (spherical)")
  ax1.set_xlabel(r"R")
  ax2.set_xlabel(r"R")
  ax3.set_xlabel(r"R")
  ax1.set_xscale("log")
  ax2.set_xscale("log")
  ax3.set_xscale("log")
  ax1.legend()
  ax2.legend()
  ax3.legend()
  plt.savefig("./figs/FLD_f")

# Spherical_Eddington()
# FLD()