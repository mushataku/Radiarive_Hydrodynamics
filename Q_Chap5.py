# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from math import sqrt, pi
from matplotlib.animation import FuncAnimation
from numpy import exp, sin, cos, cosh, abs
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

def Q_5_6():
  phi1 = lambda theta,a : 1.0/(4.0*pi)*(1+a*cos(theta))
  phi2 = lambda theta,a : 1.0/(4.0*pi)/(1+4*a/3)*(1+a*(1+cos(theta))**2)
  phi3 = lambda theta,a : 1.0/(4.0*pi)/(1+2*a)*( 1+a*(1+cos(theta) )**3)
  fig = plt.figure(figsize=(8, 8))
  # fig1.subplots_adjust(left=0.2)
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  set_ax(ax1)
  set_ax(ax2)
  theta = np.linspace(-pi,pi,100)
  a_list = [0,0.1,0.3,0.5,0.9,1.0,1.5]
  for a in a_list:
    norm = (a-a_list[0])/(a_list[-1]-a_list[0])
    ax1.plot(theta,phi2(theta,a),"-",color=cm.jet(norm),label="a=%.1f"%a)
    if(a == a_list[-1]):
      ax1.plot(theta,phi2(theta,a),"-",color=cm.jet(norm),label=r"$\phi_2$")
  
  a = 0.5
  ax2.plot(theta,phi1(theta,a),"--", label=r"$\phi_1$")
  ax2.plot(theta,phi2(theta,a),"-", label=r"$\phi_2$")
  ax2.plot(theta,phi3(theta,a),"*", label=r"$\phi_3$")

  ax1.set_xlabel(r"$\theta$")
  ax1.set_ylabel(r"$\phi(\theta)$")
  ax1.legend()
  ax2.set_xlabel(r"$\theta$")
  ax2.set_ylabel(r"$\phi(\theta)$")
  ax2.legend()
  # plt.show()
  plt.savefig("./figs/Q_5_6.png")
Q_5_6()