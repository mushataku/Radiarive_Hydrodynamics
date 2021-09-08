# -*- coding: utf-8 -*-
from posix import EX_DATAERR
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import log, sqrt, pi, tau
from matplotlib.animation import FuncAnimation
from numpy import exp, sin, cos, arcsin, log, cosh, abs
from matplotlib.ticker import ScalarFormatter
from scipy import integrate
import os
import csv

plt.rcParams["font.size"] = 18

##############################
SAVE_dir = "./figs/magic_number/"
os.makedirs(SAVE_dir, exist_ok=True)
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

# 星からの一様光源を仮定し、星の見込み角を入れるとmoment量と星の大きさの比 R を返す
def cE_F_cP_R(theta):
  cE = 1.0-cos(theta)
  F  = 0.5*sin(theta)*sin(theta)
  cP = (1.0-cos(theta)**3)/3.0
  R  = sin(theta)
  return cE,F,cP,R

def beta_magic(theta):
  cE,F,cP,_ = cE_F_cP_R(theta)
  return ( (cE+cP)-((cE+cP)**2-4*F*F) )/(2*F)

def x_to_theta(x):
  ret = x.copy()
  tmp = x > 1.0
  ret[tmp] = arcsin(1.0/x[tmp])
  ret[~tmp] = pi/2
  return ret
def theta_to_x(theta):
  ret = theta.copy()
  tmp = (0.0 < theta) & (theta < pi/2)
  ret[tmp] = 1.0/sin(theta[tmp])
  ret[theta>pi/2] = 1.0
  ret[theta<0] = np.inf
  return ret

def plot_magic_number():
  fig1 = plt.figure(figsize=(8, 8))
  fig1.subplots_adjust(left=0.2)
  # theta = np.linspace(pi/10,pi/4.0)
  theta = np.linspace(pi/20,pi/2.0)[::-1]
  beta = beta_magic(theta)
  _,_,_,R = cE_F_cP_R(theta)

  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("magic-number")
  ax1.set_xlabel(r"$\theta$")
  ax1.set_xticks([0,pi/10,pi/6,pi/4,pi/2])
  ax1.set_xticklabels(["0",r"$\pi$/10",r"$\pi$/6",r"$\pi$/4",r"$\pi$/2"])
  ax1.set_ylabel(r"$\beta_{\mathrm{magic}}$")
  ax1.set_ylim(ymin=0,ymax=1.1)
  # ax1.set_xlim(xmin=0.9,xmax=2)
  ax1.plot(theta, beta, ".")
  ax1.invert_xaxis()

  ax2 = ax1.secondary_xaxis("top", functions=(theta_to_x, x_to_theta))
  ax2.set_xlabel(r"r/R$_{\mathrm{star}}$")
  plt.tight_layout()
  fig1.savefig(SAVE_dir+"/magic_number.png")

def plot_EddingtonTensor():
  fig1 = plt.figure(figsize=(8, 8))
  fig1.subplots_adjust(left=0.2)
  # theta = np.linspace(pi/10,pi/4.0)
  theta = np.linspace(pi/20,pi/2.0)
  cE,_,cP,_ = cE_F_cP_R(theta)

  ax1 = fig1.add_subplot(111)
  set_ax(ax1)
  ax1.set_title("Eddington tensor")
  ax1.set_xlabel(r"$\theta$")
  ax1.set_xticks([0,pi/10,pi/6,pi/4,pi/2])
  ax1.set_xticklabels(["0",r"$\pi$/10",r"$\pi$/6",r"$\pi$/4",r"$\pi$/2"])
  ax1.set_ylabel("f")
  ax1.set_ylim(ymin=0,ymax=1.1)
  # ax1.set_xlim(xmin=0.9,xmax=2)
  ax1.plot(theta, cP/cE, ".")
  ax1.plot(theta, [1/3]*len(theta), "--", label="1/3")
  ax1.invert_xaxis()
  ax1.legend()

  ax2 = ax1.secondary_xaxis("top", functions=(theta_to_x, x_to_theta))
  ax2.set_xlabel(r"r/R$_{\mathrm{star}}$")
  plt.tight_layout()
  fig1.savefig(SAVE_dir+"/EddingtonTensor.png")


plot_magic_number()
plot_EddingtonTensor()

# def ThetaToX(theta):
#   # ret = theta.copy()
#   ret = sin(theta)
#   print("theta")
#   print(theta)
#   print(ret)
#   return ret

# def XToTheta(x):
#   ret = x.copy()
#   tmp = (-1 < x) & (x < 1)
#   print("x")
#   print(x)
#   print(tmp)
#   ret[tmp] = arcsin(x[tmp])
#   ret[[x > 1.0]] = pi/2.0
#   ret[[x < -1.0]] = -pi/2.0
#   print(ret)
#   return ret


# def test():
#   fig1 = plt.figure(figsize=(8, 8))
#   fig1.subplots_adjust(left=0.2)
#   theta = np.linspace(0,pi/2.1)
#   ax1 = fig1.add_subplot(111)
#   ax1.set_xticks([0,pi/4,pi/2])
#   ax1.set_xticklabels(["0",r"$\pi$/4",r"$\pi$/2"])

#   set_ax(ax1)
#   ax1.plot(theta, sin(theta), ".")

#   ax2 = ax1.secondary_xaxis("top", functions=(ThetaToX, XToTheta))
#   ax2.set_xlabel("x")
#   ax2.set_xticks([0.0,0.2,0.4,0.6,0.8,0.9,0.95,1.0])
#   plt.show()
#   # fig1.savefig(SAVE_dir+"/magic_number.png")

# def test2():
#   fig1 = plt.figure(figsize=(8, 8))
#   fig1.subplots_adjust(left=0.2)
#   x = np.linspace(0.0,1.0)
#   ax1 = fig1.add_subplot(111)

#   set_ax(ax1)
#   ax1.plot(x, arcsin(x), ".")
#   ax1.set_yticklabels(["0",r"$\pi$/4",r"$\pi$/2"])
#   ax1.set_yticks([0,pi/4,pi/2])

#   ax2 = ax1.secondary_xaxis("top", functions=(XToTheta, ThetaToX))
#   ax2.set_xlabel("x")
#   ax2.set_xticks([0,pi/4,pi/2])
#   ax2.set_xticklabels(["0",r"$\pi$/4",r"$\pi$/2"])
#   plt.show()
#   # fig1.savefig(SAVE_dir+"/magic_number.png")

# test()
# test2()



