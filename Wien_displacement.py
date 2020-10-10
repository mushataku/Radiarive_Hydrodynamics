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
# 二分探索の終了条件
EPS = 1e-12
# 二分探索の最大反復回数
ITERATION_MAX = 100000
# グラフ描画範囲
x = np.linspace(0,10)
##############################

def f(x):
  return (1-x/3.0)*exp(x) - 1
def g(y):
  return (y/5.0-1)*exp(y) + 1

# グラフの体裁を整える
def set_ax(ax):
  ax.set_ylabel("y")
  ax.set_xlabel("x")
  ax.grid(linestyle="dotted")
  ax.xaxis.set_tick_params(direction='in')
  ax.yaxis.set_tick_params(direction='in')
  ax.tick_params(labelsize=21)
  # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  # ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
  ax.yaxis.offsetText.set_fontsize(20)
  ax.set_ylim(-10,10)

# 二分探索による求解部分
def nibutan(func, x0, eps, iter_max):
  # 今の符号を覚えておく
  sign = (func(x0) > 0)
  dx = x0*0.1
  for _ in range(iter_max):
    # 一歩進める
    x0 += dx
    # 一歩進んだら符号が反転した場合
    if(sign != (func(x0) > 0)):
      sign = not sign # 符号反転
      dx *= -0.5
      if(abs(dx) < eps):
        return x0
  # iter_max 解反復しても収束しなかったらエラー報告
  print("########## nibutan does not converge ############")
  return -9999999999

# 二分探索で f(x)=0 の解を求める 
def Solve(func):
  fig = plt.figure(figsize=(8, 8))
  fig.subplots_adjust(left=0.2)
  ax = fig.add_subplot(111)
  set_ax(ax)
  ax.plot(x,func(x))
  plt.pause(interval=0.1)
  # 初期値はグラフを見て手で与えるようにした
  x0 = float(input("choose initial x0 : "))
  plt.close()

  # 二分探索で球解
  ret = nibutan(func,x0,EPS,ITERATION_MAX)
  return ret

x0 = Solve(f)
y0 = Solve(g)

print("x0=%.10e, y0=%.10e"%(x0,y0))