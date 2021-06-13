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

##################### constant #####################
au = 1.4959e13 # cm
pc = 3.09e18 # cm
M_sun = 2.0e33 # g
R_sun = 6.96e10 # cm
print("%e"%R_sun)
G = 6.67430e-8 # cm^3 g^-1 s^-2
yr = 365*24*3600 # s
c = 2.9979e10 # cm/s
##################### constant #####################

def problem_chap4():
  print("############ Problem 4.7 ############")
  f = 1.37e6
  # (1)
  print("(1)")
  dOmega = pi*(959.63/3600.0*pi/180.0)**2.0
  I = f/dOmega
  print("dOmega = %e [Sr]"%dOmega)
  print("I = %e [erg s^-1 cm^-2 Sr^-1]"%I)
  # (2)
  print("(2)")
  F = I * pi
  print("F = %e [erg s^-1 cm^-2]"%F)
  # (3)
  print("(3)")
  L_F = F*4.0*pi*R_sun*R_sun
  L_f = f*4.0*pi*au*au
  print("L_F = %e [erg s^-1]"%L_F)
  print("L_f = %e [erg s^-1]"%L_f)

  print("############ Problem 4.8 ############")
  W = R_sun*R_sun/(4.0*au*au)
  print("W = %e"%W)

  # W_ = dOmega/(4.0*pi)
  # print("W_ = %e"%w_)

  print("############ Problem 4.9 ############")
  F_6 = f * (10 ** (-32.7/2.5))
  print("F_6 = %e"%F_6)

problem_chap4()