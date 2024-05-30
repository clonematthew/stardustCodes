# Define constants that I'm constantly using 
def uDist():
    return 1e17

def uMass():
    return 1.991e33

def uRho():
    return 1.991e33 / (1e17**3)

def uTime():
    return 3.644e-13

def uVel():
    return 36447.2682

def uEnergy():
    return  1.328e9 

def G():
    return 6.67e-8

def mProt():
    return 1.67e-24

def kB():
    return 1.38e-16

def AU():
    return 1.5e13

def pc():
    return 3.09e18

def year():
    return 60 * 60 * 24 * 365

def colours():
    return ['#d7191c','#fdae61','#ffffbf','#abd9e9','#2c7bb6']

def colourMap():
    import matplotlib
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colours())

# Define useful equations
import numpy as np

# Temp in K and n in cm^-3
def jeansMass(T, n):
    top = 375 * (kB()**3) * (T**3)
    bottom = 4 * np.pi * ((2 * mProt())**4) * (G()**3) * n
    return np.sqrt(top/bottom) / uMass()

def jeansLength(T, n):
    top = 15 * kB() * T
    bottom = 4 * np.pi * G() * n * (2 * mProt())**2
    return np.sqrt(top/bottom) / 1.5e13

def sizeFromDense(n, mass):
    rho = n * mProt() * 1.4
    return ((mass / rho) * (3/4) * (1/np.pi))**(1/3)

