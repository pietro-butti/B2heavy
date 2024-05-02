import tomllib
import gvar as gv
import pickle
import os

def load_toml(file) -> dict:
    with open(file,'rb') as f:
        toml_data: dict = tomllib.load(f)
    return toml_data


def Ndof(Npol,trange,nexc,Nsmr=2):
    Npar = 2*nexc + 2*nexc + 2*nexc + 2*(nexc-1)
    Npoints = Npol*(Nsmr*(Nsmr+1)//2)*(max(trange)-min(trange))
    return Npoints-Npar
