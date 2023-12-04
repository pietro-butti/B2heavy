import numpy as np
import gvar as gv

ensDir = './'

#   Constants for the energy priors

y0   =  1.
y0s  =  1.

y2   = -2.5
y2s  =  5.
z2   =  0.
z2s  =  1.

y4   =  3.
y4s  =  5.
z4   =  1.
z4s  =  2.

y4p  =  0.
y4ps =  0.
z4p  =  0.
z4ps =  2.

#   Functions to determine the energy priors

def m1_0(m0):
    return np.log(1.+m0)

def m2_0(m0):
    return 1./(2./(m0*(2.+m0)) + 1./(m0+1.))

def invm4_0_cu(m0):
    invm4_0_cuA = 4.*(1. + (1. + m0)*(1 + m0))/(m0*(2. + m0))**3
    invm4_0_cuB = 8.*(1. + m0)/(m0*(2. + m0))**2
    invm4_0_cuC = 1./((1. + m0)*(1. + m0))
    return invm4_0_cuA + invm4_0_cuB + invm4_0_cuC

def invm4_0(m0):
    ovm4  = 8.0/((m0*(2.0 + m0))**3) + (4.0 + 8.0*(1.0 + m0))/((m0*(2.0 + m0))**2) + 1.0/((1.0 + m0)**2)

    return ovm4

def w4_0(m0):
    return 2./(m0*(2. + m0)) + 1./(4.*(1. + m0))

def g0(m0):
    return m0

def l0(m0):
    return 1.

def g2(m0):
    return m1_0(m0) * m0*m0/(m2_0(m0)*(1.+m0)**4)

def h2(m0):
    return m1_0(m0) * m0*m0*m0*(2.+m0)/(m2_0(m0)*(1+m0)**4)

def l2(m0):
    return 1./m2_0(m0)*(1 - m1_0(m0)/m2_0(m0))

def g4(m0):
    return 1./((1.+m0)*(1.+m0))

def h4(m0):
    return 1./((1.+m0)*(1.+m0))*np.log(1. + m0)

def l4(m0):
    return (3.*m1_0(m0)/m2_0(m0) - 1.)*invm4_0_cu(m0) - 2./(m2_0(m0))**3

def g4p(m0):
    return m0/(1.+m0)

def h4p(m0):
    return np.log(1.+m0)/(1.+m0)

def l4p(m0):
    return w4_0(m0)*(1 - m1_0(m0)/m2_0(m0))

def als(aSc):
    return 1./3.*(1. -3./(2.*np.pi)*np.log(0.09/aSc))


def params(ensName):
  pDict = {}

  pDict['hbarc'] = 0.1973269788 #[0.1973269788, 0.0000000012]
  pDict['w0'] = gv.gvar('0.1714(15)')
  pDict['r1'] = gv.gvar('0.3117(22)') # Should I use this number?
  pDict['Ratio ml ms'] = gv.gvar('27.35(11)')
  pDict['Ratio ms mc'] = 1.0 / gv.gvar('11.747(62)')
  pDict['mB'] = 5279.58 #/ inputs['hbarc'] * gv.mean(inputs['w0'])
  pDict['mBs'] = 5366.77 #/ inputs['hbarc'] * gv.mean(inputs['w0'])
  pDict['mK'] = 497.614 #/ inputs['hbarc'] * gv.mean(inputs['w0'])
  pDict['mPi'] = 134.977 #/ inputs['hbarc'] * gv.mean(inputs['w0'])

  pDict['moms']       = ['000', '100', '110', '200', '211', '222', '300', '400']
  pDict['sms']        = ['1S_RW', 'RW_RW', 'rot_rot', 'rot_1S', 'rot_d', '1S_d', 'd_d']
  pDict['lambdaQcd']  = 0.5

  pDict['name']       = ensName
  pDict['hdf5File']   = ensName + ".hdf5"

  if   ensName == "MediumCoarse":
    pDict['folder']     = "Ensembles/FnalHISQ/a0.15/"
    pDict['hdf5File2']  = "l3248f211b580m002426m06730m8447-HISQscript.hdf5"
    pDict['sqLiteFile'] = "l3248f211b580m002426m06730m8447-HISQscript.sqlite"

    pDict['w0a']        =  gv.gvar('1.1468(4)')
    pDict['u0']         =  0.8203
    pDict['naSpc']      =  0.15
    pDict['aSpc']       =  pDict['w0'] / pDict['w0a']
    pDict['alphaS']     =  0.3478 # Check
    pDict['beta']       =  5.80 # Check
    pDict['ml']         =  0.002426
    pDict['mlStr']      = '0.002426'
    pDict['ms']         =  0.06730
    pDict['msStr']      = '0.06730'
    pDict['mc']         =  0.8447
    pDict['mcStr']      = '0.8447'
    pDict['kappaD']     =  0.11879
    pDict['kDStr']      = '0.11879'
    pDict['kappaB']     =  0.07732
    pDict['kBStr']      = '0.07732'
    pDict['kappaCr']    =  0.13951
    pDict['mBs']        =  2.956
    pDict['mDs']        =  1.652

    pDict['L']          = 32
    pDict['T']          = 48
    pDict['nConfs']     = 3630
    pDict['hSinks']     = [10, 11]
    pDict['lSinks']     = [14, 15]

  elif ensName == "Coarse-2":
    pDict['folder']     = "Ensembles/FnalHISQ/a0.12/"
    pDict['hdf5File2']   = "l2464f211b600m0102m0509m635-HISQscript.hdf5"
    pDict['sqLiteFile'] = "l2464f211b600m0102m0509m635-HISQscript.sqlite"

    pDict['w0a']        = gv.gvar('1.3835(10))')
    pDict['u0']         = 0.8350
    pDict['naSpc']      = 0.12
    pDict['aSpc']       = pDict['w0'] / pDict['w0a']
    pDict['alphaS']     = 0.3068
    pDict['beta']       = 6.00
    pDict['ml']         =  0.0102
    pDict['mlStr']      = '0.0102'
    pDict['ms']         =  0.0509
    pDict['msStr']      = '0.0509'
    pDict['mc']         =  0.635
    pDict['mcStr']      = '0.635'
    pDict['kappaD']     =  0.12201
    pDict['kDStr']      = '0.12201'
    pDict['kappaB']     =  0.08574
    pDict['kBStr']      = '0.08574'
    pDict['kappaCr']    =  0.13901
    pDict['mBs']        =  3.187
    pDict['mDs']        =  1.730

    pDict['L']          = 24
    pDict['T']          = 64
    pDict['nConfs']     = 1053
    pDict['hSinks']     = [12, 13]
    pDict['lSinks']     = [17, 18]
    pDict['moms'].remove('400')
    pDict['moms'].remove('222')

  elif ensName == "Coarse-1":
    pDict['folder']     = "Ensembles/FnalHISQ/a0.12/"
    pDict['hdf5File2']   = "l3264f211b600m00507m0507m628-HISQscript.hdf5"
    pDict['sqLiteFile'] = "l3264f211b600m00507m0507m628-HISQscript.sqlite"

    pDict['w0a']        = gv.gvar('1.4047(9)')
    pDict['u0']         = 0.8350
    pDict['naSpc']      = 0.12
    pDict['aSpc']       = pDict['w0'] / pDict['w0a']
    pDict['alphaS']     = 0.3068
    pDict['beta']       = 6.00
    pDict['ml']         =  0.00507
    pDict['mlStr']      = '0.00507'
    pDict['ms']         =  0.0507
    pDict['msStr']      = '0.0507'
    pDict['mc']         =  0.628
    pDict['mcStr']      = '0.628'
    pDict['kappaD']     =  0.12201
    pDict['kDStr']      = '0.12201'
    pDict['kappaB']     =  0.08574
    pDict['kBStr']      = '0.08574'
    pDict['kappaCr']    =  0.13901
    pDict['mBs']        =  3.187
    pDict['mDs']        =  1.730

    pDict['L']          = 32
    pDict['T']          = 64
    pDict['nConfs']     = 1000
    pDict['hSinks']     = [12, 13]
    pDict['lSinks']     = [17, 18]
    pDict['moms'].remove('400')
    pDict['moms'].remove('222')

  elif ensName == "Coarse-Phys":
    pDict['folder']     = "Ensembles/FnalHISQ/a0.12/"
    pDict['hdf5File2']   = "l4864f211b600m001907m05252m6382-HISQscript.hdf5"
    pDict['sqLiteFile'] = "l4864f211b600m001907m05252m6382-HISQscript.sqlite"

    pDict['w0a']        = gv.gvar('1.4168(10)')
    pDict['u0']         = 0.8350
    pDict['naSpc']      = 0.12
    pDict['aSpc']       = pDict['w0'] / pDict['w0a']
    pDict['alphaS']     = 0.3068
    pDict['beta']       = 6.00
    pDict['ml']         =  0.001907
    pDict['mlStr']      = '0.001907'
    pDict['ms']         =  0.05252
    pDict['msStr']      = '0.05252'
    pDict['mc']         =  0.6382 
    pDict['mcStr']      = '0.6382'
    pDict['kappaD']     =  0.12201
    pDict['kDStr']      = '0.12201'
    pDict['kappaB']     =  0.08574
    pDict['kBStr']      = '0.08574'
    pDict['kappaCr']    =  0.13901
    pDict['mBs']        =  3.187
    pDict['mDs']        =  1.730

    pDict['L']          = 48
    pDict['T']          = 64
    pDict['nConfs']     = 986
    pDict['hSinks']     = [12, 13]
    pDict['lSinks']     = [17, 18]

  elif ensName == "Fine-1":
    pDict['folder']     = "Ensembles/FnalHISQ/a0.088/"
    pDict['hdf5File2']   = "l4896f211b630m00363m0363m430-HISQscript.hdf5"
    pDict['sqLiteFile'] = "l4896f211b630m00363m0363m430-HISQscript.sqlite"

    pDict['w0a']        = gv.gvar('1.9299(12)')
    pDict['u0']         = 0.8527
    pDict['naSpc']      = 0.088
    pDict['aSpc']       = pDict['w0'] / pDict['w0a']
    pDict['alphaS']     = 0.2641
    pDict['beta']       = 6.30
    pDict['ml']         =  0.00363
    pDict['mlStr']      = '0.00363'
    pDict['ms']         =  0.0363
    pDict['msStr']      = '0.0363'
    pDict['mc']         =  0.430
    pDict['mcStr']      = '0.430'
    pDict['kappaD']     =  0.12565
    pDict['kDStr']      = '0.12565'
    pDict['kappaB']     =  0.09569
    pDict['kBStr']      = '0.09569'
    pDict['kappaCr']    =  0.13811
    pDict['mBs']        =  3.603
    pDict['mDs']        =  1.830

    pDict['L']          = 48
    pDict['T']          = 96
    pDict['nConfs']     = 1017
    pDict['hSinks']     = [17, 18]
    pDict['lSinks']     = [25, 26]

  elif ensName == "Fine-Phys":
    pDict['folder']     = "Ensembles/FnalHISQ/a0.088/"
    pDict['hdf5File2']   = "l6496f211b630m0012m0363m432-HISQscript.hdf5"
    pDict['sqLiteFile'] = "l6496f211b630m0012m0363m432-HISQscript.sqlite"

    pDict['w0a']        = gv.gvar('1.9470(13)')
    pDict['u0']         = 0.8527
    pDict['naSpc']      = 0.088
    pDict['aSpc']       = pDict['w0'] / pDict['w0a']
    pDict['alphaS']     = 0.2641
    pDict['beta']       = 6.30
    pDict['ml']         =  0.0012
    pDict['mlStr']      = '0.0012'
    pDict['ms']         =  0.0363
    pDict['msStr']      = '0.0363'
    pDict['mc']         =  0.432
    pDict['mcStr']      = '0.432'
    pDict['kappaD']     =  0.12565
    pDict['kDStr']      = '0.12565'
    pDict['kappaB']     =  0.09569
    pDict['kBStr']      = '0.09569'
    pDict['kappaCr']    =  0.13811
    pDict['mBs']        =  3.603
    pDict['mDs']        =  1.830

    pDict['L']          = 64
    pDict['T']          = 96
    pDict['nConfs']     = 1535
    pDict['hSinks']     = [17, 18]
    pDict['lSinks']     = [25, 26]

  elif ensName == "SuperFine":
    pDict['folder']     = "Ensembles/FnalHISQ/a0.057/"
    pDict['hdf5File2']   = "l96192f211b672m0008m022m260-HISQscript.hdf5"
    pDict['sqLiteFile'] = "l96192f211b672m0008m022m260-HISQscript.sqlite"

    pDict['w0a']        = gv.gvar('3.0119(19)')
    pDict['u0']         = 0.8711
    pDict['naSpc']      = 0.057
    pDict['aSpc']       = pDict['w0'] / pDict['w0a']
    pDict['alphaS']     = 0.2235 # Check
    pDict['beta']       = 6.72   # Check
    pDict['ml']         =  0.0008
    pDict['mlStr']      = '0.0008'
    pDict['ms']         =  0.022
    pDict['msStr']      = '0.022'
    pDict['mc']         =  0.260
    pDict['mcStr']      = '0.260'
    pDict['kappaD']     =  0.129311
    pDict['kDStr']      = '0.129311'
    pDict['kappaB']     =  0.10604
    pDict['kBStr']      = '0.10604'
    pDict['kappaCr']    =  0.13698
    pDict['mBs']        =  4.244
    pDict['mDs']        =  1.911

    pDict['L']          = 96
    pDict['T']          = 192
    pDict['nConfs']     = 1027
    pDict['hSinks']     = [24, 25]
    pDict['lSinks']     = [36, 37]

  pDict['lQcd']       = pDict['lambdaQcd']/pDict['hbarc']*pDict['aSpc'].mean

#   M1 = Rest mass prior, won't be used anyway

  #   Bare mass
  m0D = 0.5/pDict['u0']*(1./pDict['kappaD'] - 1/pDict['kappaCr'])
  m0B = 0.5/pDict['u0']*(1./pDict['kappaB'] - 1/pDict['kappaCr'])

  pDict['m2c'] = m2_0(m0D)
  pDict['m2b'] = m2_0(m0B)

  #   mth
  mThD = m0D*(2.+m0D)/(((1.+m0D)**2)+1.)
  mThB = m0B*(2.+m0B)/(((1.+m0B)**2)+1.)

  #   One loop correction

  m1cD = -(2./np.pi)*mThD*np.log(mThD) + g0(mThD)*y0
  m1cB = -(2./np.pi)*mThB*np.log(mThB) + g0(mThB)*y0

  #   Prior and error
    #        Quarks        1-loop corr         binding energy
  pDict['mDth']  = m1_0(m0D) + als(pDict['aSpc'].mean)*m1cD + pDict['lQcd']*l0(m0D)
  pDict['mBth']  = m1_0(m0B) + als(pDict['aSpc'].mean)*m1cB + pDict['lQcd']*l0(m0B)

  pDict['mDsth'] = np.sqrt((y0s*g0(mThD)*als(pDict['aSpc'].mean))**2 + (pDict['lQcd']*l0(m0D))**2)
  pDict['mBsth'] = np.sqrt((y0s*g0(mThB)*als(pDict['aSpc'].mean))**2 + (pDict['lQcd']*l0(m0B))**2)

#   c^2 = M1/M2 Speed of light prior, only for D* meson

  c2_0 = m1_0(m0D)/m2_0(m0D)                      # Tree level
  c2_1 = y2*g2(m0D) + z2*h2(m0D)                  # 1-loop correction

  pDict['c2']   = c2_0 + als(pDict['aSpc'].mean)*c2_1 + pDict['lQcd']*l2(m0D)
  pDict['c2s']  = np.sqrt((y2s*g2(m0D)*als(pDict['aSpc'].mean))**2 + (z2s*h2(m0D)*als(pDict['aSpc'].mean))**2 + (pDict['lQcd']*l2(m0D))**2)

#   A4' = M1 x W4 prior

  cq_0 = w4_0(m0D)*m1_0(m0D)                      # Tree level
  cq_1 = y4p*g4p(m0D) + z4p*h4p(m0D)              # 1-loop correction

  pDict['A4p']  = (-1./3.)*(cq_0 + als(pDict['aSpc'].mean)*cq_1 + pDict['lQcd']*l4p(m0D))
  pDict['A4ps'] = ( 1./3.)*np.sqrt((y4ps*g4p(m0D)*als(pDict['aSpc'].mean))**2 + (z4ps*h4p(m0D)*als(pDict['aSpc'].mean))**2 + (pDict['lQcd']*l4p(m0D))**2)   # Error l2 --> l4p

#   A4 = cp prior (A4)

  cp_0 = 1/(m2_0(m0D)*m2_0(m0D)) - m1_0(m0D)*invm4_0_cu(m0D)  # Tree level
  cp_1 = y4*g4(m0D) + z4*h4(m0D)                              # 1-loop correction

  pDict['A4']   = 0.25*(cp_0 + als(pDict['aSpc'].mean)*cp_1 + pDict['lQcd']*l4(m0D))       # Error 1 --> als(p['aSpc'])*cp_1
  pDict['A4s']  = 0.25*np.sqrt((y4s*g4(m0D)*als(pDict['aSpc'].mean))**2 + (z4s*h4(m0D)*als(pDict['aSpc'].mean))**2 + (pDict['lQcd']*l4(m0D))**2)

  return pDict
