import h5py

import pandas as pd
import numpy as np

from b2heavy import FnalHISQMetadata
from b2heavy.ThreePointFunctions.types3pts import test

def jkGen(data, binSize=1):
  rData = (data.shape[0]//binSize)
  nData = rData*binSize
  dIn   = data[0:nData,:]

  avg   = dIn.mean(axis=0)

  base  = dIn.sum(axis=0) # avg * nData

  dOut  = dIn * 0.0

  for cf in range(rData):
    pvg = dIn[cf*binSize:(cf+1)*binSize,:].sum(axis=0)
    dOut[cf,:] = (base - pvg)/(nData - binSize)

  return dOut[:rData,:]

def selectCols(key, tRange=[4,15]):
    return [ key+'_'+str(i) for i in range(tRange[0], tRange[1]) ]


# test()

def getThreepHdf5(ensName, ratio, mom, sms = ['RW', '1S'], eType ='jackknife', binSize=1): # Alternative 'bootstrap'
    mData   = FnalHISQMetadata.params(ensName)
    # ensFile = '/Users/pietro/code/data_analysis/BtoD/Alex/' + mData['folder'] + mData['hdf5File']
    ensFAlt = '/Users/pietro/code/data_analysis/BtoD/Alex/' + mData['folder'] + mData['hdf5File2']
    ensFile = ensFAlt
    data    = h5py.File(ensFile, 'r')
    dAlt    = h5py.File(ensFAlt, 'r')
    tSinks  = mData['hSinks']
    T       = tSinks[0]		# The lowest tSink


    if isinstance(sms, type(None)):
      sms   = mData['sms']

    bStr    = '_k' + mData['kBStr'] #getMassStr(mData['kappaB'])
    cStr    = '_k' + mData['kDStr'] #getMassStr(mData['kappaD'])
    sStr    = '_m' + mData['msStr'] #getMassStr(mData['ms'])
    mStr    = '_m' + mData['mlStr'] #getMassStr(mData['ml'])

    if   ratio.upper() == 'RA1':
      hStr   = bStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['P5_A2_V2_'], ['P5_A2_V2_']] # 'R'
      nFacs  = [[1., 1.], [ 1., 1.]]
      dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
    elif ratio.upper() == 'ZRA1':
      hStr   = bStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
      nFacs  = [[1., 1., 1.], [1., 1.]]
      dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
    elif ratio.upper() == 'ZRA1S':
      hStr   = bStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_']] # 'R'
      nFacs  = [[1., 1., 1.], [1., 1.]]
      dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
    elif ratio.upper() == 'RA1S':
      hStr   = bStr
      lStr   = cStr
      qStr   = sStr
      nNames = [['P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
      nFacs  = [[1., 1.], [1., 1., 1.]]
      dNames = [['V1_V4_V1_'],['P5_V4_P5']] # 'V1_V4_V2_', 'V1_V4_V3_']
    elif ratio.upper() == 'R0':
      hStr   = bStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['P5_A4_V1_']]
      nFacs  = [[1.]]
      dNames = ['P5_A2_V2_', 'P5_A3_V3_']
    elif ratio.upper() == 'R0S':
      hStr   = bStr
      lStr   = cStr
      qStr   = sStr
      nNames = [['P5_A4_V1_']]
      nFacs  = [[1.]]
      dNames = ['P5_A2_V2_', 'P5_A3_V3_']
    elif ratio.upper() == 'R1':
      hStr   = bStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['P5_A1_V1_']]
      nFacs  = [[1.0]]
      dNames = ['P5_A2_V2_', 'P5_A3_V3_']
    elif ratio.upper() == 'R1S':
      hStr   = bStr
      lStr   = cStr
      qStr   = sStr
      nNames = [['P5_A1_V1_']]
      nFacs  = [[0.5]]
      dNames = ['P5_A2_V2_', 'P5_A3_V3_']
    elif ratio.upper() == 'XV':
      hStr   = bStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['P5_V3_V2_', 'P5_V2_V3_']]
      nFacs  = [[1., -1.]]
      dNames = ['P5_A2_V2_', 'P5_A3_V3_']
    elif ratio.upper() == 'XVS':
      hStr   = bStr
      lStr   = cStr
      qStr   = sStr
      nNames = [['P5_V3_V2_', 'P5_V3_V2_']]
      nFacs  = [[1., -1.]]
      dNames = ['P5_A2_V2_', 'P5_A3_V3_']
    elif ratio.upper() == 'XFSTPAR':
      hStr   = cStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['V1_V1_V1_']]
      nFacs  = [[1.]]
      dNames = ['V1_V4_V1_']
    elif ratio.upper() == 'XFSTBOT':
      hStr   = cStr
      lStr   = cStr
      qStr   = mStr
      nNames = [['V1_V2_V1_', 'V1_V3_V1_']]
      nFacs  = [[1., 1.]]
      dNames = ['V1_V4_V2_', 'V1_V4_V3_']
    elif ratio.upper() == 'XFSSTPAR':
      hStr   = cStr
      lStr   = cStr
      qStr   = sStr
      nNames = [['V1_V1_V1_']]
      nFacs  = [[1.]]
      dNames = ['V1_V4_V1_']
    elif ratio.upper() == 'XFSSTBOT':
      hStr   = cStr
      lStr   = cStr
      qStr   = sStr
      nNames = [['V1_V2_V1_', 'V1_V3_V1_']]
      nFacs  = [[1., 1.]]
      dNames = ['V1_V4_V3_']#, 'V1_V4_V3_']
    else:
      print("Unrecognized ratio", ratio.upper())
      return

    crData   = {}

    genDataset = jkGen
    # if ratio.upper()[0:3] == 'RA1' or ratio.upper()[0:2] == 'ZR':
    #   mesonData = gzip.open("../2pts/data/%s.Dsst.000.Twop.PyDat" % ensName, "rb") if 'S' in ratio.upper() else gzip.open("../2pts/data/%s.Dst.000.Twop.PyDat" % ensName, "rb")
    #   tFit      = IOFit(mesonData)
    #   m0        = tFit.p['E'][0].mean
      
    # else:
    #   m0 = 0.0
    #   E0 = 0.0

    crData = pd.DataFrame([], columns=[])
    momStr = '_p' + mom

    # if ratio.upper()[0:3] == 'RA1' or ratio.upper()[0:2] == 'ZR':
    #   if mom != '000':
    #     try:
    #       mesonData = gzip.open("../2pts/data/%s.Dsst.%s.Twop.PyDat" % (ensName, mom), "rb") if 'S' in ratio.upper() else gzip.open("../2pts/data/%s.Dst.%s.Twop.PyDat" % (ensName, mom), "rb")
    #     except FileNotFoundError:
    #       return
    #     tFit      = IOFit(mesonData)
    #     E0        = tFit.p['E'][0].mean
    #   else:
    #     E0 = m0

    m0 = 0.
    E0 = 0.


    RETURN = {}


    for hss in sms:
      #if 'rot' in lss or 'rot' in hss: # FIXME
      #  continue
      ratioCorr = None

      if ratio.upper() != 'ZRA1S' and ratio.upper() != 'RA1S' and ratio.upper() != 'ZRA1' and ratio.upper()[0:3] != 'RA1' and mom != '000':          # At zero momentum only RA1(s) matters
        tRatioCorr = []

        for tSink in tSinks:
          iData = []

          for j,tup in enumerate(zip(nNames[0],nFacs[0])):
            name, nFac = tup
            corrName = name + 'T' + str(tSink) + hStr + '_RW_' + hss + '_rot_rot' + qStr + lStr + momStr
            print(f'### nu: {corrName}')  

            try:
                iData.append(data['data'][corrName][()]*nFac)
                print(data['data'][corrName][()]*nFac)
            except KeyError:
                return
                # try:
                #     iData.append(np.array(dAlt['data'][corrName])*nFac)
                # except KeyError:
                #     return


          iData  = np.array(iData).mean(axis=0)
          nuCorr = genDataset(iData, binSize = binSize)

          if len(dNames) > 0:
            iData = []

            for j,name in enumerate(dNames):
                corrName = name + 'T' + str(tSink) + hStr + '_RW_' + hss + '_rot_rot' + qStr + lStr + momStr
                # print(f'### du: {corrName}')  
       
                try:
                    iData.append(data['data'][corrName][()])
                except KeyError:
                    return
                    # print("Correlator", name, "will come from original file.")
                    # try:
                    #     iData.append(np.array(dAlt['data'][corrName]))
                    # except KeyError:
                    #     # print("Missing correlator %s" % corrName)
                    #     return

            iData  = np.array(iData).sum(axis=0)
            duCorr = genDataset(iData, binSize = binSize)

            tRatioCorr.append(nuCorr/duCorr)
          else:
            tRatioCorr.append(nuCorr)

        ratioCorr = 0.5*tRatioCorr[0][:,0:T+1] + 0.25*tRatioCorr[1][:,0:T+1] + 0.25*np.roll(tRatioCorr[1], -1, axis=1)[:,0:T+1]
      
      
      
      
      
      
      elif ratio.upper()[0:3] == 'RA1' or ratio.upper() == 'ZRA1' or ratio.upper()[0:3] == 'RA1S' or ratio.upper() == 'ZRA1S':
        tRatioCorr = []

        for tSink in tSinks:
          iData  = []
          nuCorr = []
          for j,name in enumerate(nNames[0]):
            corrName = name + 'T' + str(tSink) + hStr + '_RW_' + hss + '_rot_rot' + qStr + lStr + momStr

            try:
              iData.append(dAlt['data'][corrName][()] if mom == '000' else data['data'][corrName][()])
            except KeyError:
              try:
                iData.append(data['data'][corrName][()] if mom == '000' else dAlt['data'][corrName][()])
              except KeyError:
                print("Missing correlator %s" % corrName)
                return

            iData[j] = genDataset(iData[j], binSize = binSize)
          iData  = np.array(iData)[:,:,0:(tSink+1)]
          nuCorr.append(iData)

          iData  = []

          target = nNames[0] if ratio.upper() != 'ZRA1' else nNames[1]
          for j,name in enumerate(target):
            if ratio.upper() != 'ZRA1':     # Reverse
              corrName = name + 'T' + str(tSink) + hStr + '_RW_' + hss + '_rot_rot' + qStr + lStr + momStr

            else:
              corrName = name + 'T' + str(tSink) + lStr + '_RW_' + hss + '_rot_rot' + qStr + hStr + momStr

            try:
              iData.append(dAlt['data'][corrName][()] if mom == '000' else data['data'][corrName][()])
            except KeyError:
              try:
                iData.append(data['data'][corrName][()] if mom == '000' else dAlt['data'][corrName][()])
              except KeyError:
                print("Missing correlator %s" % corrName)
                return

            iData[j] = genDataset(iData[j], binSize = binSize)
          iData  = np.flip(np.array(iData)[:,:,0:(tSink+1)], axis=2) if ratio.upper() != 'ZRA1' else np.array(iData)[:,:,0:(tSink+1)]
          nuCorr.append(iData)

          duCorr = []

          for mesStr,cName in zip([lStr, hStr],dNames):
            iData  = []
            for j,name in enumerate(cName):
              corrName = name + 'T' + str(tSink) + mesStr + '_RW_' + hss + '_rot_rot' + qStr + mesStr + '_p000'

              try:
                iData.append(np.array(dAlt['data'][corrName]))
              except KeyError:
                try:
                  iData.append(np.array(data['data'][corrName]))
                except KeyError:
                  print("Missing correlator %s" % corrName)
                  return

              #iData[j] = genDataset(iData[j], binSize = binSize)
            #duCorr = np.array(iData).mean(axis=0)
            iData  = np.array(iData).mean(axis=0)[:,0:(tSink+1)]
            duCorr.append(genDataset(iData, binSize = binSize))

          #print("Ex", ratio, nuCorr[0].mean(axis=0)[0:2], nuCorr[1].mean(axis=0)[0:2], duCorr[0].mean(axis=0)[0:2], duCorr[1].mean(axis=0)[0:2])
          print((nuCorr[0]*nuCorr[1]).mean(axis=0).shape, (nuCorr[0]*nuCorr[1]).mean(axis=0).mean(axis=0)[0:2])
          tRatioCorr.append((nuCorr[0]*nuCorr[1]).sum(axis=0)/(duCorr[0]*duCorr[1]))

        ratioCorr = 0.5*tRatioCorr[0][:,0:T+1]*np.exp((E0 - m0)*T) + 0.25*tRatioCorr[1][:,0:T+1]*np.exp((E0 - m0)*(T+1)) + 0.25*np.roll(tRatioCorr[1], -1, axis=1)[:,0:T+1]*np.exp((E0 - m0)*(T+1))


      RETURN[hss] = ratioCorr

    return RETURN
    #   if ratioCorr is not None:
    #     colBase  = ratio + '-' + hss
    #     colNames = selectCols(colBase, [0,T+1])

    #     crData = pd.concat([crData, pd.DataFrame(ratioCorr, columns=colNames)], axis=1)

    # return crData




def alex():
    return getThreepHdf5('Coarse-1','RA1','000')
al = alex()


print('==========================')

io = test()

print('==========================')


assert np.array_equal(io['RW'],al['RW'])
assert np.array_equal(io['1S'],al['1S'])


# print(io['RW'][0,:])
# print(al['RW'][0,:])