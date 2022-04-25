# Quantile utilities for processing MERRA/AIRS data 

import numpy
import numpy.ma as ma
import calculate_VPD
import netCDF4
from netCDF4 import Dataset
from numpy import random, linalg
import datetime
import pandas
import os, sys
from scipy import stats
import h5py

def quantile_cloud_locmask(airsdr, mtdr, indr, dtdr, yrlst, mnst, mnfn, hrchc, rgchc, msk):
    # Construct cloud variable quantiles and z-scores, with a possibly irregular location mask

    # Read probs and pressure levels
    rnm = '%s/AIRS_Levels_Quantiles.nc' % (airsdr)
    f = Dataset(rnm,'r')
    plev = f['level'][:]
    prbs = f['probability'][:]
    alts = f['altitude'][:]
    f.close()

    nyr = len(yrlst)
    nprb = prbs.shape[0]

    # Mask, lat, lon
    fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_South_Southeast_US_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[0],hrchc)
    f = Dataset(fnm,'r')
    mask = f[msk][:,:]
    latmet = f['plat'][:]
    lonmet = f['plon'][:]
    f.close()

    mask[mask <= 0] = 0
    lnsq = numpy.arange(lonmet.shape[0])
    ltsq = numpy.arange(latmet.shape[0])

    # Subset a bit
    lnsm = numpy.sum(mask,axis=0)
    print(lnsq.shape)
    print(lnsm.shape)
    print(lnsm) 
    ltsm = numpy.sum(mask,axis=1)
    print(ltsq.shape)
    print(ltsm.shape)
    print(ltsm)

    lnmn = numpy.amin(lnsq[lnsm > 0])
    lnmx = numpy.amax(lnsq[lnsm > 0]) + 1
    ltmn = numpy.amin(ltsq[ltsm > 0])
    ltmx = numpy.amax(ltsq[ltsm > 0]) + 1

    stridx = 'Lon Range: %d, %d\nLat Range: %d, %d \n' % (lnmn,lnmx,ltmn,ltmx)
    print(stridx)

    #latflt = latin.flatten()
    #lonflt = lonin.flatten()
    #mskflt = mask.flatten()
 
    #lcsq = numpy.arange(mskflt.shape[0])
    #lcsb = lcsq[mskflt > 0]

    nx = lnmx - lnmn
    ny = ltmx - ltmn 

    lnrp = numpy.tile(lonmet[lnmn:lnmx],ny)
    ltrp = numpy.repeat(latmet[ltmn:ltmx],nx)
    mskblk = mask[ltmn:ltmx,lnmn:lnmx]
    mskflt = mskblk.flatten()


    tsmp = 0
    for k in range(nyr):
        dyinit = datetime.date(yrlst[k],6,1) 
        dyst = datetime.date(yrlst[k],mnst,1)
        ttst = dyst.timetuple()
        jst = ttst.tm_yday
        if mnfn < 12:
            dyfn = datetime.date(yrlst[k],mnfn+1,1)
            ttfn = dyfn.timetuple()
            jfn = ttfn.tm_yday
        else:
             dyfn = datetime.date(yrlst[k]+1,1,1)
             dy31 = datetime.date(yrlst[k],12,31)
             tt31 = dy31.timetuple()
             jfn = tt31.tm_yday + 1
 
        dystidx = abs((dyst-dyinit).days)
        dyfnidx = abs((dyfn-dyinit).days)

        jdsq = numpy.arange(jst,jfn)
        print(jdsq)
        tmhld = numpy.repeat(jdsq,nx*ny)
        print(tmhld.shape)
        print(numpy.amin(tmhld))
        print(numpy.amax(tmhld))

        stridx = 'Day Range: %d, %d\n' % (dystidx,dyfnidx)
        print(stridx)

        fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_South_Southeast_US_%02dUTC_no_vertical_variation_for_missing_IncludesCloudParams.h5' % (indr,yrlst[k],hrchc)
        f = h5py.File(fnm,'r')
        ctyp1 = f['/ctype'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        ctyp2 = f['/ctype2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprt1 = f['/cprtop'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprt2 = f['/cprtop2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprb1 = f['/cprbot'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprb2 = f['/cprbot2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cfrc1 = f['/cfrac'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cfrc2 = f['/cfrac2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cfrc12 = f['/cfrac12'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cngwt1 = f['/cngwat'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cngwt2 = f['/cngwat2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cttp1 = f['/cstemp'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cttp2 = f['/cstemp2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        f.close()

        mtnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_South_Southeast_US_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[k],hrchc)
        f = Dataset(mtnm,'r')
        psfc = f.variables['spres'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        f.close()

        nt = ctyp1.shape[0]
        mskall = numpy.tile(mskflt,nt)
        msksq = numpy.arange(mskall.shape[0])
        msksb = msksq[mskall > 0]
        mskstr = 'Total Obs: %d, Within Mask: %d \n' % (msksq.shape[0],msksb.shape[0])
        print(mskstr)

        lthld = numpy.tile(ltrp,nt)
        lnhld = numpy.tile(lnrp,nt)

        ctyp1 = ctyp1.flatten()
        ctyp2 = ctyp2.flatten()
        cfrc1 = cfrc1.flatten()
        cfrc2 = cfrc2.flatten()
        cfrc12 = cfrc12.flatten()
        cngwt1 = cngwt1.flatten() 
        cngwt2 = cngwt2.flatten() 
        cttp1 = cttp1.flatten() 
        cttp2 = cttp2.flatten() 
        psfc = psfc.flatten()

        # Number of slabs
        nslbtmp = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16)
        nslbtmp[(ctyp1 > 100) & (ctyp2 > 100)] = 2
        nslbtmp[(ctyp1 > 100) & (ctyp2 < 100)] = 1
   
        if tsmp == 0:
            nslabout = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            nslabout[:] = nslbtmp[msksb]
        else:
            nslabout = numpy.append(nslabout,nslbtmp[msksb]) 

        flsq = numpy.arange(ctyp1.shape[0])

        # For two slabs, slab 1 must have highest cloud bottom pressure
        cprt1 = cprt1.flatten()
        cprt2 = cprt2.flatten()
        cprb1 = cprb1.flatten()
        cprb2 = cprb2.flatten()
        slabswap = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16)
        swpsq = flsq[(nslbtmp == 2) & (cprb1 < cprb2)] 
        slabswap[swpsq] = 1
        print(numpy.mean(slabswap))

        # Cloud Pressure variables
        pbttmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        pbttmp1[nslbtmp >= 1] = cprb1[nslbtmp >= 1]
        pbttmp1[swpsq] = cprb2[swpsq]

        ptptmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        ptptmp1[nslbtmp >= 1] = cprt1[nslbtmp >= 1]
        ptptmp1[swpsq] = cprt2[swpsq]

        pbttmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        pbttmp2[nslbtmp == 2] = cprb2[nslbtmp == 2]
        pbttmp2[swpsq] = cprb1[swpsq]

        ptptmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        ptptmp2[nslbtmp == 2] = cprt2[nslbtmp == 2]
        ptptmp2[swpsq] = cprt1[swpsq]

        # DP Cloud transformation
        dptmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dptmp1[nslbtmp >= 1] = pbttmp1[nslbtmp >= 1] - ptptmp1[nslbtmp >= 1]

        dpslbtmp = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dpslbtmp[nslbtmp == 2] = ptptmp1[nslbtmp == 2] - pbttmp2[nslbtmp == 2]

        dptmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dptmp2[nslbtmp == 2] = pbttmp2[nslbtmp == 2] - ptptmp2[nslbtmp == 2]

        # Adjust negative DPSlab values
        dpnsq = flsq[(nslbtmp == 2) & (dpslbtmp < 0.0) & (dpslbtmp > -1000.0)] 
        dpadj = numpy.zeros((ctyp1.shape[0],)) 
        dpadj[dpnsq] = numpy.absolute(dpslbtmp[dpnsq])
 
        dpslbtmp[dpnsq] = 1.0
        dptmp1[dpnsq] = dptmp1[dpnsq] / 2.0
        dptmp2[dpnsq] = dptmp2[dpnsq] / 2.0

        # Sigma / Logit Adjustments
        zpbtmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdp1tmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdslbtmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdp2tmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        ncldct = 0
        for t in range(psfc.shape[0]):
            if ( (pbttmp1[t] >= 0.0) and (dpslbtmp[t] >= 0.0) ):
                prptmp = numpy.array( [ (psfc[t] - pbttmp1[t]) / psfc[t], \
                                         dptmp1[t] / psfc[t], dpslbtmp[t] / psfc[t], \
                                         dptmp2[t] / psfc[t], 0.0 ] )
                if (prptmp[0] < 0.0):
                    # Adjustment needed
                    prpadj = prptmp[0]
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    prptmp[2] = prptmp[2] + prpadj*prptmp[2]
                    prptmp[3] = prptmp[3] + prpadj*prptmp[3]
                    ncldct = ncldct + 1
                elif (prptmp[0] == 0.0):
                    # Adjustment needed
                    prpadj = -0.01
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    prptmp[2] = prptmp[2] + prpadj*prptmp[2]
                    prptmp[3] = prptmp[3] + prpadj*prptmp[3]
                    ncldct = ncldct + 1
                prptmp[4] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2] - prptmp[3]
                ztmp = calculate_VPD.lgtzs(prptmp)
                zpbtmp[t] = ztmp[0]
                zdp1tmp[t] = ztmp[1]
                zdslbtmp[t] = ztmp[2]
                zdp2tmp[t] = ztmp[3]
            elif ( pbttmp1[t] >= 0.0  ):
                prptmp = numpy.array( [ (psfc[t] - pbttmp1[t]) / psfc[t], \
                                         dptmp1[t] / psfc[t], 0.0 ] )
                if (prptmp[0] < 0.0):
                    # Adjustment needed
                    prpadj = prptmp[0]
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    ncldct = ncldct + 1
                elif (prptmp[0] == 0.0):
                    # Adjustment needed
                    prpadj = -0.01
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    ncldct = ncldct + 1
                prptmp[2] = 1.0 - prptmp[0] - prptmp[1]
                ztmp = calculate_VPD.lgtzs(prptmp)
                zpbtmp[t] = ztmp[0]
                zdp1tmp[t] = ztmp[1]
                zdslbtmp[t] = -9999.0 
                zdp2tmp[t] = -9999.0 
            else:            
                zpbtmp[t] = -9999.0 
                zdp1tmp[t] = -9999.0 
                zdslbtmp[t] = -9999.0 
                zdp2tmp[t] = -9999.0 
        str1 = 'Cloud Bot Pres Below Sfc: %d ' % (ncldct)
        print(str1)

        if tsmp == 0:
            psfcout = numpy.zeros((msksb.shape[0],)) - 9999.0
            psfcout[:] = psfc[msksb]
            prsbot1out = numpy.zeros((msksb.shape[0],)) - 9999.0
            prsbot1out[:] = zpbtmp[msksb]
            dpcld1out = numpy.zeros((msksb.shape[0],)) - 9999.0
            dpcld1out[:] = zdp1tmp[msksb]
            dpslbout = numpy.zeros((msksb.shape[0],)) - 9999.0
            dpslbout[:] = zdslbtmp[msksb]
            dpcld2out = numpy.zeros((msksb.shape[0],)) - 9999.0
            dpcld2out[:] = zdp2tmp[msksb]
        else:
            psfcout = numpy.append(psfcout,psfc[msksb]) 
            prsbot1out = numpy.append(prsbot1out,zpbtmp[msksb])
            dpcld1out = numpy.append(dpcld1out,zdp1tmp[msksb])
            dpslbout = numpy.append(dpslbout,zdslbtmp[msksb])
            dpcld2out = numpy.append(dpcld2out,zdp2tmp[msksb])

        # Slab Types: 101.0 = Liquid, 201.0 = Ice, None else
        # Output: 0 = Liquid, 1 = Ice
        typtmp1 = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16) - 99
        typtmp1[nslbtmp >= 1] = (ctyp1[nslbtmp >= 1] - 1.0) / 100.0 - 1.0
        typtmp1[swpsq] = (ctyp2[swpsq] - 1.0) / 100.0 - 1.0

        typtmp2 = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16) - 99
        typtmp2[nslbtmp == 2] = (ctyp2[nslbtmp == 2] - 1.0) / 100.0 - 1.0 
        typtmp2[swpsq] = (ctyp1[swpsq] - 1.0) / 100.0 - 1.0

        if tsmp == 0:
            slbtyp1out = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            slbtyp1out[:] = typtmp1[msksb]
            slbtyp2out = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            slbtyp2out[:] = typtmp2[msksb]
        else:
            slbtyp1out = numpy.append(slbtyp1out,typtmp1[msksb]) 
            slbtyp2out = numpy.append(slbtyp2out,typtmp2[msksb]) 

        # Cloud Fraction Logit, still account for swapping
        z1tmp = numpy.zeros((cfrc1.shape[0],)) - 9999.0
        z2tmp = numpy.zeros((cfrc1.shape[0],)) - 9999.0
        z12tmp = numpy.zeros((cfrc1.shape[0],)) - 9999.0

        for t in range(z1tmp.shape[0]):
            if ( (cfrc1[t] > 0.0) and (cfrc2[t] > 0.0) and (cfrc12[t] > 0.0) ):
                # Must adjust amounts
                if (slabswap[t] == 0):
                    prptmp = numpy.array( [cfrc1[t]-cfrc12[t], cfrc2[t]-cfrc12[t], cfrc12[t], 0.0] )
                else:
                    prptmp = numpy.array( [cfrc2[t]-cfrc12[t], cfrc1[t]-cfrc12[t], cfrc12[t], 0.0] )
                prptmp[3] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2]
                ztmp = calculate_VPD.lgtzs(prptmp)
                z1tmp[t] = ztmp[0]
                z2tmp[t] = ztmp[1]
                z12tmp[t] = ztmp[2]
            elif ( (cfrc1[t] > 0.0) and (cfrc2[t] > 0.0) ):
                if (slabswap[t] == 0):
                    prptmp = numpy.array( [cfrc1[t], cfrc2[t], 0.0] )
                else:
                    prptmp = numpy.array( [cfrc2[t], cfrc1[t], 0.0] )
                prptmp[2] = 1.0 - prptmp[0] - prptmp[1]
                ztmp = calculate_VPD.lgtzs(prptmp)
                z1tmp[t] = ztmp[0]
                z2tmp[t] = ztmp[1]
                z12tmp[t] = -9999.0
            elif ( cfrc1[t] > 0.0 ):
                prptmp = numpy.array( [cfrc1[t], 1.0 - cfrc1[t] ] )
                ztmp = calculate_VPD.lgtzs(prptmp)
                z1tmp[t] = ztmp[0]
                z2tmp[t] = -9999.0 
                z12tmp[t] = -9999.0
            else:            
                z1tmp[t] = -9999.0
                z2tmp[t] = -9999.0
                z12tmp[t] = -9999.0

        if tsmp == 0:
            cfclgt1out = numpy.zeros((msksb.shape[0],)) - 9999.0
            cfclgt1out[:] = z1tmp[msksb]
            cfclgt2out = numpy.zeros((msksb.shape[0],)) - 9999.0
            cfclgt2out[:] = z2tmp[msksb]
            cfclgt12out = numpy.zeros((msksb.shape[0],)) - 9999.0
            cfclgt12out[:] = z12tmp[msksb]
        else:
            cfclgt1out = numpy.append(cfclgt1out,z1tmp[msksb]) 
            cfclgt2out = numpy.append(cfclgt2out,z2tmp[msksb]) 
            cfclgt12out = numpy.append(cfclgt12out,z12tmp[msksb]) 


        # Cloud Non-Gas Water
        ngwttmp1 = numpy.zeros(cngwt1.shape[0]) - 9999.0
        ngwttmp1[nslbtmp >= 1] = cngwt1[nslbtmp >= 1]
        ngwttmp1[swpsq] = cngwt2[swpsq]

        ngwttmp2 = numpy.zeros(cngwt1.shape[0]) - 9999.0
        ngwttmp2[nslbtmp == 2] = cngwt2[nslbtmp == 2] 
        ngwttmp2[swpsq] = cngwt1[swpsq] 

        if tsmp == 0:
            ngwt1out = numpy.zeros((msksb.shape[0],)) - 9999.0
            ngwt1out[:] = ngwttmp1[msksb]
            ngwt2out = numpy.zeros((msksb.shape[0],)) - 9999.0
            ngwt2out[:] = ngwttmp2[msksb]
        else:
            ngwt1out = numpy.append(ngwt1out,ngwttmp1[msksb]) 
            ngwt2out = numpy.append(ngwt2out,ngwttmp2[msksb]) 

        # Cloud Top Temperature 
        cttptmp1 = numpy.zeros(cttp1.shape[0]) - 9999.0
        cttptmp1[nslbtmp >= 1] = cttp1[nslbtmp >= 1]
        cttptmp1[swpsq] = cttp2[swpsq]

        cttptmp2 = numpy.zeros(cttp1.shape[0]) - 9999.0
        cttptmp2[nslbtmp == 2] = cttp2[nslbtmp == 2] 
        cttptmp2[swpsq] = cttp1[swpsq] 

        if tsmp == 0:
            cttp1out = numpy.zeros((msksb.shape[0],)) - 9999.0
            cttp1out[:] = cttptmp1[msksb]
            cttp2out = numpy.zeros((msksb.shape[0],)) - 9999.0
            cttp2out[:] = cttptmp2[msksb]
        else:
            cttp1out = numpy.append(cttp1out,cttptmp1[msksb]) 
            cttp2out = numpy.append(cttp2out,cttptmp2[msksb]) 

        # Loc/Time
        if tsmp == 0:
            latout = numpy.zeros((msksb.shape[0],)) - 9999.0
            latout[:] = lthld[msksb]
            lonout = numpy.zeros((msksb.shape[0],)) - 9999.0
            lonout[:] = lnhld[msksb]
            yrout = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            yrout[:] = yrlst[k]
            jdyout = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            jdyout[:] = tmhld[msksb]
        else:
            latout = numpy.append(latout,lthld[msksb])
            lonout = numpy.append(lonout,lnhld[msksb])
            yrtmp = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            yrtmp[:] = yrlst[k]
            yrout = numpy.append(yrout,yrtmp)
            jdyout = numpy.append(jdyout,tmhld[msksb])

        tsmp = tsmp + msksb.shape[0]

    # Process quantiles

    nslbqs = calculate_VPD.quantile_msgdat_discrete(nslabout,prbs)
    str1 = '%.2f Number Slab Quantile: %d' % (prbs[53],nslbqs[53])
    print(str1)
    print(nslbqs)

    psfcqs = calculate_VPD.quantile_msgdat(psfcout,prbs)
    str1 = '%.2f Surface Pressure Quantile: %.3f' % (prbs[53],psfcqs[53])
    print(str1)

    prsbt1qs = calculate_VPD.quantile_msgdat(prsbot1out,prbs)
    str1 = '%.2f CldBot1 Pressure Quantile: %.3f' % (prbs[53],prsbt1qs[53])
    print(str1)

    dpcld1qs = calculate_VPD.quantile_msgdat(dpcld1out,prbs)
    str1 = '%.2f DPCloud1 Quantile: %.3f' % (prbs[53],dpcld1qs[53])
    print(str1)

    dpslbqs = calculate_VPD.quantile_msgdat(dpslbout,prbs)
    str1 = '%.2f DPSlab Quantile: %.3f' % (prbs[53],dpslbqs[53])
    print(str1)

    dpcld2qs = calculate_VPD.quantile_msgdat(dpcld2out,prbs)
    str1 = '%.2f DPCloud2 Quantile: %.3f' % (prbs[53],dpcld2qs[53])
    print(str1)

    slb1qs = calculate_VPD.quantile_msgdat_discrete(slbtyp1out,prbs)
    str1 = '%.2f Type1 Quantile: %d' % (prbs[53],slb1qs[53])
    print(str1)

    slb2qs = calculate_VPD.quantile_msgdat_discrete(slbtyp2out,prbs)
    str1 = '%.2f Type2 Quantile: %d' % (prbs[53],slb2qs[53])
    print(str1)

    lgt1qs = calculate_VPD.quantile_msgdat(cfclgt1out,prbs)
    str1 = '%.2f Logit 1 Quantile: %.3f' % (prbs[53],lgt1qs[53])
    print(str1)

    lgt2qs = calculate_VPD.quantile_msgdat(cfclgt2out,prbs)
    str1 = '%.2f Logit 2 Quantile: %.3f' % (prbs[53],lgt2qs[53])
    print(str1)

    lgt12qs = calculate_VPD.quantile_msgdat(cfclgt12out,prbs)
    str1 = '%.2f Logit 1/2 Quantile: %.3f' % (prbs[53],lgt12qs[53])
    print(str1)

    ngwt1qs = calculate_VPD.quantile_msgdat(ngwt1out,prbs)
    str1 = '%.2f NGWater1 Quantile: %.3f' % (prbs[53],ngwt1qs[53])
    print(str1)

    ngwt2qs = calculate_VPD.quantile_msgdat(ngwt2out,prbs)
    str1 = '%.2f NGWater2 Quantile: %.3f' % (prbs[53],ngwt2qs[53])
    print(str1)

    cttp1qs = calculate_VPD.quantile_msgdat(cttp1out,prbs)
    str1 = '%.2f CTTemp1 Quantile: %.3f' % (prbs[53],cttp1qs[53])
    print(str1)

    cttp2qs = calculate_VPD.quantile_msgdat(cttp2out,prbs)
    str1 = '%.2f CTTemp2 Quantile: %.3f' % (prbs[53],cttp2qs[53])
    print(str1)

    # Should be no missing for number of slabs
    print('Slab summary')
    print(numpy.amin(nslabout))
    print(numpy.amax(nslabout))
    print(tsmp)

    # Output Quantiles
    mstr = dyst.strftime('%b')
    qfnm = '%s/%s_US_JJA_%02dUTC_%04d_Cloud_Quantile.nc' % (dtdr,rgchc,hrchc,yrlst[k])
    qout = Dataset(qfnm,'w') 

    dimp = qout.createDimension('probability',nprb)

    varprb = qout.createVariable('probability','f4',['probability'], fill_value = -9999)
    varprb[:] = prbs
    varprb.long_name = 'Probability break points'
    varprb.units = 'none'
    varprb.missing_value = -9999

    varnslb = qout.createVariable('NumberSlab_quantile','i2',['probability'], fill_value = -99)
    varnslb[:] = nslbqs
    varnslb.long_name = 'Number of cloud slabs quantiles'
    varnslb.units = 'Count'
    varnslb.missing_value = -99

    varcbprs = qout.createVariable('CloudBot1Logit_quantile','f4',['probability'], fill_value = -9999)
    varcbprs[:] = prsbt1qs
    varcbprs.long_name = 'Slab 1 cloud bottom pressure logit quantiles'
    varcbprs.units = 'hPa'
    varcbprs.missing_value = -9999

    vardpc1 = qout.createVariable('DPCloud1Logit_quantile','f4',['probability'], fill_value = -9999)
    vardpc1[:] = dpcld1qs
    vardpc1.long_name = 'Slab 1 cloud pressure depth logit quantiles'
    vardpc1.units = 'hPa'
    vardpc1.missing_value = -9999

    vardpslb = qout.createVariable('DPSlabLogit_quantile','f4',['probability'], fill_value = -9999)
    vardpslb[:] = dpslbqs
    vardpslb.long_name = 'Two-slab vertical separation logit quantiles' 
    vardpslb.units = 'hPa'
    vardpslb.missing_value = -9999

    vardpc2 = qout.createVariable('DPCloud2Logit_quantile','f4',['probability'], fill_value = -9999)
    vardpc2[:] = dpcld2qs
    vardpc2.long_name = 'Slab 2 cloud pressure depth logit quantiles'
    vardpc2.units = 'hPa'
    vardpc2.missing_value = -9999

    vartyp1 = qout.createVariable('CType1_quantile','i2',['probability'], fill_value = -99)
    vartyp1[:] = slb1qs
    vartyp1.long_name = 'Slab 1 cloud type quantiles'
    vartyp1.units = 'None'
    vartyp1.missing_value = -99
    vartyp1.comment = 'Cloud slab type: 0=Liquid, 1=Ice'

    vartyp2 = qout.createVariable('CType2_quantile','i2',['probability'], fill_value = -99)
    vartyp2[:] = slb2qs
    vartyp2.long_name = 'Slab 2 cloud type quantiles'
    vartyp2.units = 'None'
    vartyp2.missing_value = -99
    vartyp2.comment = 'Cloud slab type: 0=Liquid, 1=Ice'

    varlgt1 = qout.createVariable('CFrcLogit1_quantile','f4',['probability'], fill_value = -9999)
    varlgt1[:] = lgt1qs
    varlgt1.long_name = 'Slab 1 cloud fraction (cfrac1x) logit quantiles'
    varlgt1.units = 'None'
    varlgt1.missing_value = -9999

    varlgt2 = qout.createVariable('CFrcLogit2_quantile','f4',['probability'], fill_value = -9999)
    varlgt2[:] = lgt2qs
    varlgt2.long_name = 'Slab 2 cloud fraction (cfrac2x) logit quantiles'
    varlgt2.units = 'None'
    varlgt2.missing_value = -9999

    varlgt12 = qout.createVariable('CFrcLogit12_quantile','f4',['probability'], fill_value = -9999)
    varlgt12[:] = lgt12qs
    varlgt12.long_name = 'Slab 1/2 overlap fraction (cfrac12) logit quantiles'
    varlgt12.units = 'None'
    varlgt12.missing_value = -9999

    varngwt1 = qout.createVariable('NGWater1_quantile','f4',['probability'], fill_value = -9999)
    varngwt1[:] = ngwt1qs
    varngwt1.long_name = 'Slab 1 cloud non-gas water quantiles'
    varngwt1.units = 'g m^-2'
    varngwt1.missing_value = -9999

    varngwt2 = qout.createVariable('NGWater2_quantile','f4',['probability'], fill_value = -9999)
    varngwt2[:] = ngwt2qs
    varngwt2.long_name = 'Slab 2 cloud non-gas water quantiles'
    varngwt2.units = 'g m^-2'
    varngwt2.missing_value = -9999

    varcttp1 = qout.createVariable('CTTemp1_quantile','f4',['probability'], fill_value = -9999)
    varcttp1[:] = cttp1qs
    varcttp1.long_name = 'Slab 1 cloud top temperature'
    varcttp1.units = 'K'
    varcttp1.missing_value = -9999

    varcttp2 = qout.createVariable('CTTemp2_quantile','f4',['probability'], fill_value = -9999)
    varcttp2[:] = cttp2qs
    varcttp2.long_name = 'Slab 2 cloud top temperature'
    varcttp2.units = 'K'
    varcttp2.missing_value = -9999

    qout.close()


    # Set up transformations
    znslb = calculate_VPD.std_norm_quantile_from_obs(nslabout, nslbqs, prbs,  msgval=-99)
    zpsfc = calculate_VPD.std_norm_quantile_from_obs(psfcout, psfcqs, prbs, msgval=-9999.)
    zprsbt1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(prsbot1out, prsbt1qs, prbs,  msgval=-9999.)
    zdpcld1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpcld1out, dpcld1qs, prbs,  msgval=-9999.)
    zdpslb = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpslbout, dpslbqs, prbs,  msgval=-9999.)
    zdpcld2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpcld2out, dpcld2qs, prbs,  msgval=-9999.)
    zctyp1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(slbtyp1out, slb1qs, prbs,  msgval=-99)
    zctyp2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(slbtyp2out, slb2qs, prbs,  msgval=-99)
    zlgt1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt1out, lgt1qs, prbs,  msgval=-9999.)
    zlgt2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt2out, lgt2qs, prbs,  msgval=-9999.)
    zlgt12 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt12out, lgt12qs, prbs,  msgval=-9999.)
    zngwt1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(ngwt1out, ngwt1qs, prbs,  msgval=-9999.)
    zngwt2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(ngwt2out, ngwt2qs, prbs,  msgval=-9999.)
    zcttp1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cttp1out, cttp1qs, prbs,  msgval=-9999.)
    zcttp2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cttp2out, cttp2qs, prbs,  msgval=-9999.)


    # Output transformed quantile samples
    zfnm = '%s/%s_US_JJA_%02dUTC_%04d_Cloud_StdGausTrans.nc' % (dtdr,rgchc,hrchc,yrlst[k])
    zout = Dataset(zfnm,'w') 

    dimsmp = zout.createDimension('sample',tsmp)

    varlon = zout.createVariable('Longitude','f4',['sample'])
    varlon[:] = lonout
    varlon.long_name = 'Longitude'
    varlon.units = 'degrees_east'

    varlat = zout.createVariable('Latitude','f4',['sample'])
    varlat[:] = latout
    varlat.long_name = 'Latitude'
    varlat.units = 'degrees_north'

    varjdy = zout.createVariable('JulianDay','i2',['sample'])
    varjdy[:] = jdyout
    varjdy.long_name = 'JulianDay'
    varjdy.units = 'day'

    varyr = zout.createVariable('Year','i2',['sample'])
    varyr[:] = yrout
    varyr.long_name = 'Year'
    varyr.units = 'year'

    varnslb = zout.createVariable('NumberSlab_StdGaus','f4',['sample'], fill_value = -9999)
    varnslb[:] = znslb
    varnslb.long_name = 'Quantile transformed number of cloud slabs'
    varnslb.units = 'None'
    varnslb.missing_value = -9999.

    varcbprs = zout.createVariable('CloudBot1Logit_StdGaus','f4',['sample'], fill_value = -9999)
    varcbprs[:] = zprsbt1
    varcbprs.long_name = 'Quantile transformed slab 1 cloud bottom pressure logit'
    varcbprs.units = 'None'
    varcbprs.missing_value = -9999.

    vardpc1 = zout.createVariable('DPCloud1Logit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpc1[:] = zdpcld1
    vardpc1.long_name = 'Quantile transformed slab 1 cloud pressure depth logit'
    vardpc1.units = 'None'
    vardpc1.missing_value = -9999.

    vardpslb = zout.createVariable('DPSlabLogit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpslb[:] = zdpslb
    vardpslb.long_name = 'Quantile transformed two-slab vertical separation logit'
    vardpslb.units = 'None'
    vardpslb.missing_value = -9999.

    vardpc2 = zout.createVariable('DPCloud2Logit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpc2[:] = zdpcld2
    vardpc2.long_name = 'Quantile transformed slab 2 cloud pressure depth logit'
    vardpc2.units = 'None'
    vardpc2.missing_value = -9999.

    vartyp1 = zout.createVariable('CType1_StdGaus','f4',['sample'], fill_value = -9999)
    vartyp1[:] = zctyp1
    vartyp1.long_name = 'Quantile transformed slab 1 cloud type logit'
    vartyp1.units = 'None'
    vartyp1.missing_value = -9999.

    vartyp2 = zout.createVariable('CType2_StdGaus','f4',['sample'], fill_value = -9999)
    vartyp2[:] = zctyp2
    vartyp2.long_name = 'Quantile transformed slab 2 cloud type'
    vartyp2.units = 'None'
    vartyp2.missing_value = -9999.

    varlgt1 = zout.createVariable('CFrcLogit1_StdGaus','f4',['sample'], fill_value = -9999)
    varlgt1[:] = zlgt1
    varlgt1.long_name = 'Quantile transformed slab 1 cloud fraction logit'
    varlgt1.units = 'None'
    varlgt1.missing_value = -9999.

    varlgt2 = zout.createVariable('CFrcLogit2_StdGaus','f4',['sample'], fill_value = -9999)
    varlgt2[:] = zlgt2
    varlgt2.long_name = 'Quantile transformed slab 2 cloud fraction logit'
    varlgt2.units = 'None'
    varlgt2.missing_value = -9999.

    varlgt12 = zout.createVariable('CFrcLogit12_StdGaus','f4',['sample'], fill_value = -9999)
    varlgt12[:] = zlgt12
    varlgt12.long_name = 'Quantile transformed slab 1/2 overlap fraction logit'
    varlgt12.units = 'None'
    varlgt12.missing_value = -9999.

    varngwt1 = zout.createVariable('NGWater1_StdGaus','f4',['sample'], fill_value = -9999)
    varngwt1[:] = zngwt1
    varngwt1.long_name = 'Quantile transformed slab 1 non-gas water'
    varngwt1.units = 'None'
    varngwt1.missing_value = -9999.

    varngwt2 = zout.createVariable('NGWater2_StdGaus','f4',['sample'], fill_value = -9999)
    varngwt2[:] = zngwt2
    varngwt2.long_name = 'Quantile transformed slab 2 non-gas water'
    varngwt2.units = 'None'
    varngwt2.missing_value = -9999.

    varcttp1 = zout.createVariable('CTTemp1_StdGaus','f4',['sample'], fill_value = -9999)
    varcttp1[:] = zcttp1
    varcttp1.long_name = 'Quantile transformed slab 1 cloud top temperature'
    varcttp1.units = 'None'
    varcttp1.missing_value = -9999.

    varcttp2 = zout.createVariable('CTTemp2_StdGaus','f4',['sample'], fill_value = -9999)
    varcttp2[:] = zcttp2
    varcttp2.long_name = 'Quantile transformed slab 2 cloud top temperature'
    varcttp2.units = 'None'
    varcttp2.missing_value = -9999.

    zout.close()

    return

# Temp/RH Quantiles
def quantile_profile_locmask(airsdr, mtdr, indr, dtdr, yrlst, mnst, mnfn, hrchc, rgchc, msk):
    # Construct profile/sfc variable quantiles and z-scores, with a possibly irregular location mask

    # Read probs and pressure levels
    rnm = '%s/AIRS_Levels_Quantiles.nc' % (airsdr)
    f = Dataset(rnm,'r')
    plev = f['level'][:]
    prbs = f['probability'][:]
    alts = f['altitude'][:]
    f.close()

    nyr = len(yrlst)
    nprb = prbs.shape[0]
    nzout = 101

    tmpqout = numpy.zeros((nzout,nprb)) - 9999.
    rhqout = numpy.zeros((nzout,nprb)) - 9999.
    sftmpqs = numpy.zeros((nprb,)) - 9999.
    sfaltqs = numpy.zeros((nprb,)) - 9999.
    psfcqs = numpy.zeros((nprb,)) - 9999.
    altmed = numpy.zeros((nzout,)) - 9999.

    # Mask, lat, lon
    fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_South_Southeast_US_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[0],hrchc)
    f = Dataset(fnm,'r')
    mask = f[msk][:,:]
    latmet = f['plat'][:]
    lonmet = f['plon'][:]
    f.close()

    mask[mask <= 0] = 0
    lnsq = numpy.arange(lonmet.shape[0])
    ltsq = numpy.arange(latmet.shape[0])

    # Subset a bit
    lnsm = numpy.sum(mask,axis=0)
    print(lnsq.shape)
    print(lnsm.shape)
    print(lnsm) 
    ltsm = numpy.sum(mask,axis=1)
    print(ltsq.shape)
    print(ltsm.shape)
    print(ltsm)

    lnmn = numpy.amin(lnsq[lnsm > 0])
    lnmx = numpy.amax(lnsq[lnsm > 0]) + 1
    ltmn = numpy.amin(ltsq[ltsm > 0])
    ltmx = numpy.amax(ltsq[ltsm > 0]) + 1

    stridx = 'Lon Range: %d, %d\nLat Range: %d, %d \n' % (lnmn,lnmx,ltmn,ltmx)
    print(stridx)

    nx = lnmx - lnmn
    ny = ltmx - ltmn 

    lnrp = numpy.tile(lonmet[lnmn:lnmx],ny)
    ltrp = numpy.repeat(latmet[ltmn:ltmx],nx)
    mskblk = mask[ltmn:ltmx,lnmn:lnmx]
    mskflt = mskblk.flatten()

    tsmp = 0
    for k in range(nyr):
        dyinit = datetime.date(yrlst[k],6,1) 
        dyst = datetime.date(yrlst[k],mnst,1)
        ttst = dyst.timetuple()
        jst = ttst.tm_yday
        if mnfn < 12:
            dyfn = datetime.date(yrlst[k],mnfn+1,1)
            ttfn = dyfn.timetuple()
            jfn = ttfn.tm_yday
        else:
             dyfn = datetime.date(yrlst[k]+1,1,1)
             dy31 = datetime.date(yrlst[k],12,31)
             tt31 = dy31.timetuple()
             jfn = tt31.tm_yday + 1
 
        dystidx = abs((dyst-dyinit).days)
        dyfnidx = abs((dyfn-dyinit).days)

        jdsq = numpy.arange(jst,jfn)
        tmhld = numpy.repeat(jdsq,nx*ny)

        stridx = 'Day Range: %d, %d\n' % (dystidx,dyfnidx)
        print(stridx)

        mtnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_South_Southeast_US_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[k],hrchc)
        f = h5py.File(mtnm,'r')
        stparr = f['/stemp'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        psfarr = f['/spres'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        salarr = f['/salti'][ltmn:ltmx,lnmn:lnmx]
        f.close()

        nt = psfarr.shape[0]
        msksq1 = numpy.arange(mskflt.shape[0])
        msksb1 = msksq1[mskflt > 0]
        mskall = numpy.tile(mskflt,nt)
        msksq = numpy.arange(mskall.shape[0])
        msksb = msksq[mskall > 0]
        mskstr = 'Total Obs: %d, Within Mask: %d \n' % (msksq.shape[0],msksb.shape[0])
        print(mskstr)

        lthld = numpy.tile(ltrp,nt)
        lnhld = numpy.tile(lnrp,nt)

        stparr = stparr.flatten()
        psfarr = psfarr.flatten()
        salarr = salarr.flatten()
  
        if tsmp == 0:
            sftmpout = numpy.zeros((msksb.shape[0],)) - 9999.0
            sftmpout[:] = stparr[msksb]
            psfcout = numpy.zeros((msksb.shape[0],)) - 9999.0
            psfcout[:] = psfarr[msksb]
            sfaltout = numpy.zeros((msksb.shape[0],)) - 9999.0 
            sfaltout[:] = numpy.tile(salarr[msksb1],nt)
        else:
            sftmpout = numpy.append(sftmpout,stparr[msksb])
            psfcout = numpy.append(psfcout,psfarr[msksb])
            sfaltout = numpy.append(sfaltout,numpy.tile(salarr[msksb1],nt))

        # Loc/Time
        if tsmp == 0:
            latout = numpy.zeros((msksb.shape[0],)) - 9999.0
            latout[:] = lthld[msksb]
            lonout = numpy.zeros((msksb.shape[0],)) - 9999.0
            lonout[:] = lnhld[msksb]
            yrout = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            yrout[:] = yrlst[k]
            jdyout = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            jdyout[:] = tmhld[msksb]
        else:
            latout = numpy.append(latout,lthld[msksb])
            lonout = numpy.append(lonout,lnhld[msksb])
            yrtmp = numpy.zeros((msksb.shape[0],),dtype=numpy.int16)
            yrtmp[:] = yrlst[k]
            yrout = numpy.append(yrout,yrtmp)
            jdyout = numpy.append(jdyout,tmhld[msksb])

        tsmp = tsmp + msksb.shape[0]

    # Vertical profiles
    tmpmerout = numpy.zeros((tsmp,nzout)) - 9999.
    h2omerout = numpy.zeros((tsmp,nzout)) - 9999.
    altout = numpy.zeros((tsmp,nzout)) - 9999.
    sidx = 0

    for k in range(nyr):
        dyinit = datetime.date(yrlst[k],6,1) 
        dyst = datetime.date(yrlst[k],mnst,1)
        ttst = dyst.timetuple()
        jst = ttst.tm_yday
        if mnfn < 12:
            dyfn = datetime.date(yrlst[k],mnfn+1,1)
            ttfn = dyfn.timetuple()
            jfn = ttfn.tm_yday
        else:
             dyfn = datetime.date(yrlst[k]+1,1,1)
             dy31 = datetime.date(yrlst[k],12,31)
             tt31 = dy31.timetuple()
             jfn = tt31.tm_yday + 1
 
        dystidx = abs((dyst-dyinit).days)
        dyfnidx = abs((dyfn-dyinit).days)

        jdsq = numpy.arange(jst,jfn)
        tmhld = numpy.repeat(jdsq,nx*ny)

        stridx = 'Day Range: %d, %d\n' % (dystidx,dyfnidx)
        print(stridx)

        mtnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_South_Southeast_US_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[k],hrchc)
        f = h5py.File(mtnm,'r')
        tmparr = f['/ptemp'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        h2oarr = f['/rh'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        altarr = f['/palts'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        f.close()

        nt = tmparr.shape[0]
        mskall = numpy.tile(mskflt,nt)
        msksq = numpy.arange(mskall.shape[0])
        msksb = msksq[mskall > 0]

        lthld = numpy.tile(ltrp,nt)
        lnhld = numpy.tile(lnrp,nt)

        fidx = sidx + msksb.shape[0]

        for j in range(nzout):
            tmpvec = tmparr[:,j,:,:].flatten()
            tmpvec[tmpvec > 1e30] = -9999.
            tmpmerout[sidx:fidx,j] = tmpvec[msksb]

            altvec = altarr[:,j,:,:].flatten()
            altout[sidx:fidx,j] = altvec[msksb]

            h2ovec = h2oarr[:,j,:,:].flatten()
            h2ovec[h2ovec > 1e30] = -9999.
            h2omerout[sidx:fidx,j] = h2ovec[msksb] 

        sidx = sidx + msksb.shape[0]

    # Quantiles
    ztmpout = numpy.zeros((tsmp,nzout)) - 9999.
    zrhout = numpy.zeros((tsmp,nzout)) - 9999.
    zsftmpout = numpy.zeros((tsmp,)) - 9999.
    zsfaltout = numpy.zeros((tsmp,)) - 9999.
    zpsfcout = numpy.zeros((tsmp,)) - 9999.

    for j in range(nzout):
        tmptmp = calculate_VPD.quantile_msgdat(tmpmerout[:,j],prbs)
        tmpqout[j,:] = tmptmp[:]
        str1 = 'Plev %.2f, %.2f Temp Quantile: %.3f' % (plev[j],prbs[103],tmptmp[103])
        print(str1)

        # Transform
        ztmp = calculate_VPD.std_norm_quantile_from_obs(tmpmerout[:,j], tmptmp, prbs,  msgval=-9999.)
        ztmpout[:,j] = ztmp[:]

        alttmp = calculate_VPD.quantile_msgdat(altout[:,j],prbs)
        altmed[j] = alttmp[103]
        str1 = 'Plev %.2f, %.2f Alt Quantile: %.3f' % (plev[j],prbs[103],alttmp[103])
        print(str1)

        # Adjust RH over 100
        rhadj = h2omerout[:,j]
        rhadj[rhadj > 1.0] = 1.0
        rhqtmp = calculate_VPD.quantile_msgdat(rhadj,prbs)
        rhqout[j,:] = rhqtmp[:]
        str1 = 'Plev %.2f, %.2f RH Quantile: %.4f' % (plev[j],prbs[103],rhqtmp[103])
        print(str1)

        zrh = calculate_VPD.std_norm_quantile_from_obs(rhadj, rhqtmp, prbs,  msgval=-9999.)
        zrhout[:,j] = zrh[:]

    psfcqs = calculate_VPD.quantile_msgdat(psfcout,prbs)
    str1 = '%.2f PSfc Quantile: %.2f' % (prbs[103],psfcqs[103])
    print(str1)
    zpsfcout = calculate_VPD.std_norm_quantile_from_obs(psfcout, psfcqs, prbs, msgval=-9999.) 

    sftpqs = calculate_VPD.quantile_msgdat(sftmpout,prbs)
    str1 = '%.2f SfcTmp Quantile: %.2f' % (prbs[103],sftpqs[103])
    print(str1)
    zsftmpout = calculate_VPD.std_norm_quantile_from_obs(sftmpout, sftpqs, prbs, msgval=-9999.) 

    sfalqs = calculate_VPD.quantile_msgdat(sfaltout,prbs)
    str1 = '%.2f SfcAlt Quantile: %.2f' % (prbs[103],sfalqs[103])
    print(str1)
    zsfaltout = calculate_VPD.std_norm_quantile_from_obs(sfaltout, sfalqs, prbs, msgval=-9999.) 

    # Output Quantiles
    mstr = dyst.strftime('%b')
    qfnm = '%s/%s_US_JJA_%02dUTC_%04d_TempRHSfc_Quantile.nc' % (dtdr,rgchc,hrchc,yrlst[k])
    qout = Dataset(qfnm,'w') 

    dimz = qout.createDimension('level',nzout)
    dimp = qout.createDimension('probability',nprb)

    varlvl = qout.createVariable('level','f4',['level'], fill_value = -9999)
    varlvl[:] = plev
    varlvl.long_name = 'AIRS/SARTA pressure levels'
    varlvl.units = 'hPa'
    varlvl.missing_value = -9999

    varprb = qout.createVariable('probability','f4',['probability'], fill_value = -9999)
    varprb[:] = prbs
    varprb.long_name = 'Probability break points'
    varprb.units = 'none'
    varprb.missing_value = -9999

    # Altitude grid
    varalt = qout.createVariable('Altitude_median', 'f4', ['level'], fill_value = -9999)
    varalt[:] = altmed
    varalt.long_name = 'Altitude median value'
    varalt.units = 'm'
    varalt.missing_value = -9999

    vartmp = qout.createVariable('Temperature_quantile', 'f4', ['level','probability'], fill_value = -9999)
    vartmp[:] = tmpqout
    vartmp.long_name = 'Temperature quantiles'
    vartmp.units = 'K'
    vartmp.missing_value = -9999.

    varrh = qout.createVariable('RH_quantile', 'f4', ['level','probability'], fill_value = -9999)
    varrh[:] = rhqout
    varrh.long_name = 'Relative humidity quantiles'
    varrh.units = 'Unitless'
    varrh.missing_value = -9999.

    varstmp = qout.createVariable('SfcTemp_quantile', 'f4', ['probability'], fill_value = -9999)
    varstmp[:] = sftpqs
    varstmp.long_name = 'Surface temperature quantiles'
    varstmp.units = 'K'
    varstmp.missing_value = -9999.

    varpsfc = qout.createVariable('SfcPres_quantile', 'f4', ['probability'], fill_value = -9999)
    varpsfc[:] = psfcqs
    varpsfc.long_name = 'Surface pressure quantiles'
    varpsfc.units = 'hPa'
    varpsfc.missing_value = -9999.

    varsalt = qout.createVariable('SfcAlt_quantile', 'f4', ['probability'], fill_value = -9999)
    varsalt[:] = sfalqs
    varsalt.long_name = 'Surface altitude quantiles'
    varsalt.units = 'm'
    varsalt.missing_value = -9999.

    qout.close()


    # Output transformed quantile samples
    zfnm = '%s/%s_US_JJA_%02dUTC_%04d_TempRHSfc_StdGausTrans.nc' % (dtdr,rgchc,hrchc,yrlst[k])
    zout = Dataset(zfnm,'w') 
    dimz = zout.createDimension('level',nzout)
    dimsmp = zout.createDimension('sample',tsmp)

    varlvl = zout.createVariable('level','f4',['level'], fill_value = -9999)
    varlvl[:] = plev
    varlvl.long_name = 'AIRS/SARTA pressure levels'
    varlvl.units = 'hPa'
    varlvl.missing_value = -9999

    varlon = zout.createVariable('Longitude','f4',['sample'])
    varlon[:] = lonout
    varlon.long_name = 'Longitude'
    varlon.units = 'degrees_east'

    varlat = zout.createVariable('Latitude','f4',['sample'])
    varlat[:] = latout
    varlat.long_name = 'Latitude'
    varlat.units = 'degrees_north'

    varjdy = zout.createVariable('JulianDay','i2',['sample'])
    varjdy[:] = jdyout
    varjdy.long_name = 'JulianDay'
    varjdy.units = 'day'

    varyr = zout.createVariable('Year','i2',['sample'])
    varyr[:] = yrout
    varyr.long_name = 'Year'
    varyr.units = 'year'

    varsrt3 = zout.createVariable('Temperature_StdGaus', 'f4', ['sample','level'], fill_value = -9999)
    varsrt3[:] = ztmpout
    varsrt3.long_name = 'Quantile transformed temperature'
    varsrt3.units = 'None'
    varsrt3.missing_value = -9999.

    varsrt4 = zout.createVariable('RH_StdGaus', 'f4', ['sample','level'], fill_value = -9999)
    varsrt4[:] = zrhout
    varsrt4.long_name = 'Quantile transformed relative humidity'
    varsrt4.units = 'None'
    varsrt4.missing_value = -9999.

    varsrts1 = zout.createVariable('SfcTemp_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts1[:] = zsftmpout
    varsrts1.long_name = 'Quantile transformed surface temperature'
    varsrts1.units = 'None'
    varsrts1.missing_value = -9999.

    varsrts2 = zout.createVariable('SfcPres_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts2[:] = zpsfcout
    varsrts2.long_name = 'Quantile transformed surface pressure'
    varsrts2.units = 'None'
    varsrts2.missing_value = -9999.

    varsrts3 = zout.createVariable('SfcAlt_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts3[:] = zsfaltout
    varsrts3.long_name = 'Quantile transformed surface pressure'
    varsrts3.units = 'None'
    varsrts3.missing_value = -9999.

    zout.close()

    return

def expt_near_sfc_summary(inpdr, outdr, expfl, qclrfl, outfnm):
    # Produce experiment near-surface summaries  
    # inpdr:   Name of input directory
    # outdr:   Name of output directory
    # expfl:   Name of file with experiment results
    # qclrfl:  Input quantile file
    # outfnm:  Ouptut file name

    nzairs = 100
    nzsrt = 101

    # Read simulation results 
    f = h5py.File(expfl,'r')
    tmprtr = f['airs_ptemp'][:,:]
    h2ortr = f['airs_h2o'][:,:]
    tqflg = f['airs_ptemp_qc'][:,:]
    hqflg = f['airs_h2o_qc'][:,:]
    tmpsrt = f['ptemp'][:,1:nzsrt]
    h2osrt = f['gas_1'][:,1:nzsrt]
    psfc = f['spres'][:]
    lvs = f['level'][1:nzsrt]
    f.close()

    nszout = tmprtr.shape[0]
    tqflg = tqflg.astype(numpy.int16)
    hqflg = hqflg.astype(numpy.int16)


    # Altitude info
    qin = Dataset(qclrfl,'r')
    alts = qin['Altitude_median'][:]
    qin.close()

    alth2o = numpy.zeros((nszout,nzsrt))
    alth2o[:,nzsrt-4] = alts[nzsrt-4]
    curdlt = 0.0
    for j in range(nzsrt-5,-1,-1):
        #str1 = 'Level %d: %.4f' % (j,curdlt)
        #print(str1)
        if (alts[j] > alts[j+1]):
            curdlt = alts[j] - alts[j+1]
            alth2o[:,j] = alts[j]
        else:
            alth2o[:,j] = alts[j+1] + curdlt * 2.0
            curdlt = curdlt * 2.0
    alth2o[:,97] = 0.0

    tsfcsrt = calculate_VPD.near_sfc_temp(tmpsrt, lvs, psfc, passqual = False, qual = None)
    print(tsfcsrt[0:10])
    tsfcrtr, tqflgsfc = calculate_VPD.near_sfc_temp(tmprtr, lvs, psfc, passqual = True, qual = tqflg)
    print(tsfcrtr[0:10])
    print(tqflgsfc[0:10])

    qvsrt, rhsrt, vpdsrt = calculate_VPD.calculate_QV_and_VPD(h2osrt,tmpsrt,lvs,alth2o[:,1:nzsrt]) 
    qvrtr, rhrtr, vpdrtr = calculate_VPD.calculate_QV_and_VPD(h2ortr,tmprtr,lvs,alth2o[:,1:nzsrt]) 

    qsfsrt, rhsfsrt = calculate_VPD.near_sfc_qv_rh(qvsrt, tsfcsrt, lvs, psfc, passqual = False, qual = None)
    qsfrtr, rhsfrtr, qflgsfc = calculate_VPD.near_sfc_qv_rh(qvrtr, tsfcrtr, lvs, psfc, passqual = True, qual = hqflg)

    print(tqflgsfc.dtype)
    print(qflgsfc.dtype)

    # Output: Sfc Temp and qflg, SfC QV, RH and qflg
    fldbl = numpy.array([-9999.],dtype=numpy.float64)
    flflt = numpy.array([-9999.],dtype=numpy.float32)
    flshrt = numpy.array([-99],dtype=numpy.int16)
    #outfnm = '%s/MAGIC_%s_%s_%02dUTC_SR%02d_Sfc_UQ_Output.h5' % (outdr,rgchc,mnchc,hrchc,scnrw)
    f = h5py.File(outfnm,'w')
    dft1 = f.create_dataset('TSfcAir_True',data=tsfcsrt)
    dft1.attrs['missing_value'] = fldbl 
    dft1.attrs['_FillValue'] = fldbl
    dft2 = f.create_dataset('TSfcAir_Retrieved',data=tsfcrtr)
    dft2.attrs['missing_value'] = fldbl 
    dft2.attrs['_FillValue'] = fldbl
    dft3 = f.create_dataset('TSfcAir_QC',data=tqflgsfc)

    dfq1 = f.create_dataset('QVSfcAir_True',data=qsfsrt)
    dfq1.attrs['missing_value'] = fldbl 
    dfq1.attrs['_FillValue'] = fldbl
    dfq2 = f.create_dataset('QVSfcAir_Retrieved',data=qsfrtr)
    dfq2.attrs['missing_value'] = fldbl 
    dfq2.attrs['_FillValue'] = fldbl
    dfq3 = f.create_dataset('RHSfcAir_True',data=rhsfsrt)
    dfq3.attrs['missing_value'] = fldbl 
    dfq3.attrs['_FillValue'] = fldbl
    dfq4 = f.create_dataset('RHSfcAir_Retrieved',data=rhsfrtr)
    dfq4.attrs['missing_value'] = fldbl 
    dfq4.attrs['_FillValue'] = fldbl
    dfq5 = f.create_dataset('RHSfcAir_QC',data=qflgsfc)

    dfp1 = f.create_dataset('SfcPres',data=psfc)
    dfp1.attrs['missing_value'] = fldbl 
    dfp1.attrs['_FillValue'] = fldbl

    f.close()
    return

def quantile_cfrac_locmask_conus(rfdr, mtdr, csdr, airdr, dtdr, yrlst, mnst, mnfn, hrchc, rgchc, mskvr, mskvl):
    # Construct cloud variable quantiles and z-scores, with a possibly irregular location mask
    # rfdr:    Directory for reference data (Levels/Quantiles)
    # mtdr:    Directory for MERRA data
    # csdr:    Directory for cloud slab data
    # airdr:   Directory for AIRS cloud fraction
    # dtdr:    Output directory
    # yrlst:   List of years to process
    # mnst:    Starting Month
    # mnfn:    Ending Month 
    # hrchc:   Template Hour Choice
    # rgchc:   Template Region Choice
    # mskvr:   Name of region mask variable
    # mskvl:   Value of region mask for Region Choice

    # Read probs and pressure levels
    rnm = '%s/AIRS_Levels_Quantiles.nc' % (rfdr)
    f = Dataset(rnm,'r')
    plev = f['level'][:]
    prbs = f['probability'][:]
    alts = f['altitude'][:]
    f.close()

    nyr = len(yrlst)
    nprb = prbs.shape[0]

    # RN generator
    sdchc = 542354 + yrlst[0] + hrchc
    random.seed(sdchc)

    # Mask, lat, lon
    fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_CONUS_with_NCA_regions_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[0],hrchc)
    f = Dataset(fnm,'r')
    mask = f.variables[mskvr][:,:]
    latmet = f.variables['plat'][:]
    lonmet = f.variables['plon'][:]
    tminf = f.variables['time'][:]
    tmunit = f.variables['time'].units[:]
    f.close()

    mskind = numpy.zeros((mask.shape),dtype=mask.dtype)
    print(mskvl)
    mskind[mask == mskvl] = 1
    lnsq = numpy.arange(lonmet.shape[0])
    ltsq = numpy.arange(latmet.shape[0])

    # Subset a bit
    lnsm = numpy.sum(mskind,axis=0)
    #print(lnsq.shape)
    #print(lnsm.shape)
    #print(lnsm) 
    ltsm = numpy.sum(mskind,axis=1)
    #print(ltsq.shape)
    #print(ltsm.shape)
    #print(ltsm)

    lnmn = numpy.amin(lnsq[lnsm > 0])
    lnmx = numpy.amax(lnsq[lnsm > 0]) + 1
    ltmn = numpy.amin(ltsq[ltsm > 0])
    ltmx = numpy.amax(ltsq[ltsm > 0]) + 1

    stridx = 'Lon Range: %d, %d\nLat Range: %d, %d \n' % (lnmn,lnmx,ltmn,ltmx)
    print(stridx)

    nx = lnmx - lnmn
    ny = ltmx - ltmn 

    lnrp = numpy.tile(lonmet[lnmn:lnmx],ny)
    ltrp = numpy.repeat(latmet[ltmn:ltmx],nx)
    mskblk = mskind[ltmn:ltmx,lnmn:lnmx]
    mskflt = mskblk.flatten()


    tsmp = 0
    for k in range(nyr):
        fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_CONUS_with_NCA_regions_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[k],hrchc)
        f = Dataset(fnm,'r')
        tminf = f.variables['time'][:]
        tmunit = f.variables['time'].units[:]
        f.close()

        tmunit = tmunit.replace("days since ","")
        dybs = datetime.datetime.strptime(tmunit,"%Y-%m-%d %H:%M:%S")
        print(dybs)
        dy0 = dybs + datetime.timedelta(days=tminf[0]) 
        dyinit = datetime.date(dy0.year,dy0.month,dy0.day)
        print(dyinit)
 
        dyst = datetime.date(yrlst[k],mnst,1)
        ttst = dyst.timetuple()
        jst = ttst.tm_yday
        if mnfn < 12:
            dyfn = datetime.date(yrlst[k],mnfn+1,1)
            ttfn = dyfn.timetuple()
            jfn = ttfn.tm_yday
        else:
             dyfn = datetime.date(yrlst[k]+1,1,1)
             dy31 = datetime.date(yrlst[k],12,31)
             tt31 = dy31.timetuple()
             jfn = tt31.tm_yday + 1
 
        dystidx = abs((dyst-dyinit).days)
        dyfnidx = abs((dyfn-dyinit).days)

        jdsq = numpy.arange(jst,jfn)
        print(jdsq)
        tmhld = numpy.repeat(jdsq,nx*ny)
        #print(tmhld.shape)
        #print(numpy.amin(tmhld))
        #print(numpy.amax(tmhld))

        stridx = 'Day Range: %d, %d\n' % (dystidx,dyfnidx)
        print(stridx)

        fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_CONUS_with_NCA_regions_%02dUTC_no_vertical_variation_for_missing_IncludesCloudParams.h5' % (csdr,yrlst[k],hrchc)
        f = h5py.File(fnm,'r')
        tms = f['/time'][:,dystidx:dyfnidx]
        ctyp1 = f['/ctype'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        ctyp2 = f['/ctype2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprt1 = f['/cprtop'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprt2 = f['/cprtop2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprb1 = f['/cprbot'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cprb2 = f['/cprbot2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cfrc1 = f['/cfrac'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cfrc2 = f['/cfrac2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cfrc12 = f['/cfrac12'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cngwt1 = f['/cngwat'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cngwt2 = f['/cngwat2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cttp1 = f['/cstemp'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        cttp2 = f['/cstemp2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        f.close()

        tmflt = tms.flatten()
        nt = tmflt.shape[0]
        lnhld = numpy.tile(lnrp,nt)
        lthld = numpy.tile(ltrp,nt)

        mtnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_CONUS_with_NCA_regions_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[k],hrchc)
        f = Dataset(mtnm,'r')
        psfc = f.variables['spres'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        f.close()

        nt = ctyp1.shape[0]
        mskall = numpy.tile(mskflt,nt)
        msksq = numpy.arange(mskall.shape[0])
        msksb = msksq[mskall > 0]
        mskstr = 'Total Obs: %d, Within Mask: %d \n' % (msksq.shape[0],msksb.shape[0])
        print(mskstr)

#        lthld = numpy.tile(ltrp,nt)
#        lnhld = numpy.tile(lnrp,nt)

        nslbtmp = numpy.zeros((ctyp1.shape),dtype=numpy.int16)
        nslbtmp[(ctyp1 > 100) & (ctyp2 > 100)] = 2
        nslbtmp[(ctyp1 > 100) & (ctyp2 < 100)] = 1
   
        # AIRS clouds
        anm = '%s/CONUS_AIRS_CldFrc_Match_JJA_%d_%02d_UTC.nc' % (airdr,yrlst[k],hrchc)
        f = Dataset(anm,'r')
        arsfrc1 = f.variables['AIRS_CldFrac_1'][:,dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        arsfrc2 = f.variables['AIRS_CldFrac_2'][:,dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        f.close()

        # Sum
        frctot = arsfrc1 + arsfrc2

        # Construct Clr/PC/Ovc indicator for AIRS total cloud frac
        totclr = numpy.zeros(frctot.shape,dtype=numpy.int16)
        totclr[frctot == 0.0] = -1
        totclr[frctot == 1.0] = 1
        totclr = ma.masked_array(totclr, mask = frctot.mask)

        frc0 = frctot[0,:,:,:]
        frc0 = frc0.flatten()
        frcsq = numpy.arange(tmhld.shape[0])
        # Subset by AIRS matchup and location masks
        frcsb = frcsq[(numpy.logical_not(frc0.mask)) & (mskall > 0)]

        nairs = frcsb.shape[0]
        print(tmhld.shape)
        print(frcsb.shape)

        ctyp1 = ctyp1.flatten()
        ctyp2 = ctyp2.flatten()
        nslbtmp = nslbtmp.flatten()
        cngwt1 = cngwt1.flatten() 
        cngwt2 = cngwt2.flatten() 
        cttp1 = cttp1.flatten() 
        cttp2 = cttp2.flatten() 
        psfc = psfc.flatten()

        # Number of slabs
        if tsmp == 0:
            nslabout = numpy.zeros((nairs,),dtype=numpy.int16)
            nslabout[:] = nslbtmp[frcsb]
        else:
            nslabout = numpy.append(nslabout,nslbtmp[frcsb]) 

        # For two slabs, slab 1 must have highest cloud bottom pressure
        cprt1 = cprt1.flatten()
        cprt2 = cprt2.flatten()
        cprb1 = cprb1.flatten()
        cprb2 = cprb2.flatten()
        slabswap = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16)
        swpsq = frcsq[(nslbtmp == 2) & (cprb1 < cprb2)] 
        slabswap[swpsq] = 1

        # Cloud Pressure variables
        pbttmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        pbttmp1[nslbtmp >= 1] = cprb1[nslbtmp >= 1]
        pbttmp1[swpsq] = cprb2[swpsq]

        ptptmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        ptptmp1[nslbtmp >= 1] = cprt1[nslbtmp >= 1]
        ptptmp1[swpsq] = cprt2[swpsq]

        pbttmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        pbttmp2[nslbtmp == 2] = cprb2[nslbtmp == 2]
        pbttmp2[swpsq] = cprb1[swpsq]

        ptptmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        ptptmp2[nslbtmp == 2] = cprt2[nslbtmp == 2]
        ptptmp2[swpsq] = cprt1[swpsq]

        # DP Cloud transformation
        dptmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dptmp1[nslbtmp >= 1] = pbttmp1[nslbtmp >= 1] - ptptmp1[nslbtmp >= 1]

        dpslbtmp = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dpslbtmp[nslbtmp == 2] = ptptmp1[nslbtmp == 2] - pbttmp2[nslbtmp == 2]

        dptmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dptmp2[nslbtmp == 2] = pbttmp2[nslbtmp == 2] - ptptmp2[nslbtmp == 2]

        # Adjust negative DPSlab values
        dpnsq = frcsq[(nslbtmp == 2) & (dpslbtmp < 0.0) & (dpslbtmp > -1000.0)] 
        dpadj = numpy.zeros((ctyp1.shape[0],)) 
        dpadj[dpnsq] = numpy.absolute(dpslbtmp[dpnsq])
 
        dpslbtmp[dpnsq] = 1.0
        dptmp1[dpnsq] = dptmp1[dpnsq] / 2.0
        dptmp2[dpnsq] = dptmp2[dpnsq] / 2.0

        # Sigma / Logit Adjustments
        zpbtmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdp1tmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdslbtmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdp2tmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        ncldct = 0
        for t in range(psfc.shape[0]):
            if ( (pbttmp1[t] >= 0.0) and (dpslbtmp[t] >= 0.0) ):
                prptmp = numpy.array( [ (psfc[t] - pbttmp1[t]) / psfc[t], \
                                         dptmp1[t] / psfc[t], dpslbtmp[t] / psfc[t], \
                                         dptmp2[t] / psfc[t], 0.0 ] )
                if (prptmp[0] < 0.0):
                    # Adjustment needed
                    prpadj = prptmp[0]
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    prptmp[2] = prptmp[2] + prpadj*prptmp[2]
                    prptmp[3] = prptmp[3] + prpadj*prptmp[3]
                    ncldct = ncldct + 1
                elif (prptmp[0] == 0.0):
                    # Adjustment needed
                    prpadj = -0.01
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    prptmp[2] = prptmp[2] + prpadj*prptmp[2]
                    prptmp[3] = prptmp[3] + prpadj*prptmp[3]
                    ncldct = ncldct + 1
                prptmp[4] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2] - prptmp[3]
                ztmp = calculate_VPD.lgtzs(prptmp)
                zpbtmp[t] = ztmp[0]
                zdp1tmp[t] = ztmp[1]
                zdslbtmp[t] = ztmp[2]
                zdp2tmp[t] = ztmp[3]
            elif ( pbttmp1[t] >= 0.0  ):
                prptmp = numpy.array( [ (psfc[t] - pbttmp1[t]) / psfc[t], \
                                         dptmp1[t] / psfc[t], 0.0 ] )
                if (prptmp[0] < 0.0):
                    # Adjustment needed
                    prpadj = prptmp[0]
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    ncldct = ncldct + 1
                elif (prptmp[0] == 0.0):
                    # Adjustment needed
                    prpadj = -0.01
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    ncldct = ncldct + 1
                prptmp[2] = 1.0 - prptmp[0] - prptmp[1]
                ztmp = calculate_VPD.lgtzs(prptmp)
                zpbtmp[t] = ztmp[0]
                zdp1tmp[t] = ztmp[1]
                zdslbtmp[t] = -9999.0 
                zdp2tmp[t] = -9999.0 
            else:            
                zpbtmp[t] = -9999.0 
                zdp1tmp[t] = -9999.0 
                zdslbtmp[t] = -9999.0 
                zdp2tmp[t] = -9999.0 
        str1 = 'Cloud Bot Pres Below Sfc: %d ' % (ncldct)
        print(str1)

        if tsmp == 0:
            psfcout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            psfcout[:] = psfc[frcsb]
            prsbot1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            prsbot1out[:] = zpbtmp[frcsb]
            dpcld1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            dpcld1out[:] = zdp1tmp[frcsb]
            dpslbout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            dpslbout[:] = zdslbtmp[frcsb]
            dpcld2out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            dpcld2out[:] = zdp2tmp[frcsb]
        else:
            psfcout = numpy.append(psfcout,psfc[frcsb]) 
            prsbot1out = numpy.append(prsbot1out,zpbtmp[frcsb])
            dpcld1out = numpy.append(dpcld1out,zdp1tmp[frcsb])
            dpslbout = numpy.append(dpslbout,zdslbtmp[frcsb])
            dpcld2out = numpy.append(dpcld2out,zdp2tmp[frcsb])

        # Slab Types: 101.0 = Liquid, 201.0 = Ice, None else
        # Output: 0 = Liquid, 1 = Ice
        typtmp1 = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16) - 99
        typtmp1[nslbtmp >= 1] = (ctyp1[nslbtmp >= 1] - 1.0) / 100.0 - 1.0
        typtmp1[swpsq] = (ctyp2[swpsq] - 1.0) / 100.0 - 1.0

        typtmp2 = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16) - 99
        typtmp2[nslbtmp == 2] = (ctyp2[nslbtmp == 2] - 1.0) / 100.0 - 1.0 
        typtmp2[swpsq] = (ctyp1[swpsq] - 1.0) / 100.0 - 1.0

        if tsmp == 0:
            slbtyp1out = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            slbtyp1out[:] = typtmp1[frcsb]
            slbtyp2out = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            slbtyp2out[:] = typtmp2[frcsb]
        else:
            slbtyp1out = numpy.append(slbtyp1out,typtmp1[frcsb]) 
            slbtyp2out = numpy.append(slbtyp2out,typtmp2[frcsb]) 

        # Cloud Cover Indicators
        totclrtmp = numpy.zeros((frcsb.shape[0],3,3),dtype=numpy.int16)
        cctr = 0
        for frw in range(3):
            for fcl in range(3):
                clrvec = totclr[cctr,:,:,:].flatten()
                totclrtmp[:,frw,fcl] = clrvec[frcsb]
                cctr = cctr + 1
        if tsmp == 0:
            totclrout = numpy.zeros(totclrtmp.shape,dtype=numpy.int16)
            totclrout[:,:,:] = totclrtmp
        else:
            totclrout = numpy.append(totclrout,totclrtmp,axis=0)

        # Cloud Fraction Logit, still account for swapping
        z1tmp = numpy.zeros((frcsb.shape[0],3,3)) - 9999.0
        z2tmp = numpy.zeros((frcsb.shape[0],3,3)) - 9999.0
        z12tmp = numpy.zeros((frcsb.shape[0],3,3)) - 9999.0

        # Cloud Fraction
        cctr = 0
        for frw in range(3):
            for fcl in range(3):
                frcvect = frctot[cctr,:,:,:].flatten()
                frcvec1 = arsfrc1[cctr,:,:,:].flatten()        
                frcvec2 = arsfrc2[cctr,:,:,:].flatten()        

                # Quick fix for totals over 1.0
                fvsq = numpy.arange(frcvect.shape[0])
                fvsq2 = fvsq[frcvect > 1.0]
                frcvect[fvsq2] = frcvect[fvsq2] / 1.0
                frcvec1[fvsq2] = frcvec1[fvsq2] / 1.0
                frcvec2[fvsq2] = frcvec2[fvsq2] / 1.0
              
                for t in range(nairs):
                    crslb = nslbtmp[frcsb[t]]
                    crclr = totclrtmp[t,frw,fcl]
                    if ( (crslb == 0) or (crclr == -1) ):
                        z1tmp[t,frw,fcl] = -9999.0
                        z2tmp[t,frw,fcl] = -9999.0
                        z12tmp[t,frw,fcl] = -9999.0
                    elif ( (crslb == 1) and (crclr == 1) ):
                        z1tmp[t,frw,fcl] = -9999.0
                        z2tmp[t,frw,fcl] = -9999.0
                        z12tmp[t,frw,fcl] = -9999.0
                    elif ( (crslb == 1) and (crclr == 0) ):
                        prptmp = numpy.array( [frcvect[frcsb[t]], 1.0 - frcvect[frcsb[t]] ] )
                        ztmp = calculate_VPD.lgtzs(prptmp)
                        z1tmp[t,frw,fcl] = ztmp[0] 
                        z2tmp[t,frw,fcl] = -9999.0
                        z12tmp[t,frw,fcl] = -9999.0
                    # For 2 slabs, recall AIRS cloud layers go upper/lower, ours is opposite
                    # Also apply random overlap adjust AIRS zero values
                    elif ( (crslb == 2) and (crclr == 0) ):
                        frcs = numpy.array([frcvec2[frcsb[t]],frcvec1[frcsb[t]]])
                        if (numpy.sum(frcs) < 0.01):
                            frcs[0] = 0.005
                            frcs[1] = 0.005
                        elif frcs[0] < 0.005:
                            frcs[0] = 0.005
                            frcs[1] = frcs[1] - 0.005
                        elif frcs[1] < 0.005:
                            frcs[1] = 0.005
                            frcs[0] = frcs[0] - 0.005
                        mnfrc = numpy.amin(frcs)
                        c12tmp = random.uniform(0.0,mnfrc,size=1)
                        prptmp = numpy.array( [frcs[0] - c12tmp[0]*frcs[1], \
                                               frcs[1] - c12tmp[0]*frcs[0], c12tmp[0], 0.0])
                        prptmp[3] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2]
                        ztmp = calculate_VPD.lgtzs(prptmp)
                        z1tmp[t,frw,fcl] = ztmp[0] 
                        z2tmp[t,frw,fcl] = ztmp[1] 
                        z12tmp[t,frw,fcl] = ztmp[2]
                    elif ( (crslb == 2) and (crclr == 1) ):
                        frcs = numpy.array([frcvec2[frcsb[t]],frcvec1[frcsb[t]]])
                        if frcs[0] < 0.005: 
                            frcs[0] = 0.005
                            frcs[1] = frcs[1] - 0.005
                        elif frcs[1] < 0.005:
                            frcs[1] = 0.005
                            frcs[0] = frcs[0] - 0.005
                        mnfrc = numpy.amin(frcs)
                        c12tmp = random.uniform(0.0,mnfrc,size=1)
                        prptmp = numpy.array( [0.999 * (frcs[0] - c12tmp[0]*frcs[1]), \
                                               0.999 * (frcs[1] - c12tmp[0]*frcs[0]), 0.999 * c12tmp[0], 0.001])
                        prptmp[3] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2]
                        ztmp = calculate_VPD.lgtzs(prptmp)
                        z1tmp[t,frw,fcl] = ztmp[0] 
                        z2tmp[t,frw,fcl] = ztmp[1] 
                        z12tmp[t,frw,fcl] = ztmp[2]
                    

                cctr = cctr + 1


        if tsmp == 0:
            cfclgt1out = numpy.zeros(z1tmp.shape)
            cfclgt1out[:,:,:] = z1tmp
            cfclgt2out = numpy.zeros(z2tmp.shape)
            cfclgt2out[:,:,:] = z2tmp
            cfclgt12out = numpy.zeros(z12tmp.shape)
            cfclgt12out[:,:,:] = z12tmp
        else:
            cfclgt1out = numpy.append(cfclgt1out,z1tmp,axis=0) 
            cfclgt2out = numpy.append(cfclgt2out,z2tmp,axis=0) 
            cfclgt12out = numpy.append(cfclgt12out,z12tmp,axis=0) 


        # Cloud Non-Gas Water
        ngwttmp1 = numpy.zeros(cngwt1.shape[0]) - 9999.0
        ngwttmp1[nslbtmp >= 1] = cngwt1[nslbtmp >= 1]
        ngwttmp1[swpsq] = cngwt2[swpsq]

        ngwttmp2 = numpy.zeros(cngwt1.shape[0]) - 9999.0
        ngwttmp2[nslbtmp == 2] = cngwt2[nslbtmp == 2] 
        ngwttmp2[swpsq] = cngwt1[swpsq] 

        if tsmp == 0:
            ngwt1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            ngwt1out[:] = ngwttmp1[frcsb]
            ngwt2out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            ngwt2out[:] = ngwttmp2[frcsb]
        else:
            ngwt1out = numpy.append(ngwt1out,ngwttmp1[frcsb]) 
            ngwt2out = numpy.append(ngwt2out,ngwttmp2[frcsb]) 

        # Cloud Top Temperature 
        cttptmp1 = numpy.zeros(cttp1.shape[0]) - 9999.0
        cttptmp1[nslbtmp >= 1] = cttp1[nslbtmp >= 1]
        cttptmp1[swpsq] = cttp2[swpsq]

        cttptmp2 = numpy.zeros(cttp1.shape[0]) - 9999.0
        cttptmp2[nslbtmp == 2] = cttp2[nslbtmp == 2] 
        cttptmp2[swpsq] = cttp1[swpsq] 

        if tsmp == 0:
            cttp1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            cttp1out[:] = cttptmp1[frcsb]
            cttp2out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            cttp2out[:] = cttptmp2[frcsb]
        else:
            cttp1out = numpy.append(cttp1out,cttptmp1[frcsb]) 
            cttp2out = numpy.append(cttp2out,cttptmp2[frcsb]) 

        # Loc/Time
        if tsmp == 0:
            latout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            latout[:] = lthld[frcsb]
            lonout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            lonout[:] = lnhld[frcsb]
            yrout = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            yrout[:] = yrlst[k]
            jdyout = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            jdyout[:] = tmhld[frcsb]
        else:
            latout = numpy.append(latout,lthld[frcsb])
            lonout = numpy.append(lonout,lnhld[frcsb])
            yrtmp = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            yrtmp[:] = yrlst[k]
            yrout = numpy.append(yrout,yrtmp)
            jdyout = numpy.append(jdyout,tmhld[frcsb])

        tsmp = tsmp + nairs 

    # Process quantiles

    nslbqs = calculate_VPD.quantile_msgdat_discrete(nslabout,prbs)
    str1 = '%.2f Number Slab Quantile: %d' % (prbs[103],nslbqs[103])
    print(str1)
    print(nslbqs)

#    psfcqs = calculate_VPD.quantile_msgdat(psfcout,prbs)
#    str1 = '%.2f Surface Pressure Quantile: %.3f' % (prbs[53],psfcqs[53])
#    print(str1)

    prsbt1qs = calculate_VPD.quantile_msgdat(prsbot1out,prbs)
    str1 = '%.2f CldBot1 Pressure Quantile: %.3f' % (prbs[103],prsbt1qs[103])
    print(str1)

    dpcld1qs = calculate_VPD.quantile_msgdat(dpcld1out,prbs)
    str1 = '%.2f DPCloud1 Quantile: %.3f' % (prbs[103],dpcld1qs[103])
    print(str1)

    dpslbqs = calculate_VPD.quantile_msgdat(dpslbout,prbs)
    str1 = '%.2f DPSlab Quantile: %.3f' % (prbs[103],dpslbqs[103])
    print(str1)

    dpcld2qs = calculate_VPD.quantile_msgdat(dpcld2out,prbs)
    str1 = '%.2f DPCloud2 Quantile: %.3f' % (prbs[103],dpcld2qs[103])
    print(str1)

    slb1qs = calculate_VPD.quantile_msgdat_discrete(slbtyp1out,prbs)
    str1 = '%.2f Type1 Quantile: %d' % (prbs[103],slb1qs[103])
    print(str1)

    slb2qs = calculate_VPD.quantile_msgdat_discrete(slbtyp2out,prbs)
    str1 = '%.2f Type2 Quantile: %d' % (prbs[103],slb2qs[103])
    print(str1)

    # Indicators
    totclrqout = numpy.zeros((3,3,nprb)) - 99
    lgt1qs = numpy.zeros((3,3,nprb)) - 9999.0    
    lgt2qs = numpy.zeros((3,3,nprb)) - 9999.0    
    lgt12qs = numpy.zeros((3,3,nprb)) - 9999.0    

    for frw in range(3):
        for fcl in range(3):
            tmpclr = calculate_VPD.quantile_msgdat_discrete(totclrout[:,frw,fcl],prbs)
            totclrqout[frw,fcl,:] = tmpclr[:]
            str1 = 'Clr/Ovc Indicator %d, %d %.2f Quantile: %d' % (frw,fcl,prbs[103],tmpclr[103])
            print(str1)

            tmplgtq = calculate_VPD.quantile_msgdat(cfclgt1out[:,frw,fcl],prbs)
            lgt1qs[frw,fcl,:] = tmplgtq[:]
            tmplgtq = calculate_VPD.quantile_msgdat(cfclgt2out[:,frw,fcl],prbs)
            lgt2qs[frw,fcl,:] = tmplgtq[:]
            tmplgtq = calculate_VPD.quantile_msgdat(cfclgt12out[:,frw,fcl],prbs)
            lgt12qs[frw,fcl,:] = tmplgtq[:]
            str1 = 'CFrac Logit %d, %d %.2f Quantile: %.3f, %.3f, %.3f' % (frw,fcl,prbs[103], \
                                lgt1qs[frw,fcl,103],lgt2qs[frw,fcl,103],lgt12qs[frw,fcl,103])
            print(str1)

    ngwt1qs = calculate_VPD.quantile_msgdat(ngwt1out,prbs)
    str1 = '%.2f NGWater1 Quantile: %.3f' % (prbs[103],ngwt1qs[103])
    print(str1)

    ngwt2qs = calculate_VPD.quantile_msgdat(ngwt2out,prbs)
    str1 = '%.2f NGWater2 Quantile: %.3f' % (prbs[103],ngwt2qs[103])
    print(str1)

    cttp1qs = calculate_VPD.quantile_msgdat(cttp1out,prbs)
    str1 = '%.2f CTTemp1 Quantile: %.3f' % (prbs[103],cttp1qs[103])
    print(str1)

    cttp2qs = calculate_VPD.quantile_msgdat(cttp2out,prbs)
    str1 = '%.2f CTTemp2 Quantile: %.3f' % (prbs[103],cttp2qs[103])
    print(str1)

    # Output Quantiles
    qfnm = '%s/CONUS_AIRS_JJA_%04d_%02dUTC_%s_Cloud_Quantile.nc' % (dtdr,yrlst[k],hrchc,rgchc)
    qout = Dataset(qfnm,'w') 

    dimp = qout.createDimension('probability',nprb)
    dimfov1 = qout.createDimension('fovrow',3)
    dimfov2 = qout.createDimension('fovcol',3)

    varprb = qout.createVariable('probability','f4',['probability'], fill_value = -9999)
    varprb[:] = prbs
    varprb.long_name = 'Probability break points'
    varprb.units = 'none'
    varprb.missing_value = -9999

    varnslb = qout.createVariable('NumberSlab_quantile','i2',['probability'], fill_value = -99)
    varnslb[:] = nslbqs
    varnslb.long_name = 'Number of cloud slabs quantiles'
    varnslb.units = 'Count'
    varnslb.missing_value = -99

    varcbprs = qout.createVariable('CloudBot1Logit_quantile','f4',['probability'], fill_value = -9999)
    varcbprs[:] = prsbt1qs
    varcbprs.long_name = 'Slab 1 cloud bottom pressure logit quantiles'
    varcbprs.units = 'hPa'
    varcbprs.missing_value = -9999

    vardpc1 = qout.createVariable('DPCloud1Logit_quantile','f4',['probability'], fill_value = -9999)
    vardpc1[:] = dpcld1qs
    vardpc1.long_name = 'Slab 1 cloud pressure depth logit quantiles'
    vardpc1.units = 'hPa'
    vardpc1.missing_value = -9999

    vardpslb = qout.createVariable('DPSlabLogit_quantile','f4',['probability'], fill_value = -9999)
    vardpslb[:] = dpslbqs
    vardpslb.long_name = 'Two-slab vertical separation logit quantiles' 
    vardpslb.units = 'hPa'
    vardpslb.missing_value = -9999

    vardpc2 = qout.createVariable('DPCloud2Logit_quantile','f4',['probability'], fill_value = -9999)
    vardpc2[:] = dpcld2qs
    vardpc2.long_name = 'Slab 2 cloud pressure depth logit quantiles'
    vardpc2.units = 'hPa'
    vardpc2.missing_value = -9999

    vartyp1 = qout.createVariable('CType1_quantile','i2',['probability'], fill_value = -99)
    vartyp1[:] = slb1qs
    vartyp1.long_name = 'Slab 1 cloud type quantiles'
    vartyp1.units = 'None'
    vartyp1.missing_value = -99
    vartyp1.comment = 'Cloud slab type: 0=Liquid, 1=Ice'

    vartyp2 = qout.createVariable('CType2_quantile','i2',['probability'], fill_value = -99)
    vartyp2[:] = slb2qs
    vartyp2.long_name = 'Slab 2 cloud type quantiles'
    vartyp2.units = 'None'
    vartyp2.missing_value = -99
    vartyp2.comment = 'Cloud slab type: 0=Liquid, 1=Ice'

    varcvr = qout.createVariable('CCoverInd_quantile','i2',['fovrow','fovcol','probability'], fill_value = 99)
    varcvr[:] = totclrqout
    varcvr.long_name = 'Cloud cover indicator quantiles'
    varcvr.units = 'None'
    varcvr.missing_value = -99
    varcvr.comment = 'Cloud cover indicators: -1=Clear, 0=Partly cloudy, 1=Overcast'

    varlgt1 = qout.createVariable('CFrcLogit1_quantile','f4',['fovrow','fovcol','probability'], fill_value = -9999)
    varlgt1[:] = lgt1qs
    varlgt1.long_name = 'Slab 1 cloud fraction (cfrac1x) logit quantiles'
    varlgt1.units = 'None'
    varlgt1.missing_value = -9999

    varlgt2 = qout.createVariable('CFrcLogit2_quantile','f4',['fovrow','fovcol','probability'], fill_value = -9999)
    varlgt2[:] = lgt2qs
    varlgt2.long_name = 'Slab 2 cloud fraction (cfrac2x) logit quantiles'
    varlgt2.units = 'None'
    varlgt2.missing_value = -9999

    varlgt12 = qout.createVariable('CFrcLogit12_quantile','f4',['fovrow','fovcol','probability'], fill_value = -9999)
    varlgt12[:] = lgt12qs
    varlgt12.long_name = 'Slab 1/2 overlap fraction (cfrac12) logit quantiles'
    varlgt12.units = 'None'
    varlgt12.missing_value = -9999

    varngwt1 = qout.createVariable('NGWater1_quantile','f4',['probability'], fill_value = -9999)
    varngwt1[:] = ngwt1qs
    varngwt1.long_name = 'Slab 1 cloud non-gas water quantiles'
    varngwt1.units = 'g m^-2'
    varngwt1.missing_value = -9999

    varngwt2 = qout.createVariable('NGWater2_quantile','f4',['probability'], fill_value = -9999)
    varngwt2[:] = ngwt2qs
    varngwt2.long_name = 'Slab 2 cloud non-gas water quantiles'
    varngwt2.units = 'g m^-2'
    varngwt2.missing_value = -9999

    varcttp1 = qout.createVariable('CTTemp1_quantile','f4',['probability'], fill_value = -9999)
    varcttp1[:] = cttp1qs
    varcttp1.long_name = 'Slab 1 cloud top temperature'
    varcttp1.units = 'K'
    varcttp1.missing_value = -9999

    varcttp2 = qout.createVariable('CTTemp2_quantile','f4',['probability'], fill_value = -9999)
    varcttp2[:] = cttp2qs
    varcttp2.long_name = 'Slab 2 cloud top temperature'
    varcttp2.units = 'K'
    varcttp2.missing_value = -9999

    qout.close()


    # Set up transformations
    zccvout = numpy.zeros((tsmp,3,3,)) - 9999.
    zlgt1 = numpy.zeros((tsmp,3,3)) - 9999.
    zlgt2 = numpy.zeros((tsmp,3,3)) - 9999.
    zlgt12 = numpy.zeros((tsmp,3,3)) - 9999.

    znslb = calculate_VPD.std_norm_quantile_from_obs(nslabout, nslbqs, prbs,  msgval=-99)
    zprsbt1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(prsbot1out, prsbt1qs, prbs,  msgval=-9999.)
    zdpcld1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpcld1out, dpcld1qs, prbs,  msgval=-9999.)
    zdpslb = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpslbout, dpslbqs, prbs,  msgval=-9999.)
    zdpcld2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpcld2out, dpcld2qs, prbs,  msgval=-9999.)
    zctyp1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(slbtyp1out, slb1qs, prbs,  msgval=-99)
    zctyp2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(slbtyp2out, slb2qs, prbs,  msgval=-99)

    for frw in range(3):
        for fcl in range(3):
            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(totclrout[:,frw,fcl], totclrqout[frw,fcl,:], \
                                                                     prbs, msgval=-99)
            zccvout[:,frw,fcl] = ztmp[:]

            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt1out[:,frw,fcl], lgt1qs[frw,fcl,:], \
                                                                     prbs, msgval=-9999.)
            zlgt1[:,frw,fcl] = ztmp[:]

            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt2out[:,frw,fcl], lgt2qs[frw,fcl,:], \
                                                                     prbs, msgval=-9999.)
            zlgt2[:,frw,fcl] = ztmp[:]

            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt12out[:,frw,fcl], lgt12qs[frw,fcl,:], \
                                                                     prbs, msgval=-9999.)
            zlgt12[:,frw,fcl] = ztmp[:]

    zngwt1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(ngwt1out, ngwt1qs, prbs,  msgval=-9999.)
    zngwt2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(ngwt2out, ngwt2qs, prbs,  msgval=-9999.)
    zcttp1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cttp1out, cttp1qs, prbs,  msgval=-9999.)
    zcttp2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cttp2out, cttp2qs, prbs,  msgval=-9999.)


    # Output transformed quantile samples
    zfnm = '%s/CONUS_AIRS_JJA_%04d_%02dUTC_%s_Cloud_StdGausTrans.nc' % (dtdr,yrlst[k],hrchc,rgchc)
    zout = Dataset(zfnm,'w') 

    dimsmp = zout.createDimension('sample',tsmp)
    dimfov1 = zout.createDimension('fovrow',3)
    dimfov2 = zout.createDimension('fovcol',3)

    varlon = zout.createVariable('Longitude','f4',['sample'])
    varlon[:] = lonout
    varlon.long_name = 'Longitude'
    varlon.units = 'degrees_east'

    varlat = zout.createVariable('Latitude','f4',['sample'])
    varlat[:] = latout
    varlat.long_name = 'Latitude'
    varlat.units = 'degrees_north'

    varjdy = zout.createVariable('JulianDay','i2',['sample'])
    varjdy[:] = jdyout
    varjdy.long_name = 'JulianDay'
    varjdy.units = 'day'

    varyr = zout.createVariable('Year','i2',['sample'])
    varyr[:] = yrout
    varyr.long_name = 'Year'
    varyr.units = 'year'

    varnslb = zout.createVariable('NumberSlab_StdGaus','f4',['sample'], fill_value = -9999)
    varnslb[:] = znslb
    varnslb.long_name = 'Quantile transformed number of cloud slabs'
    varnslb.units = 'None'
    varnslb.missing_value = -9999.

    varcbprs = zout.createVariable('CloudBot1Logit_StdGaus','f4',['sample'], fill_value = -9999)
    varcbprs[:] = zprsbt1
    varcbprs.long_name = 'Quantile transformed slab 1 cloud bottom pressure logit'
    varcbprs.units = 'None'
    varcbprs.missing_value = -9999.

    vardpc1 = zout.createVariable('DPCloud1Logit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpc1[:] = zdpcld1
    vardpc1.long_name = 'Quantile transformed slab 1 cloud pressure depth logit'
    vardpc1.units = 'None'
    vardpc1.missing_value = -9999.

    vardpslb = zout.createVariable('DPSlabLogit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpslb[:] = zdpslb
    vardpslb.long_name = 'Quantile transformed two-slab vertical separation logit'
    vardpslb.units = 'None'
    vardpslb.missing_value = -9999.

    vardpc2 = zout.createVariable('DPCloud2Logit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpc2[:] = zdpcld2
    vardpc2.long_name = 'Quantile transformed slab 2 cloud pressure depth logit'
    vardpc2.units = 'None'
    vardpc2.missing_value = -9999.

    vartyp1 = zout.createVariable('CType1_StdGaus','f4',['sample'], fill_value = -9999)
    vartyp1[:] = zctyp1
    vartyp1.long_name = 'Quantile transformed slab 1 cloud type logit'
    vartyp1.units = 'None'
    vartyp1.missing_value = -9999.

    vartyp2 = zout.createVariable('CType2_StdGaus','f4',['sample'], fill_value = -9999)
    vartyp2[:] = zctyp2
    vartyp2.long_name = 'Quantile transformed slab 2 cloud type'
    vartyp2.units = 'None'
    vartyp2.missing_value = -9999.

    varcov = zout.createVariable('CCoverInd_StdGaus','f4',['sample','fovrow','fovcol'], fill_value= -9999)
    varcov[:] = zccvout
    varcov.long_name = 'Quantile transformed cloud cover indicator'
    varcov.units = 'None'
    varcov.missing_value = -9999.

    varlgt1 = zout.createVariable('CFrcLogit1_StdGaus','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    varlgt1[:] = zlgt1
    varlgt1.long_name = 'Quantile transformed slab 1 cloud fraction logit'
    varlgt1.units = 'None'
    varlgt1.missing_value = -9999.

    varlgt2 = zout.createVariable('CFrcLogit2_StdGaus','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    varlgt2[:] = zlgt2
    varlgt2.long_name = 'Quantile transformed slab 2 cloud fraction logit'
    varlgt2.units = 'None'
    varlgt2.missing_value = -9999.

    varlgt12 = zout.createVariable('CFrcLogit12_StdGaus','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    varlgt12[:] = zlgt12
    varlgt12.long_name = 'Quantile transformed slab 1/2 overlap fraction logit'
    varlgt12.units = 'None'
    varlgt12.missing_value = -9999.

    varngwt1 = zout.createVariable('NGWater1_StdGaus','f4',['sample'], fill_value = -9999)
    varngwt1[:] = zngwt1
    varngwt1.long_name = 'Quantile transformed slab 1 non-gas water'
    varngwt1.units = 'None'
    varngwt1.missing_value = -9999.

    varngwt2 = zout.createVariable('NGWater2_StdGaus','f4',['sample'], fill_value = -9999)
    varngwt2[:] = zngwt2
    varngwt2.long_name = 'Quantile transformed slab 2 non-gas water'
    varngwt2.units = 'None'
    varngwt2.missing_value = -9999.

    varcttp1 = zout.createVariable('CTTemp1_StdGaus','f4',['sample'], fill_value = -9999)
    varcttp1[:] = zcttp1
    varcttp1.long_name = 'Quantile transformed slab 1 cloud top temperature'
    varcttp1.units = 'None'
    varcttp1.missing_value = -9999.

    varcttp2 = zout.createVariable('CTTemp2_StdGaus','f4',['sample'], fill_value = -9999)
    varcttp2[:] = zcttp2
    varcttp2.long_name = 'Quantile transformed slab 2 cloud top temperature'
    varcttp2.units = 'None'
    varcttp2.missing_value = -9999.

    zout.close()

    return

def quantile_profile_locmask_conus(rfdr, mtdr, csdr, airdr, dtdr, yrlst, mnst, mnfn, hrchc, rgchc, mskvr, mskvl):
    # Construct profile/sfc variable quantiles and z-scores, with a possibly irregular location mask
    # rfdr:    Directory for reference data (Levels/Quantiles)
    # mtdr:    Directory for MERRA data
    # csdr:    Directory for cloud slab data
    # airdr:   Directory for AIRS cloud fraction
    # dtdr:    Output directory
    # yrlst:   List of years to process
    # mnst:    Starting Month
    # mnfn:    Ending Month 
    # hrchc:   Template Hour Choice
    # rgchc:   Template Region Choice
    # mskvr:   Name of region mask variable
    # mskvl:   Value of region mask for Region Choice

    # Read probs and pressure levels
    rnm = '%s/AIRS_Levels_Quantiles.nc' % (rfdr)
    f = Dataset(rnm,'r')
    plev = f['level'][:]
    prbs = f['probability'][:]
    alts = f['altitude'][:]
    f.close()

    nyr = len(yrlst)
    nprb = prbs.shape[0]
    nzout = 101

    tmpqout = numpy.zeros((nzout,nprb)) - 9999.
    rhqout = numpy.zeros((nzout,nprb)) - 9999.
    sftmpqs = numpy.zeros((nprb,)) - 9999.
    sfaltqs = numpy.zeros((nprb,)) - 9999.
    psfcqs = numpy.zeros((nprb,)) - 9999.
    altmed = numpy.zeros((nzout,)) - 9999.


    # Mask, lat, lon
    fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_CONUS_with_NCA_regions_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[0],hrchc)
    f = Dataset(fnm,'r')
    mask = f.variables[mskvr][:,:]
    latmet = f.variables['plat'][:]
    lonmet = f.variables['plon'][:]
    tminf = f.variables['time'][:]
    tmunit = f.variables['time'].units[:]
    f.close()

    mskind = numpy.zeros((mask.shape),dtype=mask.dtype)
    print(mskvl)
    mskind[mask == mskvl] = 1
    lnsq = numpy.arange(lonmet.shape[0])
    ltsq = numpy.arange(latmet.shape[0])

    # Subset a bit
    lnsm = numpy.sum(mskind,axis=0)
    #print(lnsq.shape)
    #print(lnsm.shape)
    #print(lnsm) 
    ltsm = numpy.sum(mskind,axis=1)
    #print(ltsq.shape)
    #print(ltsm.shape)
    #print(ltsm)

    lnmn = numpy.amin(lnsq[lnsm > 0])
    lnmx = numpy.amax(lnsq[lnsm > 0]) + 1
    ltmn = numpy.amin(ltsq[ltsm > 0])
    ltmx = numpy.amax(ltsq[ltsm > 0]) + 1

    stridx = 'Lon Range: %d, %d\nLat Range: %d, %d \n' % (lnmn,lnmx,ltmn,ltmx)
    print(stridx)

    nx = lnmx - lnmn
    ny = ltmx - ltmn 

    lnrp = numpy.tile(lonmet[lnmn:lnmx],ny)
    ltrp = numpy.repeat(latmet[ltmn:ltmx],nx)
    mskblk = mskind[ltmn:ltmx,lnmn:lnmx]
    mskflt = mskblk.flatten()


    tsmp = 0
    for k in range(nyr):
        fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_CONUS_with_NCA_regions_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[k],hrchc)
        f = Dataset(fnm,'r')
        tminf = f.variables['time'][:]
        tmunit = f.variables['time'].units[:]
        f.close()

        tmunit = tmunit.replace("days since ","")
        dybs = datetime.datetime.strptime(tmunit,"%Y-%m-%d %H:%M:%S")
        print(dybs)
        dy0 = dybs + datetime.timedelta(days=tminf[0]) 
        dyinit = datetime.date(dy0.year,dy0.month,dy0.day)
        print(dyinit)
 
        dyst = datetime.date(yrlst[k],mnst,1)
        ttst = dyst.timetuple()
        jst = ttst.tm_yday
        if mnfn < 12:
            dyfn = datetime.date(yrlst[k],mnfn+1,1)
            ttfn = dyfn.timetuple()
            jfn = ttfn.tm_yday
        else:
             dyfn = datetime.date(yrlst[k]+1,1,1)
             dy31 = datetime.date(yrlst[k],12,31)
             tt31 = dy31.timetuple()
             jfn = tt31.tm_yday + 1
 
        dystidx = abs((dyst-dyinit).days)
        dyfnidx = abs((dyfn-dyinit).days)

        jdsq = numpy.arange(jst,jfn)
        print(jdsq)
        tmhld = numpy.repeat(jdsq,nx*ny)
        #print(tmhld.shape)
        #print(numpy.amin(tmhld))
        #print(numpy.amax(tmhld))

        stridx = 'Day Range: %d, %d\n' % (dystidx,dyfnidx)
        print(stridx)

        # MERRA variables
        fnm = '%s/interpolated_merra2_for_SARTA_two_slab_%d_JJA_CONUS_with_NCA_regions_%02dUTC_no_vertical_variation_for_missing.nc' % (mtdr,yrlst[k],hrchc)
        f = Dataset(fnm,'r')
        tms = f.variables['time'][:]
        stparr = f['/stemp'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        psfarr = f['/spres'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        salarr = f['/salti'][ltmn:ltmx,lnmn:lnmx]
        tmparr = f['/ptemp'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        h2oarr = f['/rh'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        altarr = f['/palts'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        f.close()

        tmflt = tms.flatten()
        nt = tmflt.shape[0]
        lnhld = numpy.tile(lnrp,nt)
        lthld = numpy.tile(ltrp,nt)

        mskall = numpy.tile(mskflt,nt)
        msksq = numpy.arange(mskall.shape[0])
        msksb = msksq[mskall > 0]
        mskstr = 'Total Obs: %d, Within Mask: %d \n' % (msksq.shape[0],msksb.shape[0])
        print(mskstr)


        # AIRS Clouds
        anm = '%s/CONUS_AIRS_CldFrc_Match_JJA_%d_%02d_UTC.nc' % (airdr,yrlst[k],hrchc)
        f = Dataset(anm,'r')
        arsfrc1 = f.variables['AIRS_CldFrac_1'][:,dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        arsfrc2 = f.variables['AIRS_CldFrac_2'][:,dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        f.close()

        # Sum
        frctot = arsfrc1 + arsfrc2

        frc0 = frctot[0,:,:,:]
        frc0 = frc0.flatten()
        frcsq = numpy.arange(tmhld.shape[0])
        # Subset by AIRS matchup and location masks
        frcsb = frcsq[(numpy.logical_not(frc0.mask)) & (mskall > 0)]

        nairs = frcsb.shape[0]
        print(tmhld.shape)
        print(frcsb.shape)

        tmptmp = numpy.zeros((nairs,nzout))
        h2otmp = numpy.zeros((nairs,nzout))
        alttmp = numpy.zeros((nairs,nzout))
        for j in range(nzout):
            tmpvec = tmparr[:,j,:,:].flatten()
            tmpvec[tmpvec > 1e30] = -9999.
            tmptmp[:,j] = tmpvec[frcsb]

            altvec = altarr[:,j,:,:].flatten()
            alttmp[:,j] = altvec[frcsb]

            h2ovec = h2oarr[:,j,:,:].flatten()
            h2ovec[h2ovec > 1e30] = -9999.
            h2otmp[:,j] = h2ovec[frcsb]
        if tsmp == 0:
            tmpmerout = numpy.zeros(tmptmp.shape)
            tmpmerout[:,:] = tmptmp
            h2omerout = numpy.zeros(h2otmp.shape)
            h2omerout[:,:] = h2otmp
            altout = numpy.zeros(alttmp.shape)
            altout[:,:] = alttmp
        else:
            tmpmerout = numpy.append(tmpmerout,tmptmp,axis=0)
            h2omerout = numpy.append(h2omerout,h2otmp,axis=0)
            altout = numpy.append(altout,alttmp,axis=0)
             

        stparr = stparr.flatten()
        psfarr = psfarr.flatten()
        salarr = salarr.flatten()
        salfl = numpy.tile(salarr[:],nt) 
 
        if tsmp == 0:
            sftmpout = numpy.zeros((nairs,)) - 9999.0
            sftmpout[:] = stparr[frcsb]
            psfcout = numpy.zeros((nairs,)) - 9999.0
            psfcout[:] = psfarr[frcsb]
            sfaltout = numpy.zeros((nairs,)) - 9999.0 
            sfaltout[:] = salfl[frcsb] 
        else:
            sftmpout = numpy.append(sftmpout,stparr[frcsb])
            psfcout = numpy.append(psfcout,psfarr[frcsb])
            sfaltout = numpy.append(sfaltout,salfl[frcsb])

        # Loc/Time
        if tsmp == 0:
            latout = numpy.zeros((nairs,)) - 9999.0
            latout[:] = lthld[frcsb]
            lonout = numpy.zeros((nairs,)) - 9999.0
            lonout[:] = lnhld[frcsb]
            yrout = numpy.zeros((nairs,),dtype=numpy.int16)
            yrout[:] = yrlst[k]
            jdyout = numpy.zeros((nairs,),dtype=numpy.int16)
            jdyout[:] = tmhld[frcsb]
        else:
            latout = numpy.append(latout,lthld[frcsb])
            lonout = numpy.append(lonout,lnhld[frcsb])
            yrtmp = numpy.zeros((nairs,),dtype=numpy.int16)
            yrtmp[:] = yrlst[k]
            yrout = numpy.append(yrout,yrtmp)
            jdyout = numpy.append(jdyout,tmhld[frcsb])

        tsmp = tsmp + nairs 


    # Quantiles
    tmpqout = numpy.zeros((nzout,nprb)) - 9999.
    rhqout = numpy.zeros((nzout,nprb)) - 9999.
    sftmpqs = numpy.zeros((nprb,)) - 9999.
    sfaltqs = numpy.zeros((nprb,)) - 9999.
    psfcqs = numpy.zeros((nprb,)) - 9999.
    altmed = numpy.zeros((nzout,)) - 9999.

    ztmpout = numpy.zeros((tsmp,nzout)) - 9999.
    zrhout = numpy.zeros((tsmp,nzout)) - 9999.
    zsftmpout = numpy.zeros((tsmp,)) - 9999.
    zsfaltout = numpy.zeros((tsmp,)) - 9999.
    zpsfcout = numpy.zeros((tsmp,)) - 9999.

    # Quantiles
    for j in range(nzout):
        tmptmp = calculate_VPD.quantile_msgdat(tmpmerout[:,j],prbs)
        tmpqout[j,:] = tmptmp[:]
        str1 = 'Plev %.2f, %.2f Temp Quantile: %.3f' % (plev[j],prbs[103],tmptmp[103])
        print(str1)

        # Transform
        ztmp = calculate_VPD.std_norm_quantile_from_obs(tmpmerout[:,j], tmptmp, prbs,  msgval=-9999.)
        ztmpout[:,j] = ztmp[:]

        alttmp = calculate_VPD.quantile_msgdat(altout[:,j],prbs)
        altmed[j] = alttmp[103]
        str1 = 'Plev %.2f, %.2f Alt Quantile: %.3f' % (plev[j],prbs[103],alttmp[103])
        print(str1)

        # Adjust RH over 100
        rhadj = h2omerout[:,j]
        rhadj[rhadj > 1.0] = 1.0
        rhqtmp = calculate_VPD.quantile_msgdat(rhadj,prbs)
        rhqout[j,:] = rhqtmp[:]
        str1 = 'Plev %.2f, %.2f RH Quantile: %.4f' % (plev[j],prbs[103],rhqtmp[103])
        print(str1)

        zrh = calculate_VPD.std_norm_quantile_from_obs(rhadj, rhqtmp, prbs,  msgval=-9999.)
        zrhout[:,j] = zrh[:]

    psfcqs = calculate_VPD.quantile_msgdat(psfcout,prbs)
    str1 = '%.2f PSfc Quantile: %.2f' % (prbs[103],psfcqs[103])
    print(str1)
    zpsfcout = calculate_VPD.std_norm_quantile_from_obs(psfcout, psfcqs, prbs, msgval=-9999.) 

    sftpqs = calculate_VPD.quantile_msgdat(sftmpout,prbs)
    str1 = '%.2f SfcTmp Quantile: %.2f' % (prbs[103],sftpqs[103])
    print(str1)
    zsftmpout = calculate_VPD.std_norm_quantile_from_obs(sftmpout, sftpqs, prbs, msgval=-9999.) 

    sfalqs = calculate_VPD.quantile_msgdat(sfaltout,prbs)
    str1 = '%.2f SfcAlt Quantile: %.2f' % (prbs[103],sfalqs[103])
    print(str1)
    zsfaltout = calculate_VPD.std_norm_quantile_from_obs(sfaltout, sfalqs, prbs, msgval=-9999.) 

    # Output Quantiles
    qfnm = '%s/CONUS_AIRS_JJA_%04d_%02dUTC_%s_TempRHSfc_Quantile.nc' % (dtdr,yrlst[k],hrchc,rgchc)
    qout = Dataset(qfnm,'w') 

    dimz = qout.createDimension('level',nzout)
    dimp = qout.createDimension('probability',nprb)

    varlvl = qout.createVariable('level','f4',['level'], fill_value = -9999)
    varlvl[:] = plev
    varlvl.long_name = 'AIRS/SARTA pressure levels'
    varlvl.units = 'hPa'
    varlvl.missing_value = -9999

    varprb = qout.createVariable('probability','f4',['probability'], fill_value = -9999)
    varprb[:] = prbs
    varprb.long_name = 'Probability break points'
    varprb.units = 'none'
    varprb.missing_value = -9999

    # Altitude grid
    varalt = qout.createVariable('Altitude_median', 'f4', ['level'], fill_value = -9999)
    varalt[:] = altmed
    varalt.long_name = 'Altitude median value'
    varalt.units = 'm'
    varalt.missing_value = -9999

    vartmp = qout.createVariable('Temperature_quantile', 'f4', ['level','probability'], fill_value = -9999)
    vartmp[:] = tmpqout
    vartmp.long_name = 'Temperature quantiles'
    vartmp.units = 'K'
    vartmp.missing_value = -9999.

    varrh = qout.createVariable('RH_quantile', 'f4', ['level','probability'], fill_value = -9999)
    varrh[:] = rhqout
    varrh.long_name = 'Relative humidity quantiles'
    varrh.units = 'Unitless'
    varrh.missing_value = -9999.

    varstmp = qout.createVariable('SfcTemp_quantile', 'f4', ['probability'], fill_value = -9999)
    varstmp[:] = sftpqs
    varstmp.long_name = 'Surface temperature quantiles'
    varstmp.units = 'K'
    varstmp.missing_value = -9999.

    varpsfc = qout.createVariable('SfcPres_quantile', 'f4', ['probability'], fill_value = -9999)
    varpsfc[:] = psfcqs
    varpsfc.long_name = 'Surface pressure quantiles'
    varpsfc.units = 'hPa'
    varpsfc.missing_value = -9999.

    varsalt = qout.createVariable('SfcAlt_quantile', 'f4', ['probability'], fill_value = -9999)
    varsalt[:] = sfalqs
    varsalt.long_name = 'Surface altitude quantiles'
    varsalt.units = 'm'
    varsalt.missing_value = -9999.

    qout.close()


    # Output transformed quantile samples
    zfnm = '%s/CONUS_AIRS_JJA_%04d_%02dUTC_%s_TempRHSfc_StdGausTrans.nc' % (dtdr,yrlst[k],hrchc,rgchc)
    zout = Dataset(zfnm,'w') 
    dimz = zout.createDimension('level',nzout)
    dimsmp = zout.createDimension('sample',tsmp)

    varlvl = zout.createVariable('level','f4',['level'], fill_value = -9999)
    varlvl[:] = plev
    varlvl.long_name = 'AIRS/SARTA pressure levels'
    varlvl.units = 'hPa'
    varlvl.missing_value = -9999

    varlon = zout.createVariable('Longitude','f4',['sample'])
    varlon[:] = lonout
    varlon.long_name = 'Longitude'
    varlon.units = 'degrees_east'

    varlat = zout.createVariable('Latitude','f4',['sample'])
    varlat[:] = latout
    varlat.long_name = 'Latitude'
    varlat.units = 'degrees_north'

    varjdy = zout.createVariable('JulianDay','i2',['sample'])
    varjdy[:] = jdyout
    varjdy.long_name = 'JulianDay'
    varjdy.units = 'day'

    varyr = zout.createVariable('Year','i2',['sample'])
    varyr[:] = yrout
    varyr.long_name = 'Year'
    varyr.units = 'year'

    varsrt3 = zout.createVariable('Temperature_StdGaus', 'f4', ['sample','level'], fill_value = -9999)
    varsrt3[:] = ztmpout
    varsrt3.long_name = 'Quantile transformed temperature'
    varsrt3.units = 'None'
    varsrt3.missing_value = -9999.

    varsrt4 = zout.createVariable('RH_StdGaus', 'f4', ['sample','level'], fill_value = -9999)
    varsrt4[:] = zrhout
    varsrt4.long_name = 'Quantile transformed relative humidity'
    varsrt4.units = 'None'
    varsrt4.missing_value = -9999.

    varsrts1 = zout.createVariable('SfcTemp_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts1[:] = zsftmpout
    varsrts1.long_name = 'Quantile transformed surface temperature'
    varsrts1.units = 'None'
    varsrts1.missing_value = -9999.

    varsrts2 = zout.createVariable('SfcPres_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts2[:] = zpsfcout
    varsrts2.long_name = 'Quantile transformed surface pressure'
    varsrts2.units = 'None'
    varsrts2.missing_value = -9999.

    varsrts3 = zout.createVariable('SfcAlt_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts3[:] = zsfaltout
    varsrts3.long_name = 'Quantile transformed surface pressure'
    varsrts3.units = 'None'
    varsrts3.missing_value = -9999.

    zout.close()

    return

def airscld_invtransf_mix_cloud9_conus_nosfc(rfdr, dtdr, yrchc, hrchc, rgchc, rfmn, rfdy, rfgrn, scnrw, nrep = 10, \
                                             l2dir = '/archive/AIRSOps/airs/gdaac/v6'):
    # Read in mixture model parameters, draw random samples and set up SARTA input files
    # Use AIRS FOV cloud fraction information
    # Use designated AIRS reference granule, and pull surface pressure temperature from there 
    # dtdr:    Output directory
    # yrchc:   Template Year Choice
    # hrchc:   Template Hour Choice
    # rgchc:   Template Region Choice
    # rfmn:    Month for reference granule
    # rfdy:    Day for reference granule
    # rfgrn:   Reference granule number
    # scnrw:   Scan row for experiment
    # nrep:    Number of replicate granules
    # l2dir:   Local AIRS Level 2 directory (to retrieve reference info)

    # RN Generator
    sdchc = 165434 + yrchc + hrchc
    random.seed(sdchc)
    cldprt = numpy.array([0.4,0.2,0.08])

    nszout = 45 * 30 * nrep
    sfrps = 45 * nrep
    nlvsrt = 98
    msgdbl = -9999.0

    # Read probs and pressure levels
    rnm = '%s/AIRS_Levels_Quantiles.nc' % (rfdr)
    f = Dataset(rnm,'r')
    airs_sarta_levs = f['level'][:]
    f.close()

    # Get reference granule info
    airsdr = '%s/%04d/%02d/%02d/airs2sup' % (l2dir,yrchc,rfmn,rfdy)
    if (os.path.exists(airsdr)):
        fllst = os.listdir(airsdr)
        l2str = 'AIRS.%04d.%02d.%02d.%03d' % (yrchc,rfmn,rfdy,rfgrn) 
        rffd = -1
        j = 0
        while ( (j < len(fllst)) and (rffd < 0) ):
            lncr = len(fllst[j])
            l4 = lncr - 4
            if ( (fllst[j][l4:lncr] == '.hdf') and (l2str in fllst[j])):
                l2fl = '%s/%s' % (airsdr,fllst[j])
                ncl2 = Dataset(l2fl)
                psfc = ncl2.variables['PSurfStd'][:,:]
                topg = ncl2.variables['topog'][:,:]
                ncl2.close()
                rffd = j
            j = j + 1
    else:
        print('L2 directory not found')

    # Surface replicates
    psfcvc = psfc[scnrw-1,:]
    topgvc = topg[scnrw-1,:]

    spres = numpy.tile(psfcvc,(sfrps,))
    salti = numpy.tile(topgvc,(sfrps,))

    # Variable list
    clrlst = ['Temperature','RH','SfcTemp']
    clrst = [1,64,0]
    clrct = [98,35,1]
    cldlst = ['NumberSlab','CloudBot1Logit','DPCloud1Logit','DPSlabLogit','DPCloud2Logit', \
              'CType1','CType2','CCoverInd','CFrcLogit1','CFrcLogit2','CFrcLogit12', \
              'NGWater1','NGWater2','CTTemp1','CTTemp2']
    cldst = [0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0]
    cldct = [1,1,1,1,1, 1,1,9,9,9,9, 1,1,1,1]
    nvar = 0 
    for q in range(len(clrct)):
        nvar = nvar + clrct[q]
    nclr = nvar
    for q in range(len(cldlst)):
        nvar = nvar + cldct[q]
    ncld = nvar - nclr

    # Discrete/Continuous Indicator
    typind = []
    for q in range(len(clrct)):
        for p in range(clrct[q]):
            typind.append('Continuous')
    cldtypind = ['Discrete','Continuous','Continuous','Continuous','Continuous', \
                 'Discrete','Discrete','Discrete','Continuous','Continuous','Continuous', \
                 'Continuous','Continuous','Continuous','Continuous']
    for q in range(len(cldct)):
        for p in range(cldct[q]):
            typind.append(cldtypind[q])

    # Quantile files 
    qclrnm = '%s/CONUS_AIRS_JJA_%04d_%02dUTC_%s_TempRHSfc_Quantile.nc' % (dtdr,yrchc,hrchc,rgchc)
    qcldnm = '%s/CONUS_AIRS_JJA_%04d_%02dUTC_%s_Cloud_Quantile.nc' % (dtdr,yrchc,hrchc,rgchc)

    qin = Dataset(qclrnm,'r')
    prbs = qin.variables['probability'][:]
    nprb = prbs.shape[0]
    qsclr = numpy.zeros((nclr,nprb))
    lvs = qin.variables['level'][:]
    alts = qin.variables['Altitude_median'][:]
    rhmd = qin.variables['RH_quantile'][:,103]
    nlvl = lvs.shape[0]
    cctr = 0
    for j in range(len(clrlst)):
        print(clrlst[j])
        if clrst[j] == 0:
            vr1 = '%s_quantile' % (clrlst[j])
            qsclr[cctr,:] = qin.variables[vr1][:]
        else:
            inst = clrst[j] - 1
            infn = inst + clrct[j]
            otst = cctr
            otfn = cctr + clrct[j]
            vr1 = '%s_quantile' % (clrlst[j])
            qsclr[otst:otfn,:] = qin.variables[vr1][inst:infn,:]
        cctr = cctr + clrct[j]
    qin.close()
    print('Clear medians')
    print(qsclr[:,103])

    cldnmout = []
    qin = Dataset(qcldnm,'r')
    qscld = numpy.zeros((ncld,nprb))
    dctr = 0
    for j in range(len(cldlst)):
        print(cldlst[j])
        vr1 = '%s_quantile' % (cldlst[j])
        vrinf = qin.variables[vr1]
        if cldct[j] == 1:
            qscld[dctr,:] = qin.variables[vr1][:]
            dctr = dctr + 1
            cldnmout.append(cldlst[j])
        elif (len(vrinf.shape) == 2):
            inst = cldst[j]
            infn = inst + cldct[j]
            for n2 in range(inst,infn):
                clnm = '%s_%d' % (cldlst[j],n2)
                cldnmout.append(clnm)
            otst = dctr
            otfn = dctr + cldct[j]
            vr1 = '%s_quantile' % (clrlst[j])
            qscld[otst:otfn,:] = qin.variables[vr1][inst:infn,:]
            dctr = dctr + cldct[j]
        elif (len(vrinf.shape) == 3):
            for cl0 in range(vrinf.shape[0]):
                for rw0 in range(vrinf.shape[1]):
                    otst = dctr
                    otfn = dctr + 1
                    qscld[otst:otfn,:] = qin.variables[vr1][cl0,rw0,:]
                    clnm = '%s_%d_%d' % (cldlst[j],cl0,rw0)
                    cldnmout.append(clnm)
                    dctr = dctr + 1
    qin.close()
    print('Cloud medians')
    print(qscld[:,103])

    # Read GMM Results
    gmmnm = '%s/CONUS_AIRS_JJA_%04d_%02dUTC_%s_GMM_parameters.nc' % (dtdr,yrchc,hrchc,rgchc)
    gmin = Dataset(gmmnm,'r')
    gmnms = gmin['State_Vector_Names'][:,:]
    gmmean = gmin['Mean'][:,:]
    gmpkcv = gmin['Packed_Covariance'][:,:]
    gmprps = gmin['Mixture_Proportion'][:]
    gmin.close()

    nmclps = gmnms.tolist()
    strvrs = list(map(calculate_VPD.clean_byte_list,nmclps))
    if sys.version_info[0] < 3:
        print('Version 2')
        strvrs = map(str,strvrs)
    nmix = gmmean.shape[0]
    nmxvar = gmmean.shape[1]

    mrgcv = numpy.zeros((nmix,nmxvar,nmxvar),dtype=numpy.float64)
    for j in range(nmix):
        mrgcv[j,:,:] = calculate_VPD.unpackcov(gmpkcv[j,:], nelm=nmxvar)

    # Component sizes
    dtall = numpy.zeros((nszout,nmxvar),dtype=numpy.float)
    cmpidx = numpy.zeros((nszout,),dtype=numpy.int16)
    csmp = random.multinomial(nszout,pvals=gmprps)
    cmsz = 0
    for j in range(nmix):
        cvfl = mrgcv[j,:,:]
        s1 = numpy.sqrt(numpy.diagonal(cvfl))
        crmt = calculate_VPD.cov2cor(cvfl)
        sdmt = numpy.diag(numpy.sqrt(cvfl.diagonal()))
        w, v = linalg.eig(crmt)
        print(numpy.amin(w))

        sdfn = cmsz + csmp[j]
        dtz = random.multivariate_normal(numpy.zeros((nmxvar,)),crmt,size=csmp[j])
        dttmp = numpy.tile(gmmean[j,:],(csmp[j],1)) + numpy.dot(dtz,sdmt)
        dtall[cmsz:sdfn,:] = dttmp[:,:]
        cmpidx[cmsz:sdfn] = j + 1

        cmsz = cmsz + csmp[j]

    # Re-shuffle
    ssq = numpy.arange(nszout)
    sqsmp = random.choice(ssq,size=nszout,replace=False)
    csmpshf = cmpidx[sqsmp]
    dtshf = dtall[sqsmp,:] 
    print(dtshf.shape) 

    ### Inverse Transform
    qout = numpy.zeros(dtshf.shape)
    for j in range(nclr):
        if typind[j] == 'Discrete':
            qout[:,j] = calculate_VPD.data_quantile_from_std_norm_discrete(dtshf[:,j],qsclr[j,:],prbs,minval=qsclr[j,0],maxval=qsclr[j,nprb-1])
        else:
            qout[:,j] = calculate_VPD.data_quantile_from_std_norm(dtshf[:,j],qsclr[j,:],prbs,minval=qsclr[j,0],maxval=qsclr[j,nprb-1])
    for j in range(nclr,nvar):
        if typind[j] == 'Discrete':
            qout[:,j] = calculate_VPD.data_quantile_from_std_norm_discrete(dtshf[:,j],qscld[j-nclr,:],prbs,minval=qsclr[j-nclr,0],maxval=qscld[j-nclr,nprb-1])
        else:
            qout[:,j] = calculate_VPD.data_quantile_from_std_norm(dtshf[:,j],qscld[j-nclr,:],prbs,minval=qscld[j-nclr,0],maxval=qsclr[j-nclr,nprb-1])

    ### Prepare for SARTA
    varlstout = ['cngwat','cngwat2','cprbot','cprbot2','cprtop','cprtop2', \
                 'cpsize','cpsize2','cstemp','cstemp2','ctype','ctype2','salti','spres','stemp']
    # Adjust altitudes
    alth2o = numpy.zeros((nszout,nlvsrt+3))
    alth2o[:,nlvsrt-1] = alts[nlvsrt-1]
    curdlt = 0.0
    for j in range(nlvsrt-2,-1,-1):
        str1 = 'Level %d: %.4f' % (j,curdlt)
        print(str1)
        if (alts[j] > alts[j+1]):
            curdlt = alts[j] - alts[j+1]
            alth2o[:,j] = alts[j]
        else:
            alth2o[:,j] = alts[j+1] + curdlt * 2.0
            curdlt = curdlt * 2.0
    alth2o[:,97] = 0.0

    # Convert cloud items to data frame
    smpfrm = pandas.DataFrame(data=qout[:,nclr:nvar],columns=cldnmout)
 
    dtout = numpy.zeros((nszout,len(varlstout)), dtype=numpy.float64)
    frmout = pandas.DataFrame(data=dtout,columns=varlstout)

    # Cloud Types
    frmout['ctype'] = (smpfrm['CType1'] + 1.0) * 100.0 + 1.0
    frmout['ctype2'] = (smpfrm['CType2'] + 1.0) * 100.0 + 1.0
    frmout.loc[(smpfrm.NumberSlab == 0),'ctype'] = msgdbl
    frmout.loc[(smpfrm.NumberSlab < 2),'ctype2'] = msgdbl 

    # Met/Sfc Components, arrays sized for SARTA and AIRS
    cctr = 0
    prhout = numpy.zeros((nszout,nlvsrt+3)) - 9999.0
    ptmpout = numpy.zeros((nszout,nlvsrt+3)) - 9999.0
    for j in range(len(clrst)):
        if clrst[j] == 0:
            if clrlst[j] == 'SfcTemp':
                frmout['stemp'] = qout[:,cctr]
        elif clrlst[j] == 'Temperature':
            inst = clrst[j] - 1
            infn = inst + clrct[j]
            otst = cctr
            otfn = cctr + clrct[j]
            ptmpout[:,inst:infn] = qout[:,otst:otfn]
        elif clrlst[j] == 'RH':
            inst = clrst[j] - 1
            infn = inst + clrct[j]
            otst = cctr
            otfn = cctr + clrct[j]
            prhout[:,inst:infn] = qout[:,otst:otfn]
            bsrh = rhmd[inst]
            for k in range(inst-1,-1,-1):
                if ma.is_masked(rhmd[k]):
                    prhout[:,k] = bsrh / 2.0
                    t2 = 'RH masked: %d' % (k)
                    print(t2)
                elif rhmd[k] < 0:
                    t2 = 'RH below 0: %d' % (k)
                    print(t2)
                    prhout[:,k] = bsrh
                else:
                    prhout[:,k] = rhmd[k]
                    bsrh = rhmd[k]
        cctr = cctr + clrct[j]
    str1 = '''RH at Level 1: %.4e, %.4e ''' % (numpy.amin(prhout[:,0]),rhmd[0])
    str2 = '''RH at Level 2: %.4e, %.4e ''' % (numpy.amin(prhout[:,1]),rhmd[1])
    print(str1)
    print(str2)
    h2oout = calculate_VPD.calculate_h2odens(prhout,ptmpout,airs_sarta_levs,alth2o)

    # Surface from reference
    frmout['salti'] = salti
    # Need for clouds
    frmout['spres'] = spres 
    smpfrm['SfcPres'] = spres 

    # Pressure Variables
    for i in range(nszout):
        if smpfrm['NumberSlab'][smpfrm.index[i]] == 0:
            frmout.at[i,'cprbot'] = msgdbl 
            frmout.at[i,'cprtop'] = msgdbl
            frmout.at[i,'cprbot2'] = msgdbl 
            frmout.at[i,'cprtop2'] = msgdbl
        elif smpfrm['NumberSlab'][smpfrm.index[i]] == 1:
            tmplgts = numpy.array( [smpfrm['CloudBot1Logit'][smpfrm.index[i]], \
                                    smpfrm['DPCloud1Logit'][smpfrm.index[i]] ] )
            frctmp = calculate_VPD.lgttoprp(tmplgts)
            frmout.at[i,'cprbot'] = smpfrm['SfcPres'][smpfrm.index[i]] * (1.0 - frctmp[0])
            frmout.at[i,'cprtop'] = smpfrm['SfcPres'][smpfrm.index[i]] * (1.0 - frctmp[0] - frctmp[1])
            frmout.at[i,'cprbot2'] = msgdbl 
            frmout.at[i,'cprtop2'] = msgdbl
        elif smpfrm['NumberSlab'][smpfrm.index[i]] == 2:
            tmplgts = numpy.array( [smpfrm['CloudBot1Logit'][smpfrm.index[i]], \
                                    smpfrm['DPCloud1Logit'][smpfrm.index[i]], \
                                    smpfrm['DPSlabLogit'][smpfrm.index[i]], \
                                    smpfrm['DPCloud2Logit'][smpfrm.index[i]] ] )
            frctmp = calculate_VPD.lgttoprp(tmplgts)
            frmout.at[i,'cprbot'] = smpfrm['SfcPres'][smpfrm.index[i]] * (1.0 - frctmp[0])
            frmout.at[i,'cprtop'] = smpfrm['SfcPres'][smpfrm.index[i]] * (1.0 - frctmp[0] - frctmp[1])
            frmout.at[i,'cprbot2'] = smpfrm['SfcPres'][smpfrm.index[i]] * (1.0 - frctmp[0] - frctmp[1] - frctmp[2])
            frmout.at[i,'cprtop2'] = smpfrm['SfcPres'][smpfrm.index[i]] * (1.0 - frctmp[0] - frctmp[1] - frctmp[2] - frctmp[3])

    # Non-Gas Water
    frmout['cngwat'] = smpfrm['NGWater1']
    frmout.loc[(smpfrm.NumberSlab == 0),'cngwat'] = msgdbl
    frmout['cngwat2'] = smpfrm['NGWater2']
    frmout.loc[(smpfrm.NumberSlab < 2),'cngwat2'] = msgdbl

    # Temperature
    frmout['cstemp'] = smpfrm['CTTemp1']
    frmout.loc[(smpfrm.NumberSlab == 0),'cstemp'] = msgdbl
    frmout['cstemp2'] = smpfrm['CTTemp2']
    frmout.loc[(smpfrm.NumberSlab < 2),'cstemp2'] = msgdbl

    # Particle Size, from Sergio's paper
    # 20 for water, 80 for ice
             #'cpsize','cpsize2','cstemp','cstemp2','ctype','ctype2']
    frmout.loc[(frmout.ctype == 101.0),'cpsize'] = 20 
    frmout.loc[(frmout.ctype == 201.0),'cpsize'] = 80
    frmout.loc[(frmout.ctype < 0.0),'cpsize'] = msgdbl

    frmout.loc[(frmout.ctype2 == 101.0),'cpsize2'] = 20 
    frmout.loc[(frmout.ctype2 == 201.0),'cpsize2'] = 80
    frmout.loc[(frmout.ctype2 < 0.0),'cpsize2'] = msgdbl

    # Fractions, 3D Arrays
    cfrc1out = numpy.zeros((nszout,3,3)) - 9999.0
    cfrc2out = numpy.zeros((nszout,3,3)) - 9999.0
    cfrc12out = numpy.zeros((nszout,3,3)) - 9999.0
    for i in range(nszout):
        if smpfrm['NumberSlab'][smpfrm.index[i]] == 0:
            cfrc1out[i,:,:] = 0.0
            cfrc2out[i,:,:] = 0.0
            cfrc12out[i,:,:] = 0.0
        elif smpfrm['NumberSlab'][smpfrm.index[i]] == 1:
            for q in range(3):
                for p in range(3):
                    ccvnm = 'CCoverInd_%d_%d' % (q,p)
                    lgtnm1 = 'CFrcLogit1_%d_%d' % (q,p)
                    if (smpfrm[ccvnm][smpfrm.index[i]] == -1):
                        cfrc1out[i,q,p] = 0.0
                    elif (smpfrm[ccvnm][smpfrm.index[i]] == 1):
                        cfrc1out[i,q,p] = 1.0
                    else:
                        tmplgts = numpy.array( [smpfrm[lgtnm1][smpfrm.index[i]]] )
                        frctmp = calculate_VPD.lgttoprp(tmplgts)
                        cfrc1out[i,q,p] = frctmp[0]
            cfrc2out[i,:,:] = 0.0
            cfrc12out[i,:,:] = 0.0
        elif smpfrm['NumberSlab'][smpfrm.index[i]] == 2:
            for q in range(3):
                for p in range(3):
                    ccvnm = 'CCoverInd_%d_%d' % (q,p)
                    lgtnm1 = 'CFrcLogit1_%d_%d' % (q,p)
                    lgtnm2 = 'CFrcLogit2_%d_%d' % (q,p)
                    lgtnm12 = 'CFrcLogit12_%d_%d' % (q,p)
                    if (smpfrm[ccvnm][smpfrm.index[i]] == -1):
                        cfrc1out[i,q,p] = 0.0
                        cfrc2out[i,q,p] = 0.0
                        cfrc12out[i,q,p] = 0.0
                    elif (smpfrm[ccvnm][smpfrm.index[i]] == 1):
                        tmplgts = numpy.array( [smpfrm[lgtnm1][smpfrm.index[i]], \
                                                smpfrm[lgtnm2][smpfrm.index[i]], \
                                                smpfrm[lgtnm12][smpfrm.index[i]]] ) 
                        frctmp = calculate_VPD.lgttoprp(tmplgts)
                        frcadj = 1.0 - frctmp[3]
                        cfrc1out[i,q,p] = (frctmp[0] + frctmp[2]) / frcadj
                        cfrc2out[i,q,p] = (frctmp[1] + frctmp[2]) / frcadj
                        cfrc12out[i,q,p] = frctmp[2] / frcadj
                    else:
                        tmplgts = numpy.array( [smpfrm[lgtnm1][smpfrm.index[i]], \
                                                smpfrm[lgtnm2][smpfrm.index[i]], \
                                                smpfrm[lgtnm12][smpfrm.index[i]]] ) 
                        frctmp = calculate_VPD.lgttoprp(tmplgts)
                        cfrc1out[i,q,p] = frctmp[0] + frctmp[2]
                        cfrc2out[i,q,p] = frctmp[1] + frctmp[2]
                        cfrc12out[i,q,p] = frctmp[2]

    # Write Sample Output
    print(frmout[166:180])

    fldbl = numpy.array([-9999.],dtype=numpy.float64)
    flflt = numpy.array([-9999.],dtype=numpy.float32)
    flshrt = numpy.array([-99],dtype=numpy.int16)

    dfnm = '%s/SampledStateVectors/CONUS_AIRS_JJA_%04d_%02dUTC_%s_SR%02d_SimSARTAStates_Mix_CloudFOV.h5' % (dtdr,yrchc,hrchc,rgchc,scnrw)
    f = h5py.File(dfnm,'w')
    for j in range(len(varlstout)): 
        dftmp = f.create_dataset(varlstout[j],data=frmout[varlstout[j]])
        dftmp.attrs['missing_value'] = -9999.
        dftmp.attrs['_FillValue'] = -9999.
    dfpt = f.create_dataset('ptemp',data=ptmpout)
    dfpt.attrs['missing_value'] = fldbl 
    dfpt.attrs['_FillValue'] = fldbl
    dfrh = f.create_dataset('relative_humidity',data=prhout)
    dfrh.attrs['missing_value'] = fldbl
    dfrh.attrs['_FillValue'] = fldbl
    dfgs = f.create_dataset('gas_1',data=h2oout)
    dfgs.attrs['missing_value'] = fldbl
    dfgs.attrs['_FillValue'] = fldbl
    dfcf1 = f.create_dataset('cfrac',data=cfrc1out)
    dfcf1.attrs['missing_value'] = fldbl
    dfcf1.attrs['_FillValue'] = fldbl
    dfcf2 = f.create_dataset('cfrac2',data=cfrc2out)
    dfcf2.attrs['missing_value'] = fldbl
    dfcf2.attrs['_FillValue'] = fldbl
    dfcf12 = f.create_dataset('cfrac12',data=cfrc12out)
    dfcf12.attrs['missing_value'] = fldbl
    dfcf12.attrs['_FillValue'] = fldbl
    dfcsmp = f.create_dataset('mixture_component',data=csmpshf)
    dfcsmp.attrs['missing_value'] = flshrt 
    dfcsmp.attrs['_FillValue'] = flshrt
    dflv = f.create_dataset('level',data=airs_sarta_levs)
    f.close()

    return

def setup_airs_cloud(flnm, tms, lats, lons, tmunit = 'Seconds since 1993-01-01 00:00:00'): 
    # Set up matched AIRS/MERRA cloud file 
    # flnm:    Name of output file
    # tms:     Time variable array 
    # lats:    Latitude variable array 
    # lons:    Longitude variable array

    ntm = tms.shape[0]
    nlat = lats.shape[0]
    nlon = lons.shape[0]

    # Create Output file 
    qout = Dataset(flnm,'w') 

    dimln = qout.createDimension('lon',nlon)
    dimlt = qout.createDimension('lat',nlat)
    dimtm = qout.createDimension('time',ntm)
    dimtrk = qout.createDimension('AIRSFOV',9)

    if (lons.dtype == numpy.float32):
        lntp = 'f4'
    else:
        lntp = 'f8'
    varlon = qout.createVariable('lon',lntp,['lon'], fill_value = -9999)
    varlon[:] = lons
    varlon.long_name = 'longitude'
    varlon.units='degrees_east'
    varlon.missing_value = -9999

    if (lats.dtype == numpy.float32):
        lttp = 'f4'
    else:
        lttp = 'f8'
    varlat = qout.createVariable('lat',lttp,['lat'], fill_value = -9999)
    varlat[:] = lats
    varlat.long_name = 'latitude'
    varlat.units='degrees_north'
    varlat.missing_value = -9999

    if (tms.dtype == numpy.float32):
        tmtp = 'f4'
    else:
        tmtp = 'f8'
    vartm = qout.createVariable('time',lttp,['time'], fill_value = -9999)
    vartm[:] = tms
    vartm.long_name = 'time'
    vartm.units = tmunit
    vartm.missing_value = -9999

    # Other output variables
    varcfrc1 = qout.createVariable('AIRS_CldFrac_1','f4',['time','lat','lon','AIRSFOV'], fill_value = -9999)
    varcfrc1.long_name = 'AIRS cloud fraction, upper level'
    varcfrc1.units = 'unitless'
    varcfrc1.missing_value = -9999

    varcfrc2 = qout.createVariable('AIRS_CldFrac_2','f4',['time','lat','lon','AIRSFOV'], fill_value = -9999)
    varcfrc2.long_name = 'AIRS cloud fraction, lower level'
    varcfrc2.units = 'unitless'
    varcfrc2.missing_value = -9999

    varcqc1 = qout.createVariable('AIRS_CldFrac_QC_1','i2',['time','lat','lon','AIRSFOV'], fill_value = -99)
    varcqc1.long_name = 'AIRS cloud fraction quality flag, upper level'
    varcqc1.units = 'unitless'
    varcqc1.missing_value = -9999

    varcqc2 = qout.createVariable('AIRS_CldFrac_QC_2','i2',['time','lat','lon','AIRSFOV'], fill_value = -99)
    varcqc2.long_name = 'AIRS cloud fraction quality flag, lower level'
    varcqc2.units = 'unitless'
    varcqc2.missing_value = -9999

    varncld = qout.createVariable('AIRS_nCld','i2',['time','lat','lon','AIRSFOV'], fill_value = -99)
    varncld.long_name = 'AIRS number of cloud layers'
    varncld.units = 'unitless'
    varncld.missing_value = -9999

    qout.close()

    return

def airs_cfrac_match_merra(flnm, tmidx, tmday, lats, lons,  msgvl = -9999, \
                           l2srch = '/archive/AIRSOps/airs/gdaac/v6'): 
    # Set up matched AIRS/MERRA cloud file 
    # flnm:    Name of output file
    # tms:     Time index in output 
    # tmday:   Datetime object with time information
    # lats:    Longitude variable array
    # lons:    Longitude variable array

    # Search AIRS Level 2
    airsdr = '%s/%04d/%02d/%02d/airs2ret' % (l2srch,tmday.year,tmday.month,tmday.day)

    dsclst = []
    asclst = []

    nlat = lats.shape[0]
    nlon = lons.shape[0]

    lonmn = lons[0] - 5.0
    lonmx = lons[nlon-1] + 5.0
    latmn = lats[0] - 5.0
    latmx = lats[nlat-1] + 5.0
    d0 = datetime.datetime(1993,1,1,0,0,0)
    ddif = tmday - d0
    bsdif = ddif.total_seconds()

    # Set up reference frame
    ltrp = numpy.repeat(lats,nlon)
    ltidx = numpy.repeat(numpy.arange(nlat),nlon)
    lnrp = numpy.tile(lons,nlat)
    lnidx = numpy.tile(numpy.arange(nlon),nlat)
    merfrm = pandas.DataFrame({'GridLonIdx': lnidx, 'GridLatIdx': ltidx, \
                               'GridLon': lnrp, 'GridLat': ltrp})

    if (os.path.exists(airsdr)):
        fllst = os.listdir(airsdr)
        #print(fllst)

        for j in range(len(fllst)):
            lncr = len(fllst[j])
            l4 = lncr - 4
            if (fllst[j][l4:lncr] == '.hdf'):
                l2fl = '%s/%s' % (airsdr,fllst[j])
                ncl2 = Dataset(l2fl)
                slrzn = ncl2.variables['solzen'][:,:]
                l2lat = ncl2.variables['Latitude'][:,:]
                l2lon = ncl2.variables['Longitude'][:,:]
                l2tm = ncl2.variables['Time'][:,:]
                ncl2.close()

                # Check lat/lon ranges and asc/dsc
                l2tmdf = numpy.absolute(l2tm - bsdif)
                l2mntm = numpy.min(l2tmdf)

                # Within 4 hours
                if l2mntm < 14400.0:
                   ltflt = l2lat.flatten()
                   lnflt = l2lon.flatten()
                   latsb = ltflt[(ltflt >= latmn) & (ltflt <= latmx)]
                   lonsb = lnflt[(lnflt >= lonmn) & (lnflt <= lonmx)]
                   if ( (latsb.shape[0]  > 0) and (lonsb.shape[0] > 0) ):
                       asclst.append(fllst[j])
                       sstr = '%s %.2f' % (fllst[j], l2mntm)
                       print(sstr)
                

    # Set up outputs
    cld1arr = numpy.zeros((nlat,nlon,9),dtype=numpy.float32) + msgvl
    cld2arr = numpy.zeros((nlat,nlon,9),dtype=numpy.float32) + msgvl
    cld1qc = numpy.zeros((nlat,nlon,9),dtype=numpy.int16) - 99
    cld2qc = numpy.zeros((nlat,nlon,9),dtype=numpy.int16) - 99
    ncldarr = numpy.zeros((nlat,nlon,9),dtype=numpy.int16) - 99

    #print(asclst)
    if (len(asclst) > 0):
        # Start matchups
        for j in range(len(asclst)):
            l2fl = '%s/%s' % (airsdr,asclst[j])
            ncl2 = Dataset(l2fl)
            l2lat = ncl2.variables['Latitude'][:,:]
            l2lon = ncl2.variables['Longitude'][:,:]
            cfrcair = ncl2.variables['CldFrcStd'][:,:,:,:,:]
            cfrcaqc = ncl2.variables['CldFrcStd_QC'][:,:,:,:,:]
            ncldair = ncl2.variables['nCld'][:,:,:,:]
            ncl2.close()

            nairtrk = l2lat.shape[0]
            nairxtk = l2lat.shape[1]

            # Data Frame
            tkidx = numpy.repeat(numpy.arange(nairtrk),nairxtk)
            xtidx = numpy.tile(numpy.arange(nairxtk),nairtrk)
            l2lnflt = l2lon.flatten().astype(numpy.float64)
            l2ltflt = l2lat.flatten().astype(numpy.float64)
            l2frm = pandas.DataFrame({'L2LonIdx': xtidx, 'L2LatIdx': tkidx, \
                                      'L2Lon': l2lnflt, 'L2Lat': l2ltflt})
            l2frm['GridLon'] = numpy.around(l2frm['L2Lon']/0.625) * 0.625 
            l2frm['GridLat'] = numpy.around(l2frm['L2Lat']/0.5) * 0.5

            l2mrg = pandas.merge(l2frm,merfrm,on=['GridLon','GridLat'])
            print(l2mrg.shape)

            #if j  == 0:
            #    print(asclst[j])
            #    print(l2mrg[0:15])

            # Output data if available
            for k in range(l2mrg.shape[0]):
                yidxout = l2mrg['GridLatIdx'].values[k]
                xidxout = l2mrg['GridLatIdx'].values[k]
                yidxl2 = l2mrg['L2LatIdx'].values[k]
                xidxl2 = l2mrg['L2LonIdx'].values[k]
                cld1arr[yidxout,xidxout,:] = cfrcair[yidxl2,xidxl2,:,:,0].flatten().astype(numpy.float32)
                cld2arr[yidxout,xidxout,:] = cfrcair[yidxl2,xidxl2,:,:,1].flatten().astype(numpy.float32)
                cld1qc[yidxout,xidxout,:] = cfrcaqc[yidxl2,xidxl2,:,:,0].flatten().astype(numpy.int16)
                cld2qc[yidxout,xidxout,:] = cfrcaqc[yidxl2,xidxl2,:,:,1].flatten().astype(numpy.int16)
                ncldarr[yidxout,xidxout,:] = ncldair[yidxl2,xidxl2,:,:].flatten().astype(numpy.int16)

    # Output
    qout = Dataset(flnm,'r+') 

    varcfrc1 = qout.variables['AIRS_CldFrac_1']
    varcfrc1[tmidx,:,:,:] = cld1arr[:,:,:]

    varcfrc2 = qout.variables['AIRS_CldFrac_2']
    varcfrc2[tmidx,:,:,:] = cld2arr[:,:,:]

    varcfqc1 = qout.variables['AIRS_CldFrac_QC_1']
    varcfqc1[tmidx,:,:,:] = cld1qc[:,:,:]

    varcfqc2 = qout.variables['AIRS_CldFrac_QC_2']
    varcfqc2[tmidx,:,:,:] = cld2qc[:,:,:]

    varncld = qout.variables['AIRS_nCld']
    varncld[tmidx,:,:,:] = ncldarr[:,:,:]

    qout.close()

    return

def quantile_allstate_locmask_conus(rfdr, mtlst, cslst, airslst, dtdr, yrlst, mnst, mnfn, hrchc, rgchc, sstr, mskvr, mskvl):
    # Construct quantiles and z-scores, with a possibly irregular location mask,
    # for joint atmospheric state (AIRS/SARTA)
    # rfdr:    Directory for reference data (Levels/Quantiles)
    # mtlst:   Meteorology (MERRA) file list
    # cslst:   Cloud slab file list 
    # airslst: AIRS cloud fraction file list 
    # dtdr:    Output directory
    # yrlst:   List of years to process
    # mnst:    Starting Month
    # mnfn:    Ending Month 
    # hrchc:   Template Hour Choice
    # rgchc:   Template Region Choice
    # sstr:    Season string
    # mskvr:   Name of region mask variable
    # mskvl:   Value of region mask for Region Choice

    # Read probs and pressure levels
    rnm = '%s/AIRS_Levels_Quantiles.nc' % (rfdr)
    f = Dataset(rnm,'r')
    plev = f['level'][:]
    prbs = f['probability'][:]
    alts = f['altitude'][:]
    f.close()

    nyr = len(yrlst)
    nprb = prbs.shape[0]

    # RN generator
    sdchc = 542354 + yrlst[0] + hrchc
    random.seed(sdchc)

    # Mask, lat, lon
    f = Dataset(mtlst[0],'r')
    mask = f.variables[mskvr][:,:]
    latmet = f.variables['lat'][:]
    lonmet = f.variables['lon'][:]
    tminf = f.variables['time'][:]
    tmunit = f.variables['time'].units[:]
    f.close()

    mskind = numpy.zeros((mask.shape),dtype=mask.dtype)
    print(mskvl)
    mskind[mask == mskvl] = 1
    lnsq = numpy.arange(lonmet.shape[0])
    ltsq = numpy.arange(latmet.shape[0])

    # Subset a bit
    lnsm = numpy.sum(mskind,axis=0)
    ltsm = numpy.sum(mskind,axis=1)

    lnmn = numpy.amin(lnsq[lnsm > 0])
    lnmx = numpy.amax(lnsq[lnsm > 0]) + 1
    ltmn = numpy.amin(ltsq[ltsm > 0])
    ltmx = numpy.amax(ltsq[ltsm > 0]) + 1

    stridx = 'Lon Range: %d, %d\nLat Range: %d, %d \n' % (lnmn,lnmx,ltmn,ltmx)
    print(stridx)

    nx = lnmx - lnmn
    ny = ltmx - ltmn 
    nzout = 101

    lnrp = numpy.tile(lonmet[lnmn:lnmx],ny)
    ltrp = numpy.repeat(latmet[ltmn:ltmx],nx)
    mskblk = mskind[ltmn:ltmx,lnmn:lnmx]
    mskflt = mskblk.flatten()


    tsmp = 0
    for k in range(nyr):
        f = Dataset(mtlst[k],'r')
        tminf = f.variables['time'][:]
        tmunit = f.variables['time'].units[:]
        f.close()

        tmunit = tmunit.replace("days since ","")
        dybs = datetime.datetime.strptime(tmunit,"%Y-%m-%d %H:%M:%S")
        print(dybs)
        dy0 = dybs + datetime.timedelta(days=tminf[0]) 
        dyinit = datetime.date(dy0.year,dy0.month,dy0.day)
        print(dyinit)
 
        dyst = datetime.date(yrlst[k],mnst,1)
        ttst = dyst.timetuple()
        jst = ttst.tm_yday
        if mnfn < mnst:
            dyfn = datetime.date(yrlst[k]+1,mnfn+1,1)
            ttfn = dyfn.timetuple()
            jfn = ttfn.tm_yday
            dy31 = datetime.date(yrlst[k],12,31)
            tt31 = dy31.timetuple()
            jftmp = tt31.tm_yday + 1
            jsq1 = numpy.arange(jst,jftmp)
            jsq2 = numpy.arange(1,jfn)
            jdsq = numpy.append(jsq1,jsq2)
        elif mnfn < 12:
            dyfn = datetime.date(yrlst[k],mnfn+1,1)
            ttfn = dyfn.timetuple()
            jfn = ttfn.tm_yday
            jdsq = numpy.arange(jst,jfn)
        else:
            dyfn = datetime.date(yrlst[k]+1,1,1)
            dy31 = datetime.date(yrlst[k],12,31)
            tt31 = dy31.timetuple()
            jfn = tt31.tm_yday + 1
 
        print(dyst)
        print(dyfn)
        dystidx = abs((dyst-dyinit).days)
        dyfnidx = abs((dyfn-dyinit).days)

        print(jdsq)
        tmhld = numpy.repeat(jdsq,nx*ny)

        stridx = 'Day Range: %d, %d\n' % (dystidx,dyfnidx)
        print(stridx)

        # Cloud slab: HDF5 or NetCDF
        lncr = len(cslst[k])
        l3 = lncr - 3
        if (cslst[k][l3:lncr] == '.h5'):
            f = h5py.File(cslst[k],'r')
            tms = f['/time'][:,dystidx:dyfnidx]
            ctyp1 = f['/ctype'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            ctyp2 = f['/ctype2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprt1 = f['/cprtop'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprt2 = f['/cprtop2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprb1 = f['/cprbot'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprb2 = f['/cprbot2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cfrc1 = f['/cfrac'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cfrc2 = f['/cfrac2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cfrc12 = f['/cfrac12'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cngwt1 = f['/cngwat'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cngwt2 = f['/cngwat2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cttp1 = f['/cstemp'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cttp2 = f['/cstemp2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            f.close()
        elif (cslst[k][l3:lncr] == '.nc'):
            f = Dataset(cslst[k],'r')
            tms = f.variables['time'][dystidx:dyfnidx]
            ctyp1 = f.variables['ctype1'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            ctyp2 = f.variables['ctype2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprt1 = f.variables['cprtop1'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprt2 = f.variables['cprtop2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprb1 = f.variables['cprbot1'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cprb2 = f.variables['cprbot2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cngwt1 = f.variables['cngwat1'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cngwt2 = f.variables['cngwat2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cttp1 = f.variables['cstemp1'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            cttp2 = f.variables['cstemp2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
            f.close()

        tmflt = tms.flatten()
        nt = tmflt.shape[0]
        lnhld = numpy.tile(lnrp,nt)
        lthld = numpy.tile(ltrp,nt)

        # MERRA variables
        f = Dataset(mtlst[k],'r')
        psfc = f.variables['spres'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        stparr = f.variables['stemp'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        salinf = f.variables['salti']
        if salinf.ndim == 3:
            salarr = f.variables['salti'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx]
        elif salinf.ndim == 2:
            salarr = f.variables['salti'][ltmn:ltmx,lnmn:lnmx]
        tmparr = f.variables['ptemp'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        h2oarr = f.variables['rh'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        altarr = f.variables['palts'][dystidx:dyfnidx,:,ltmn:ltmx,lnmn:lnmx]
        f.close()

        # Mask
        print(ctyp1.shape)
        nt = ctyp1.shape[0]
        mskall = numpy.tile(mskflt,nt)
        msksq = numpy.arange(mskall.shape[0])
        msksb = msksq[mskall > 0]
        mskstr = 'Total Obs: %d, Within Mask: %d \n' % (msksq.shape[0],msksb.shape[0])
        print(mskstr)

        nslbtmp = numpy.zeros((ctyp1.shape),dtype=numpy.int16)
        nslbtmp[(ctyp1 > 100) & (ctyp2 > 100)] = 2
        nslbtmp[(ctyp1 > 100) & (ctyp2 < 100)] = 1
   
        # AIRS clouds
        f = Dataset(airslst[k],'r')
        arsfrc1 = f.variables['AIRS_CldFrac_1'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx,:]
        arsfrc2 = f.variables['AIRS_CldFrac_2'][dystidx:dyfnidx,ltmn:ltmx,lnmn:lnmx,:]
        f.close()

        # Sum
        frctot = arsfrc1 + arsfrc2

        # Construct Clr/PC/Ovc indicator for AIRS total cloud frac
        totclr = numpy.zeros(frctot.shape,dtype=numpy.int16)
        totclr[frctot == 0.0] = -1
        totclr[frctot == 1.0] = 1
        totclr = ma.masked_array(totclr, mask = frctot.mask)

        frc0 = frctot[:,:,:,0]
        frc0 = frc0.flatten()
        frcsq = numpy.arange(tmhld.shape[0])
        # Subset by AIRS matchup and location masks
        frcsb = frcsq[(numpy.logical_not(frc0.mask)) & (mskall > 0)]

        nairs = frcsb.shape[0]
        print(tmhld.shape)
        print(frcsb.shape)

        ctyp1 = ctyp1.flatten()
        ctyp2 = ctyp2.flatten()
        nslbtmp = nslbtmp.flatten()
        cngwt1 = cngwt1.flatten() 
        cngwt2 = cngwt2.flatten() 
        cttp1 = cttp1.flatten() 
        cttp2 = cttp2.flatten() 
        psfc = psfc.flatten()

        # Number of slabs
        if tsmp == 0:
            nslabout = numpy.zeros((nairs,),dtype=numpy.int16)
            nslabout[:] = nslbtmp[frcsb]
        else:
            nslabout = numpy.append(nslabout,nslbtmp[frcsb]) 

        # For two slabs, slab 1 must have highest cloud bottom pressure
        cprt1 = cprt1.flatten()
        cprt2 = cprt2.flatten()
        cprb1 = cprb1.flatten()
        cprb2 = cprb2.flatten()
        slabswap = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16)
        swpsq = frcsq[(nslbtmp == 2) & (cprb1 < cprb2)] 
        slabswap[swpsq] = 1

        # Cloud Pressure variables
        pbttmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        pbttmp1[nslbtmp >= 1] = cprb1[nslbtmp >= 1]
        pbttmp1[swpsq] = cprb2[swpsq]

        ptptmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        ptptmp1[nslbtmp >= 1] = cprt1[nslbtmp >= 1]
        ptptmp1[swpsq] = cprt2[swpsq]

        pbttmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        pbttmp2[nslbtmp == 2] = cprb2[nslbtmp == 2]
        pbttmp2[swpsq] = cprb1[swpsq]

        ptptmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        ptptmp2[nslbtmp == 2] = cprt2[nslbtmp == 2]
        ptptmp2[swpsq] = cprt1[swpsq]

        # DP Cloud transformation
        dptmp1 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dptmp1[nslbtmp >= 1] = pbttmp1[nslbtmp >= 1] - ptptmp1[nslbtmp >= 1]

        dpslbtmp = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dpslbtmp[nslbtmp == 2] = ptptmp1[nslbtmp == 2] - pbttmp2[nslbtmp == 2]

        dptmp2 = numpy.zeros((ctyp1.shape[0],)) - 9999.0
        dptmp2[nslbtmp == 2] = pbttmp2[nslbtmp == 2] - ptptmp2[nslbtmp == 2]

        # Adjust negative DPSlab values
        dpnsq = frcsq[(nslbtmp == 2) & (dpslbtmp <= 0.0) & (dpslbtmp > -1000.0)] 
        dpadj = numpy.zeros((ctyp1.shape[0],)) 
        dpadj[dpnsq] = numpy.absolute(dpslbtmp[dpnsq])
 
        dpslbtmp[dpnsq] = 10.0
        dptmp1[dpnsq] = dptmp1[dpnsq] / 2.0
        dptmp2[dpnsq] = dptmp2[dpnsq] / 2.0

        # Sigma / Logit Adjustments
        zpbtmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdp1tmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdslbtmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        zdp2tmp = numpy.zeros((psfc.shape[0],)) - 9999.0
        ncldct = 0
        for t in range(psfc.shape[0]):
            if ( (pbttmp1[t] >= 0.0) and (dpslbtmp[t] >= 0.0) ):
                prptmp = numpy.array( [ (psfc[t] - pbttmp1[t]) / psfc[t], \
                                         dptmp1[t] / psfc[t], dpslbtmp[t] / psfc[t], \
                                         dptmp2[t] / psfc[t], 0.0 ] )
                if (prptmp[0] < 0.0):
                    # Adjustment needed
                    prpadj = prptmp[0]
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    prptmp[2] = prptmp[2] + prpadj*prptmp[2]
                    prptmp[3] = prptmp[3] + prpadj*prptmp[3]
                    ncldct = ncldct + 1
                elif (prptmp[0] == 0.0):
                    # Adjustment needed
                    prpadj = -0.01
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    prptmp[2] = prptmp[2] + prpadj*prptmp[2]
                    prptmp[3] = prptmp[3] + prpadj*prptmp[3]
                    ncldct = ncldct + 1
                prptmp[4] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2] - prptmp[3]
                ztmp = calculate_VPD.lgtzs(prptmp)
                zpbtmp[t] = ztmp[0]
                zdp1tmp[t] = ztmp[1]
                zdslbtmp[t] = ztmp[2]
                zdp2tmp[t] = ztmp[3]
            elif ( pbttmp1[t] >= 0.0  ):
                prptmp = numpy.array( [ (psfc[t] - pbttmp1[t]) / psfc[t], \
                                         dptmp1[t] / psfc[t], 0.0 ] )
                if (prptmp[0] < 0.0):
                    # Adjustment needed
                    prpadj = prptmp[0]
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    ncldct = ncldct + 1
                elif (prptmp[0] == 0.0):
                    # Adjustment needed
                    prpadj = -0.01
                    prptmp[0] = 0.01
                    prptmp[1] = prptmp[1] + prpadj*prptmp[1]
                    ncldct = ncldct + 1
                prptmp[2] = 1.0 - prptmp[0] - prptmp[1]
                ztmp = calculate_VPD.lgtzs(prptmp)
                zpbtmp[t] = ztmp[0]
                zdp1tmp[t] = ztmp[1]
                zdslbtmp[t] = -9999.0 
                zdp2tmp[t] = -9999.0 
            else:            
                zpbtmp[t] = -9999.0 
                zdp1tmp[t] = -9999.0 
                zdslbtmp[t] = -9999.0 
                zdp2tmp[t] = -9999.0 
        str1 = 'Cloud Bot Pres Below Sfc: %d ' % (ncldct)
        print(str1)

        if tsmp == 0:
            psfcout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            psfcout[:] = psfc[frcsb]
            prsbot1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            prsbot1out[:] = zpbtmp[frcsb]
            dpcld1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            dpcld1out[:] = zdp1tmp[frcsb]
            dpslbout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            dpslbout[:] = zdslbtmp[frcsb]
            dpcld2out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            dpcld2out[:] = zdp2tmp[frcsb]
        else:
            psfcout = numpy.append(psfcout,psfc[frcsb]) 
            prsbot1out = numpy.append(prsbot1out,zpbtmp[frcsb])
            dpcld1out = numpy.append(dpcld1out,zdp1tmp[frcsb])
            dpslbout = numpy.append(dpslbout,zdslbtmp[frcsb])
            dpcld2out = numpy.append(dpcld2out,zdp2tmp[frcsb])

        # Slab Types: 101.0 = Liquid, 201.0 = Ice, None else
        # Output: 0 = Liquid, 1 = Ice
        typtmp1 = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16) - 99
        typtmp1[nslbtmp >= 1] = (ctyp1[nslbtmp >= 1] - 1.0) / 100.0 - 1.0
        typtmp1[swpsq] = (ctyp2[swpsq] - 1.0) / 100.0 - 1.0

        typtmp2 = numpy.zeros((ctyp1.shape[0],),dtype=numpy.int16) - 99
        typtmp2[nslbtmp == 2] = (ctyp2[nslbtmp == 2] - 1.0) / 100.0 - 1.0 
        typtmp2[swpsq] = (ctyp1[swpsq] - 1.0) / 100.0 - 1.0

        if tsmp == 0:
            slbtyp1out = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            slbtyp1out[:] = typtmp1[frcsb]
            slbtyp2out = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            slbtyp2out[:] = typtmp2[frcsb]
        else:
            slbtyp1out = numpy.append(slbtyp1out,typtmp1[frcsb]) 
            slbtyp2out = numpy.append(slbtyp2out,typtmp2[frcsb]) 

        # Cloud Cover Indicators
        totclrtmp = numpy.zeros((frcsb.shape[0],3,3),dtype=numpy.int16)
        frctottmp = numpy.zeros((frcsb.shape[0],3,3),dtype=frctot.dtype)
        cctr = 0
        for frw in range(3):
            for fcl in range(3):
                clrvec = totclr[:,:,:,cctr].flatten()
                frcvec = frctot[:,:,:,cctr].flatten()
                totclrtmp[:,frw,fcl] = clrvec[frcsb]
                frctottmp[:,frw,fcl] = frcvec[frcsb]
                cctr = cctr + 1
        if tsmp == 0:
            totclrout = numpy.zeros(totclrtmp.shape,dtype=numpy.int16)
            totclrout[:,:,:] = totclrtmp
            frctotout = numpy.zeros(frctottmp.shape,dtype=frctottmp.dtype)
            frctotout[:,:,:] = frctottmp
        else:
            totclrout = numpy.append(totclrout,totclrtmp,axis=0)
            frctotout = numpy.append(frctotout,frctottmp,axis=0)

        # Cloud Fraction Logit, still account for swapping
        #z1tmp = numpy.zeros((frcsb.shape[0],3,3)) - 9999.0
        z2tmp = numpy.zeros((frcsb.shape[0],3,3)) - 9999.0
        z12tmp = numpy.zeros((frcsb.shape[0],3,3)) - 9999.0

        # Cloud Fraction
        cctr = 0
        for frw in range(3):
            for fcl in range(3):
                frcvect = frctot[:,:,:,cctr].flatten()
                frcvec1 = arsfrc1[:,:,:,cctr].flatten()        
                frcvec2 = arsfrc2[:,:,:,cctr].flatten()        

                # Quick fix for totals over 1.0
                fvsq = numpy.arange(frcvect.shape[0])
                fvsq2 = fvsq[frcvect > 1.0]
                frcvect[fvsq2] = frcvect[fvsq2] / 1.0
                frcvec1[fvsq2] = frcvec1[fvsq2] / 1.0
                frcvec2[fvsq2] = frcvec2[fvsq2] / 1.0
              
                for t in range(nairs):
                    crslb = nslbtmp[frcsb[t]]
                    crclr = totclrtmp[t,frw,fcl]
                    if ( (crslb == 0) or (crclr == -1) ):
                        #z1tmp[t,frw,fcl] = -9999.0
                        z2tmp[t,frw,fcl] = -9999.0
                        z12tmp[t,frw,fcl] = -9999.0
                    elif ( (crslb == 1) and (crclr == 1) ):
                        #z1tmp[t,frw,fcl] = -9999.0
                        z2tmp[t,frw,fcl] = -9999.0
                        z12tmp[t,frw,fcl] = -9999.0
                    elif ( (crslb == 1) and (crclr == 0) ):
                        #prptmp = numpy.array( [frcvect[frcsb[t]], 1.0 - frcvect[frcsb[t]] ] )
                        #ztmp = calculate_VPD.lgtzs(prptmp)
                        #z1tmp[t,frw,fcl] = ztmp[0] 
                        z2tmp[t,frw,fcl] = -9999.0
                        z12tmp[t,frw,fcl] = -9999.0
                    # For 2 slabs, recall AIRS cloud layers go upper/lower, ours is opposite
                    # Also apply random overlap adjust AIRS zero values
                    elif ( (crslb == 2) and (crclr == 0) ):
                        frcs = numpy.array([frcvec2[frcsb[t]],frcvec1[frcsb[t]]])
                        if (numpy.sum(frcs) < 0.01):
                            frcs[0] = 0.005
                            frcs[1] = 0.005
                        elif frcs[0] < 0.005:
                            frcs[0] = 0.005
                            frcs[1] = frcs[1] - 0.005
                        elif frcs[1] < 0.005:
                            frcs[1] = 0.005
                            frcs[0] = frcs[0] - 0.005
                        mnfrc = numpy.amin(frcs)
                        c12tmp = random.uniform(0.0,mnfrc,size=1)
                        prptmp = numpy.array( [frcs[0] - c12tmp[0]*frcs[1], \
                                               frcs[1] - c12tmp[0]*frcs[0], c12tmp[0], 0.0])
                        prptmp[3] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2]
                        prpcld = (prptmp[0] + prptmp[1] + prptmp[2])
                        prpfnl = numpy.array([prptmp[1] / prpcld, prptmp[2] / prpcld, prptmp[0] / prpcld]) 
                        ztmp = calculate_VPD.lgtzs(prpfnl)
                        #z1tmp[t,frw,fcl] = ztmp[0] 
                        z2tmp[t,frw,fcl] = ztmp[0] 
                        z12tmp[t,frw,fcl] = ztmp[1]
                    elif ( (crslb == 2) and (crclr == 1) ):
                        frcs = numpy.array([frcvec2[frcsb[t]],frcvec1[frcsb[t]]])
                        if frcs[0] < 0.005: 
                            frcs[0] = 0.005
                            frcs[1] = frcs[1] - 0.005
                        elif frcs[1] < 0.005:
                            frcs[1] = 0.005
                            frcs[0] = frcs[0] - 0.005
                        mnfrc = numpy.amin(frcs)
                        c12tmp = random.uniform(0.0,mnfrc,size=1)
                        prptmp = numpy.array( [0.999 * (frcs[0] - c12tmp[0]*frcs[1]), \
                                               0.999 * (frcs[1] - c12tmp[0]*frcs[0]), 0.999 * c12tmp[0], 0.001])
                        prptmp[3] = 1.0 - prptmp[0] - prptmp[1] - prptmp[2]
                        prpcld = (prptmp[0] + prptmp[1] + prptmp[2])
                        prpfnl = numpy.array([prptmp[1] / prpcld, prptmp[2] / prpcld, prptmp[0] / prpcld]) 
                        ztmp = calculate_VPD.lgtzs(prpfnl)
                        #z1tmp[t,frw,fcl] = ztmp[0] 
                        z2tmp[t,frw,fcl] = ztmp[0] 
                        z12tmp[t,frw,fcl] = ztmp[1]
                    

                cctr = cctr + 1


        if tsmp == 0:
            #cfclgt1out = numpy.zeros(z1tmp.shape)
            #cfclgt1out[:,:,:] = z1tmp
            cfclgt2out = numpy.zeros(z2tmp.shape)
            cfclgt2out[:,:,:] = z2tmp
            cfclgt12out = numpy.zeros(z12tmp.shape)
            cfclgt12out[:,:,:] = z12tmp
        else:
            #cfclgt1out = numpy.append(cfclgt1out,z1tmp,axis=0) 
            cfclgt2out = numpy.append(cfclgt2out,z2tmp,axis=0) 
            cfclgt12out = numpy.append(cfclgt12out,z12tmp,axis=0) 


        # Cloud Non-Gas Water
        ngwttmp1 = numpy.zeros(cngwt1.shape[0]) - 9999.0
        ngwttmp1[nslbtmp >= 1] = cngwt1[nslbtmp >= 1]
        ngwttmp1[swpsq] = cngwt2[swpsq]

        ngwttmp2 = numpy.zeros(cngwt1.shape[0]) - 9999.0
        ngwttmp2[nslbtmp == 2] = cngwt2[nslbtmp == 2] 
        ngwttmp2[swpsq] = cngwt1[swpsq] 

        if tsmp == 0:
            ngwt1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            ngwt1out[:] = ngwttmp1[frcsb]
            ngwt2out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            ngwt2out[:] = ngwttmp2[frcsb]
        else:
            ngwt1out = numpy.append(ngwt1out,ngwttmp1[frcsb]) 
            ngwt2out = numpy.append(ngwt2out,ngwttmp2[frcsb]) 

        # Cloud Top Temperature 
        cttptmp1 = numpy.zeros(cttp1.shape[0]) - 9999.0
        cttptmp1[nslbtmp >= 1] = cttp1[nslbtmp >= 1]
        cttptmp1[swpsq] = cttp2[swpsq]

        cttptmp2 = numpy.zeros(cttp1.shape[0]) - 9999.0
        cttptmp2[nslbtmp == 2] = cttp2[nslbtmp == 2] 
        cttptmp2[swpsq] = cttp1[swpsq] 

        if tsmp == 0:
            cttp1out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            cttp1out[:] = cttptmp1[frcsb]
            cttp2out = numpy.zeros((frcsb.shape[0],)) - 9999.0
            cttp2out[:] = cttptmp2[frcsb]
        else:
            cttp1out = numpy.append(cttp1out,cttptmp1[frcsb]) 
            cttp2out = numpy.append(cttp2out,cttptmp2[frcsb]) 

        # Temp/RH profiles
        tmptmp = numpy.zeros((nairs,nzout))
        h2otmp = numpy.zeros((nairs,nzout))
        alttmp = numpy.zeros((nairs,nzout))
        for j in range(nzout):
            tmpvec = tmparr[:,j,:,:].flatten()
            tmpvec[tmpvec > 1e30] = -9999.
            tmptmp[:,j] = tmpvec[frcsb]

            altvec = altarr[:,j,:,:].flatten()
            alttmp[:,j] = altvec[frcsb]

            h2ovec = h2oarr[:,j,:,:].flatten()
            h2ovec[h2ovec > 1e30] = -9999.
            h2otmp[:,j] = h2ovec[frcsb]
        if tsmp == 0:
            tmpmerout = numpy.zeros(tmptmp.shape)
            tmpmerout[:,:] = tmptmp
            h2omerout = numpy.zeros(h2otmp.shape)
            h2omerout[:,:] = h2otmp
            altout = numpy.zeros(alttmp.shape)
            altout[:,:] = alttmp
        else:
            tmpmerout = numpy.append(tmpmerout,tmptmp,axis=0)
            h2omerout = numpy.append(h2omerout,h2otmp,axis=0)
            altout = numpy.append(altout,alttmp,axis=0)
             
        # Surface
        stparr = stparr.flatten()
        psfarr = psfc.flatten()
        if salarr.ndim == 2:
            salarr = salarr.flatten()
            salfl = numpy.tile(salarr[:],nt) 
        elif salarr.ndim == 3:
            salfl = salarr.flatten()
 
        if tsmp == 0:
            sftmpout = numpy.zeros((nairs,)) - 9999.0
            sftmpout[:] = stparr[frcsb]
            psfcout = numpy.zeros((nairs,)) - 9999.0
            psfcout[:] = psfarr[frcsb]
            sfaltout = numpy.zeros((nairs,)) - 9999.0 
            sfaltout[:] = salfl[frcsb] 
        else:
            sftmpout = numpy.append(sftmpout,stparr[frcsb])
            psfcout = numpy.append(psfcout,psfarr[frcsb])
            sfaltout = numpy.append(sfaltout,salfl[frcsb])



        # Loc/Time
        if tsmp == 0:
            latout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            latout[:] = lthld[frcsb]
            lonout = numpy.zeros((frcsb.shape[0],)) - 9999.0
            lonout[:] = lnhld[frcsb]
            yrout = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            yrout[:] = yrlst[k]
            jdyout = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            jdyout[:] = tmhld[frcsb]
        else:
            latout = numpy.append(latout,lthld[frcsb])
            lonout = numpy.append(lonout,lnhld[frcsb])
            yrtmp = numpy.zeros((frcsb.shape[0],),dtype=numpy.int16)
            yrtmp[:] = yrlst[k]
            yrout = numpy.append(yrout,yrtmp)
            jdyout = numpy.append(jdyout,tmhld[frcsb])

        tsmp = tsmp + nairs 

    # Process quantiles

    nslbqs = calculate_VPD.quantile_msgdat_discrete(nslabout,prbs)
    str1 = '%.2f Number Slab Quantile: %d' % (prbs[103],nslbqs[103])
    print(str1)
    print(nslbqs)

#    psfcqs = calculate_VPD.quantile_msgdat(psfcout,prbs)
#    str1 = '%.2f Surface Pressure Quantile: %.3f' % (prbs[53],psfcqs[53])
#    print(str1)

    prsbt1qs = calculate_VPD.quantile_msgdat(prsbot1out,prbs)
    str1 = '%.2f CldBot1 Pressure Quantile: %.3f' % (prbs[103],prsbt1qs[103])
    print(str1)

    dpcld1qs = calculate_VPD.quantile_msgdat(dpcld1out,prbs)
    str1 = '%.2f DPCloud1 Quantile: %.3f' % (prbs[103],dpcld1qs[103])
    print(str1)

    dpslbqs = calculate_VPD.quantile_msgdat(dpslbout,prbs)
    str1 = '%.2f DPSlab Quantile: %.3f' % (prbs[103],dpslbqs[103])
    print(str1)

    dpcld2qs = calculate_VPD.quantile_msgdat(dpcld2out,prbs)
    str1 = '%.2f DPCloud2 Quantile: %.3f' % (prbs[103],dpcld2qs[103])
    print(str1)

    slb1qs = calculate_VPD.quantile_msgdat_discrete(slbtyp1out,prbs)
    str1 = '%.2f Type1 Quantile: %d' % (prbs[103],slb1qs[103])
    print(str1)

    slb2qs = calculate_VPD.quantile_msgdat_discrete(slbtyp2out,prbs)
    str1 = '%.2f Type2 Quantile: %d' % (prbs[103],slb2qs[103])
    print(str1)

    # Indicators
    totclrqout = numpy.zeros((3,3,nprb)) - 99
    frctotqout = numpy.zeros((3,3,nprb)) - 9999.0
    #lgt1qs = numpy.zeros((3,3,nprb)) - 9999.0    
    lgt2qs = numpy.zeros((3,3,nprb)) - 9999.0    
    lgt12qs = numpy.zeros((3,3,nprb)) - 9999.0    

    for frw in range(3):
        for fcl in range(3):
            tmpclr = calculate_VPD.quantile_msgdat_discrete(totclrout[:,frw,fcl],prbs)
            totclrqout[frw,fcl,:] = tmpclr[:]
            str1 = 'Clr/Ovc Indicator %d, %d %.2f Quantile: %d' % (frw,fcl,prbs[103],tmpclr[103])
            print(str1)
 
            tmpfrcq = calculate_VPD.quantile_msgdat(frctotout[:,frw,fcl],prbs)
            frctotqout[frw,fcl,:] = tmpfrcq[:]
            str1 = 'Tot Cld Frac %d, %d %.2f Quantile: %.4f' % (frw,fcl,prbs[103],tmpfrcq[103])
            print(str1)
            #tmplgtq = calculate_VPD.quantile_msgdat(cfclgt1out[:,frw,fcl],prbs)
            #lgt1qs[frw,fcl,:] = tmplgtq[:]
            tmplgtq = calculate_VPD.quantile_msgdat(cfclgt2out[:,frw,fcl],prbs)
            lgt2qs[frw,fcl,:] = tmplgtq[:]
            tmplgtq = calculate_VPD.quantile_msgdat(cfclgt12out[:,frw,fcl],prbs)
            lgt12qs[frw,fcl,:] = tmplgtq[:]
            str1 = 'CFrac Logit %d, %d %.2f Quantile: %.3f, %.3f' % (frw,fcl,prbs[103], \
                                lgt2qs[frw,fcl,103],lgt12qs[frw,fcl,103])
            print(str1)

    ngwt1qs = calculate_VPD.quantile_msgdat(ngwt1out,prbs)
    str1 = '%.2f NGWater1 Quantile: %.3f' % (prbs[103],ngwt1qs[103])
    print(str1)

    ngwt2qs = calculate_VPD.quantile_msgdat(ngwt2out,prbs)
    str1 = '%.2f NGWater2 Quantile: %.3f' % (prbs[103],ngwt2qs[103])
    print(str1)

    cttp1qs = calculate_VPD.quantile_msgdat(cttp1out,prbs)
    str1 = '%.2f CTTemp1 Quantile: %.3f' % (prbs[103],cttp1qs[103])
    print(str1)

    cttp2qs = calculate_VPD.quantile_msgdat(cttp2out,prbs)
    str1 = '%.2f CTTemp2 Quantile: %.3f' % (prbs[103],cttp2qs[103])
    print(str1)

    # Temp/RH Quantiles
    tmpqout = numpy.zeros((nzout,nprb)) - 9999.
    rhqout = numpy.zeros((nzout,nprb)) - 9999.
    sftmpqs = numpy.zeros((nprb,)) - 9999.
    sfaltqs = numpy.zeros((nprb,)) - 9999.
    psfcqs = numpy.zeros((nprb,)) - 9999.
    altmed = numpy.zeros((nzout,)) - 9999.

    ztmpout = numpy.zeros((tsmp,nzout)) - 9999.
    zrhout = numpy.zeros((tsmp,nzout)) - 9999.
    zsftmpout = numpy.zeros((tsmp,)) - 9999.
    zsfaltout = numpy.zeros((tsmp,)) - 9999.
    zpsfcout = numpy.zeros((tsmp,)) - 9999.

    # Quantiles
    for j in range(nzout):
        tmptmp = calculate_VPD.quantile_msgdat(tmpmerout[:,j],prbs)
        tmpqout[j,:] = tmptmp[:]
        str1 = 'Plev %.2f, %.2f Temp Quantile: %.3f' % (plev[j],prbs[103],tmptmp[103])
        print(str1)

        # Transform if some not missing
        if (tmptmp[0] != -9999.):
            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(tmpmerout[:,j], tmptmp, prbs,  msgval=-9999.)
            ztmpout[:,j] = ztmp[:]

        alttmp = calculate_VPD.quantile_msgdat(altout[:,j],prbs)
        altmed[j] = alttmp[103]
        str1 = 'Plev %.2f, %.2f Alt Quantile: %.3f' % (plev[j],prbs[103],alttmp[103])
        print(str1)

        # Adjust RH over 100
        rhadj = h2omerout[:,j]
        rhadj[rhadj > 1.0] = 1.0
        rhqtmp = calculate_VPD.quantile_msgdat(rhadj,prbs)
        rhqout[j,:] = rhqtmp[:]
        str1 = 'Plev %.2f, %.2f RH Quantile: %.4f' % (plev[j],prbs[103],rhqtmp[103])
        print(str1)

        if (rhqtmp[0] != -9999.):
            zrh = calculate_VPD.std_norm_quantile_from_obs_fill_msg(rhadj, rhqtmp, prbs,  msgval=-9999.)
            zrhout[:,j] = zrh[:]
        h2omerout[:,j] = rhadj

    psfcqs = calculate_VPD.quantile_msgdat(psfcout,prbs)
    str1 = '%.2f PSfc Quantile: %.2f' % (prbs[103],psfcqs[103])
    print(str1)
    zpsfcout = calculate_VPD.std_norm_quantile_from_obs(psfcout, psfcqs, prbs, msgval=-9999.) 

    sftpqs = calculate_VPD.quantile_msgdat(sftmpout,prbs)
    str1 = '%.2f SfcTmp Quantile: %.2f' % (prbs[103],sftpqs[103])
    print(str1)
    zsftmpout = calculate_VPD.std_norm_quantile_from_obs(sftmpout, sftpqs, prbs, msgval=-9999.) 

    sfalqs = calculate_VPD.quantile_msgdat(sfaltout,prbs)
    str1 = '%.2f SfcAlt Quantile: %.2f' % (prbs[103],sfalqs[103])
    print(str1)
    zsfaltout = calculate_VPD.std_norm_quantile_from_obs(sfaltout, sfalqs, prbs, msgval=-9999.) 

    # Output Quantiles
    qfnm = '%s/CONUS_AIRS_%s_%04d_%02dUTC_%s_State_Quantile.nc' % (dtdr,sstr,yrlst[k],hrchc,rgchc)
    qout = Dataset(qfnm,'w') 

    dimp = qout.createDimension('probability',nprb)
    dimfov1 = qout.createDimension('fovrow',3)
    dimfov2 = qout.createDimension('fovcol',3)
    dimz = qout.createDimension('level',nzout)

    varlvl = qout.createVariable('level','f4',['level'], fill_value = -9999)
    varlvl[:] = plev
    varlvl.long_name = 'AIRS/SARTA pressure levels'
    varlvl.units = 'hPa'
    varlvl.missing_value = -9999

    varprb = qout.createVariable('probability','f4',['probability'], fill_value = -9999)
    varprb[:] = prbs
    varprb.long_name = 'Probability break points'
    varprb.units = 'none'
    varprb.missing_value = -9999

    varnslb = qout.createVariable('NumberSlab_quantile','i2',['probability'], fill_value = -99)
    varnslb[:] = nslbqs
    varnslb.long_name = 'Number of cloud slabs quantiles'
    varnslb.units = 'Count'
    varnslb.missing_value = -99

    varcbprs = qout.createVariable('CloudBot1Logit_quantile','f4',['probability'], fill_value = -9999)
    varcbprs[:] = prsbt1qs
    varcbprs.long_name = 'Slab 1 cloud bottom pressure logit quantiles'
    varcbprs.units = 'hPa'
    varcbprs.missing_value = -9999

    vardpc1 = qout.createVariable('DPCloud1Logit_quantile','f4',['probability'], fill_value = -9999)
    vardpc1[:] = dpcld1qs
    vardpc1.long_name = 'Slab 1 cloud pressure depth logit quantiles'
    vardpc1.units = 'hPa'
    vardpc1.missing_value = -9999

    vardpslb = qout.createVariable('DPSlabLogit_quantile','f4',['probability'], fill_value = -9999)
    vardpslb[:] = dpslbqs
    vardpslb.long_name = 'Two-slab vertical separation logit quantiles' 
    vardpslb.units = 'hPa'
    vardpslb.missing_value = -9999

    vardpc2 = qout.createVariable('DPCloud2Logit_quantile','f4',['probability'], fill_value = -9999)
    vardpc2[:] = dpcld2qs
    vardpc2.long_name = 'Slab 2 cloud pressure depth logit quantiles'
    vardpc2.units = 'hPa'
    vardpc2.missing_value = -9999

    vartyp1 = qout.createVariable('CType1_quantile','i2',['probability'], fill_value = -99)
    vartyp1[:] = slb1qs
    vartyp1.long_name = 'Slab 1 cloud type quantiles'
    vartyp1.units = 'None'
    vartyp1.missing_value = -99
    vartyp1.comment = 'Cloud slab type: 0=Liquid, 1=Ice'

    vartyp2 = qout.createVariable('CType2_quantile','i2',['probability'], fill_value = -99)
    vartyp2[:] = slb2qs
    vartyp2.long_name = 'Slab 2 cloud type quantiles'
    vartyp2.units = 'None'
    vartyp2.missing_value = -99
    vartyp2.comment = 'Cloud slab type: 0=Liquid, 1=Ice'

    varcvr = qout.createVariable('CCoverInd_quantile','i2',['fovrow','fovcol','probability'], fill_value = 99)
    varcvr[:] = totclrqout
    varcvr.long_name = 'Cloud cover indicator quantiles'
    varcvr.units = 'None'
    varcvr.missing_value = -99
    varcvr.comment = 'Cloud cover indicators: -1=Clear, 0=Partly cloudy, 1=Overcast'

    varfrc = qout.createVariable('TotCFrc_quantile','f4',['fovrow','fovcol','probability'], fill_value = -9999)
    varfrc[:] = frctotqout
    varfrc.long_name = 'Total cloud fraction quantiles'
    varfrc.units = 'None'
    varfrc.missing_value = -9999

    #varlgt1 = qout.createVariable('CFrcLogit1_quantile','f4',['fovrow','fovcol','probability'], fill_value = -9999)
    #varlgt1[:] = lgt1qs
    #varlgt1.long_name = 'Slab 1 cloud fraction (cfrac1x) logit quantiles'
    #varlgt1.units = 'None'
    #varlgt1.missing_value = -9999

    varlgt2 = qout.createVariable('CFrcLogit2_quantile','f4',['fovrow','fovcol','probability'], fill_value = -9999)
    varlgt2[:] = lgt2qs
    varlgt2.long_name = 'Slab 2 cloud fraction (cfrac2x) logit quantiles'
    varlgt2.units = 'None'
    varlgt2.missing_value = -9999

    varlgt12 = qout.createVariable('CFrcLogit12_quantile','f4',['fovrow','fovcol','probability'], fill_value = -9999)
    varlgt12[:] = lgt12qs
    varlgt12.long_name = 'Slab 1/2 overlap fraction (cfrac12) logit quantiles'
    varlgt12.units = 'None'
    varlgt12.missing_value = -9999

    varngwt1 = qout.createVariable('NGWater1_quantile','f4',['probability'], fill_value = -9999)
    varngwt1[:] = ngwt1qs
    varngwt1.long_name = 'Slab 1 cloud non-gas water quantiles'
    varngwt1.units = 'g m^-2'
    varngwt1.missing_value = -9999

    varngwt2 = qout.createVariable('NGWater2_quantile','f4',['probability'], fill_value = -9999)
    varngwt2[:] = ngwt2qs
    varngwt2.long_name = 'Slab 2 cloud non-gas water quantiles'
    varngwt2.units = 'g m^-2'
    varngwt2.missing_value = -9999

    varcttp1 = qout.createVariable('CTTemp1_quantile','f4',['probability'], fill_value = -9999)
    varcttp1[:] = cttp1qs
    varcttp1.long_name = 'Slab 1 cloud top temperature'
    varcttp1.units = 'K'
    varcttp1.missing_value = -9999

    varcttp2 = qout.createVariable('CTTemp2_quantile','f4',['probability'], fill_value = -9999)
    varcttp2[:] = cttp2qs
    varcttp2.long_name = 'Slab 2 cloud top temperature'
    varcttp2.units = 'K'
    varcttp2.missing_value = -9999

    # Altitude grid
    varalt = qout.createVariable('Altitude_median', 'f4', ['level'], fill_value = -9999)
    varalt[:] = altmed
    varalt.long_name = 'Altitude median value'
    varalt.units = 'm'
    varalt.missing_value = -9999

    vartmp = qout.createVariable('Temperature_quantile', 'f4', ['level','probability'], fill_value = -9999)
    vartmp[:] = tmpqout
    vartmp.long_name = 'Temperature quantiles'
    vartmp.units = 'K'
    vartmp.missing_value = -9999.

    varrh = qout.createVariable('RH_quantile', 'f4', ['level','probability'], fill_value = -9999)
    varrh[:] = rhqout
    varrh.long_name = 'Relative humidity quantiles'
    varrh.units = 'Unitless'
    varrh.missing_value = -9999.

    varstmp = qout.createVariable('SfcTemp_quantile', 'f4', ['probability'], fill_value = -9999)
    varstmp[:] = sftpqs
    varstmp.long_name = 'Surface temperature quantiles'
    varstmp.units = 'K'
    varstmp.missing_value = -9999.

    varpsfc = qout.createVariable('SfcPres_quantile', 'f4', ['probability'], fill_value = -9999)
    varpsfc[:] = psfcqs
    varpsfc.long_name = 'Surface pressure quantiles'
    varpsfc.units = 'hPa'
    varpsfc.missing_value = -9999.

    varsalt = qout.createVariable('SfcAlt_quantile', 'f4', ['probability'], fill_value = -9999)
    varsalt[:] = sfalqs
    varsalt.long_name = 'Surface altitude quantiles'
    varsalt.units = 'm'
    varsalt.missing_value = -9999.

    qout.close()

    # Set up transformations
    zccvout = numpy.zeros((tsmp,3,3,)) - 9999.
    zfrcout = numpy.zeros((tsmp,3,3,)) - 9999.
    #zlgt1 = numpy.zeros((tsmp,3,3)) - 9999.
    zlgt2 = numpy.zeros((tsmp,3,3)) - 9999.
    zlgt12 = numpy.zeros((tsmp,3,3)) - 9999.

    znslb = calculate_VPD.std_norm_quantile_from_obs(nslabout, nslbqs, prbs,  msgval=-99)
    zprsbt1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(prsbot1out, prsbt1qs, prbs,  msgval=-9999.)
    zdpcld1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpcld1out, dpcld1qs, prbs,  msgval=-9999.)
    zdpslb = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpslbout, dpslbqs, prbs,  msgval=-9999.)
    zdpcld2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(dpcld2out, dpcld2qs, prbs,  msgval=-9999.)
    zctyp1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(slbtyp1out, slb1qs, prbs,  msgval=-99)
    zctyp2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(slbtyp2out, slb2qs, prbs,  msgval=-99)

    for frw in range(3):
        for fcl in range(3):
            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(totclrout[:,frw,fcl], totclrqout[frw,fcl,:], \
                                                                     prbs, msgval=-99)
            zccvout[:,frw,fcl] = ztmp[:]

            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(frctotout[:,frw,fcl], frctotqout[frw,fcl,:], \
                                                                     prbs, msgval=-9999.)
            zfrcout[:,frw,fcl] = ztmp[:]

            #ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt1out[:,frw,fcl], lgt1qs[frw,fcl,:], \
            #                                                         prbs, msgval=-9999.)
            #zlgt1[:,frw,fcl] = ztmp[:]

            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt2out[:,frw,fcl], lgt2qs[frw,fcl,:], \
                                                                     prbs, msgval=-9999.)
            zlgt2[:,frw,fcl] = ztmp[:]

            ztmp = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cfclgt12out[:,frw,fcl], lgt12qs[frw,fcl,:], \
                                                                     prbs, msgval=-9999.)
            zlgt12[:,frw,fcl] = ztmp[:]

    zngwt1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(ngwt1out, ngwt1qs, prbs,  msgval=-9999.)
    zngwt2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(ngwt2out, ngwt2qs, prbs,  msgval=-9999.)
    zcttp1 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cttp1out, cttp1qs, prbs,  msgval=-9999.)
    zcttp2 = calculate_VPD.std_norm_quantile_from_obs_fill_msg(cttp2out, cttp2qs, prbs,  msgval=-9999.)


    # Output transformed quantile samples
    zfnm = '%s/CONUS_AIRS_%s_%04d_%02dUTC_%s_State_StdGausTrans.nc' % (dtdr,sstr,yrlst[k],hrchc,rgchc)
    zout = Dataset(zfnm,'w') 

    dimsmp = zout.createDimension('sample',tsmp)
    dimfov1 = zout.createDimension('fovrow',3)
    dimfov2 = zout.createDimension('fovcol',3)
    dimz = zout.createDimension('level',nzout)

    varlon = zout.createVariable('Longitude','f4',['sample'])
    varlon[:] = lonout
    varlon.long_name = 'Longitude'
    varlon.units = 'degrees_east'

    varlat = zout.createVariable('Latitude','f4',['sample'])
    varlat[:] = latout
    varlat.long_name = 'Latitude'
    varlat.units = 'degrees_north'

    varlvl = zout.createVariable('level','f4',['level'], fill_value = -9999)
    varlvl[:] = plev
    varlvl.long_name = 'AIRS/SARTA pressure levels'
    varlvl.units = 'hPa'
    varlvl.missing_value = -9999

    varjdy = zout.createVariable('JulianDay','i2',['sample'])
    varjdy[:] = jdyout
    varjdy.long_name = 'JulianDay'
    varjdy.units = 'day'

    varyr = zout.createVariable('Year','i2',['sample'])
    varyr[:] = yrout
    varyr.long_name = 'Year'
    varyr.units = 'year'

    varnslb = zout.createVariable('NumberSlab_StdGaus','f4',['sample'], fill_value = -9999)
    varnslb[:] = znslb
    varnslb.long_name = 'Quantile transformed number of cloud slabs'
    varnslb.units = 'None'
    varnslb.missing_value = -9999.

    vdtnslb = zout.createVariable('NumberSlab_Data','i2',['sample'], fill_value = -99)
    vdtnslb[:] = nslabout
    vdtnslb.long_name = 'Number of cloud slabs'
    vdtnslb.units = 'None'
    vdtnslb.missing_value = -99

    varcbprs = zout.createVariable('CloudBot1Logit_StdGaus','f4',['sample'], fill_value = -9999)
    varcbprs[:] = zprsbt1
    varcbprs.long_name = 'Quantile transformed slab 1 cloud bottom pressure logit'
    varcbprs.units = 'None'
    varcbprs.missing_value = -9999.

    vdtcbprs = zout.createVariable('CloudBot1Logit_Data','f4',['sample'], fill_value = -9999)
    vdtcbprs[:] = prsbot1out
    vdtcbprs.long_name = 'Slab 1 cloud bottom pressure logit'
    vdtcbprs.units = 'None'
    vdtcbprs.missing_value = -9999.

    vardpc1 = zout.createVariable('DPCloud1Logit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpc1[:] = zdpcld1
    vardpc1.long_name = 'Quantile transformed slab 1 cloud pressure depth logit'
    vardpc1.units = 'None'
    vardpc1.missing_value = -9999.

    vdtdpc1 = zout.createVariable('DPCloud1Logit_Data','f4',['sample'], fill_value = -9999)
    vdtdpc1[:] = dpcld1out
    vdtdpc1.long_name = 'Slab 1 cloud pressure depth logit'
    vdtdpc1.units = 'None'
    vdtdpc1.missing_value = -9999.

    vardpslb = zout.createVariable('DPSlabLogit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpslb[:] = zdpslb
    vardpslb.long_name = 'Quantile transformed two-slab vertical separation logit'
    vardpslb.units = 'None'
    vardpslb.missing_value = -9999.

    vdtdpslb = zout.createVariable('DPSlabLogit_Data','f4',['sample'], fill_value = -9999)
    vdtdpslb[:] = dpslbout
    vdtdpslb.long_name = 'Two-slab vertical separation logit'
    vdtdpslb.units = 'None'
    vdtdpslb.missing_value = -9999.

    vardpc2 = zout.createVariable('DPCloud2Logit_StdGaus','f4',['sample'], fill_value = -9999)
    vardpc2[:] = zdpcld2
    vardpc2.long_name = 'Quantile transformed slab 2 cloud pressure depth logit'
    vardpc2.units = 'None'
    vardpc2.missing_value = -9999.

    vdtdpc2 = zout.createVariable('DPCloud2Logit_Data','f4',['sample'], fill_value = -9999)
    vdtdpc2[:] = dpcld2out
    vdtdpc2.long_name = 'Slab 2 cloud pressure depth logit'
    vdtdpc2.units = 'None'
    vdtdpc2.missing_value = -9999.

    vartyp1 = zout.createVariable('CType1_StdGaus','f4',['sample'], fill_value = -9999)
    vartyp1[:] = zctyp1
    vartyp1.long_name = 'Quantile transformed slab 1 cloud type'
    vartyp1.units = 'None'
    vartyp1.missing_value = -9999.

    vdttyp1 = zout.createVariable('CType1_Data','i2',['sample'], fill_value = -99)
    vdttyp1[:] = slbtyp1out
    vdttyp1.long_name = 'Slab 1 cloud type'
    vdttyp1.units = 'None'
    vdttyp1.missing_value = -99

    vartyp2 = zout.createVariable('CType2_StdGaus','f4',['sample'], fill_value = -9999)
    vartyp2[:] = zctyp2
    vartyp2.long_name = 'Quantile transformed slab 2 cloud type'
    vartyp2.units = 'None'
    vartyp2.missing_value = -9999.

    vdttyp2 = zout.createVariable('CType2_Data','i2',['sample'], fill_value = -99)
    vdttyp2[:] = slbtyp2out
    vdttyp2.long_name = 'Slab 2 cloud type logit'
    vdttyp2.units = 'None'
    vdttyp2.missing_value = -99

    varcov = zout.createVariable('CCoverInd_StdGaus','f4',['sample','fovrow','fovcol'], fill_value= -9999)
    varcov[:] = zccvout
    varcov.long_name = 'Quantile transformed cloud cover indicator'
    varcov.units = 'None'
    varcov.missing_value = -9999.

    vdtcov = zout.createVariable('CCoverInd_Data','i2',['sample','fovrow','fovcol'], fill_value= -99)
    vdtcov[:] = totclrout
    vdtcov.long_name = 'Cloud cover indicator'
    vdtcov.units = 'None'
    vdtcov.missing_value = -99

    varfrc = zout.createVariable('TotCFrc_StdGaus','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    varfrc[:] = zfrcout
    varfrc.long_name = 'Quantile transformed total cloud fraction'
    varfrc.units = 'None'
    varfrc.missing_value = -9999

    vdtfrc = zout.createVariable('TotCFrc_Data','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    vdtfrc[:] = frctotout
    vdtfrc.long_name = 'Total cloud fraction'
    vdtfrc.units = 'None'
    vdtfrc.missing_value = -9999

    #varlgt1 = zout.createVariable('CFrcLogit1_StdGaus','f4',['fovrow','fovcol','sample'], fill_value = -9999)
    #varlgt1[:] = zlgt1
    #varlgt1.long_name = 'Quantile transformed slab 1 cloud fraction logit'
    #varlgt1.units = 'None'
    #varlgt1.missing_value = -9999.

    varlgt2 = zout.createVariable('CFrcLogit2_StdGaus','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    varlgt2[:] = zlgt2
    varlgt2.long_name = 'Quantile transformed slab 2 cloud fraction logit'
    varlgt2.units = 'None'
    varlgt2.missing_value = -9999.

    vdtlgt2 = zout.createVariable('CFrcLogit2_Data','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    vdtlgt2[:] = cfclgt2out
    vdtlgt2.long_name = 'Slab 2 cloud fraction logit'
    vdtlgt2.units = 'None'
    vdtlgt2.missing_value = -9999.

    varlgt12 = zout.createVariable('CFrcLogit12_StdGaus','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    varlgt12[:] = zlgt12
    varlgt12.long_name = 'Quantile transformed slab 1/2 overlap fraction logit'
    varlgt12.units = 'None'
    varlgt12.missing_value = -9999.

    vdtlgt12 = zout.createVariable('CFrcLogit12_Data','f4',['sample','fovrow','fovcol'], fill_value = -9999)
    vdtlgt12[:] = cfclgt12out
    vdtlgt12.long_name = 'Slab 1/2 overlap fraction logit'
    vdtlgt12.units = 'None'
    vdtlgt12.missing_value = -9999.

    varngwt1 = zout.createVariable('NGWater1_StdGaus','f4',['sample'], fill_value = -9999)
    varngwt1[:] = zngwt1
    varngwt1.long_name = 'Quantile transformed slab 1 non-gas water'
    varngwt1.units = 'None'
    varngwt1.missing_value = -9999.

    vdtngwt1 = zout.createVariable('NGWater1_Data','f4',['sample'], fill_value = -9999)
    vdtngwt1[:] = ngwt1out
    vdtngwt1.long_name = 'Slab 1 non-gas water'
    vdtngwt1.units = 'None'
    vdtngwt1.missing_value = -9999.

    varngwt2 = zout.createVariable('NGWater2_StdGaus','f4',['sample'], fill_value = -9999)
    varngwt2[:] = zngwt2
    varngwt2.long_name = 'Quantile transformed slab 2 non-gas water'
    varngwt2.units = 'None'
    varngwt2.missing_value = -9999.

    vdtngwt2 = zout.createVariable('NGWater2_Data','f4',['sample'], fill_value = -9999)
    vdtngwt2[:] = ngwt2out
    vdtngwt2.long_name = 'Slab 2 non-gas water'
    vdtngwt2.units = 'None'
    vdtngwt2.missing_value = -9999.

    varcttp1 = zout.createVariable('CTTemp1_StdGaus','f4',['sample'], fill_value = -9999)
    varcttp1[:] = zcttp1
    varcttp1.long_name = 'Quantile transformed slab 1 cloud top temperature'
    varcttp1.units = 'None'
    varcttp1.missing_value = -9999.

    vdtcttp1 = zout.createVariable('CTTemp1_Data','f4',['sample'], fill_value = -9999)
    vdtcttp1[:] = cttp1out
    vdtcttp1.long_name = 'Slab 1 cloud top temperature'
    vdtcttp1.units = 'K'
    vdtcttp1.missing_value = -9999.

    varcttp2 = zout.createVariable('CTTemp2_StdGaus','f4',['sample'], fill_value = -9999)
    varcttp2[:] = zcttp2
    varcttp2.long_name = 'Quantile transformed slab 2 cloud top temperature'
    varcttp2.units = 'None'
    varcttp2.missing_value = -9999.

    vdtcttp2 = zout.createVariable('CTTemp2_Data','f4',['sample'], fill_value = -9999)
    vdtcttp2[:] = cttp2out
    vdtcttp2.long_name = 'Slab 2 cloud top temperature'
    vdtcttp2.units = 'K'
    vdtcttp2.missing_value = -9999.

    varsrt3 = zout.createVariable('Temperature_StdGaus', 'f4', ['sample','level'], fill_value = -9999)
    varsrt3[:] = ztmpout
    varsrt3.long_name = 'Quantile transformed temperature'
    varsrt3.units = 'None'
    varsrt3.missing_value = -9999.

    vdtsrt3 = zout.createVariable('Temperature_Data', 'f4', ['sample','level'], fill_value = -9999)
    vdtsrt3[:] = tmpmerout
    vdtsrt3.long_name = 'Temperature'
    vdtsrt3.units = 'K'
    vdtsrt3.missing_value = -9999.

    varsrt4 = zout.createVariable('RH_StdGaus', 'f4', ['sample','level'], fill_value = -9999)
    varsrt4[:] = zrhout
    varsrt4.long_name = 'Quantile transformed relative humidity'
    varsrt4.units = 'None'
    varsrt4.missing_value = -9999.

    vdtsrt4 = zout.createVariable('RH_Data', 'f4', ['sample','level'], fill_value = -9999)
    vdtsrt4[:] = h2omerout
    vdtsrt4.long_name = 'Relative humidity'
    vdtsrt4.units = 'None'
    vdtsrt4.missing_value = -9999.

    varsrts1 = zout.createVariable('SfcTemp_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts1[:] = zsftmpout
    varsrts1.long_name = 'Quantile transformed surface temperature'
    varsrts1.units = 'None'
    varsrts1.missing_value = -9999.

    vdtsrts1 = zout.createVariable('SfcTemp_Data', 'f4', ['sample'], fill_value = -9999)
    vdtsrts1[:] = sftmpout
    vdtsrts1.long_name = 'Surface temperature'
    vdtsrts1.units = 'None'
    vdtsrts1.missing_value = -9999.

    varsrts2 = zout.createVariable('SfcPres_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts2[:] = zpsfcout
    varsrts2.long_name = 'Quantile transformed surface pressure'
    varsrts2.units = 'None'
    varsrts2.missing_value = -9999.

    vdtsrts2 = zout.createVariable('SfcPres_Data', 'f4', ['sample'], fill_value = -9999)
    vdtsrts2[:] = psfcout
    vdtsrts2.long_name = 'Surface pressure'
    vdtsrts2.units = 'hPa'
    vdtsrts2.missing_value = -9999.

    varsrts3 = zout.createVariable('SfcAlt_StdGaus', 'f4', ['sample'], fill_value = -9999)
    varsrts3[:] = zsfaltout
    varsrts3.long_name = 'Quantile transformed surface altitude'
    varsrts3.units = 'None'
    varsrts3.missing_value = -9999.

    vdtsrts3 = zout.createVariable('SfcAlt_Data', 'f4', ['sample'], fill_value = -9999)
    vdtsrts3[:] = sfaltout
    vdtsrts3.long_name = 'Surface altitude'
    vdtsrts3.units = 'm'
    vdtsrts3.missing_value = -9999.

    zout.close()

    return

def zscore_update_mcem(outfile, qfile, zfile, dfile, cnffile, probs, niter = 50, sdchc = 553133):
    # Monte Carlo expectation maximization update for zscores 
    # outfile:   Output file with updated z-scores, MCEM results
    # qfile:     Quantile file
    # zfile:     Input z-score file
    # dfile:     Data file
    # cnffile:   Configuration file (CSV)
    # probs:     Probability levels for quantile
    # niter:     Maximum number of EM iterations
    # sdchc:     Random seed


    df = pandas.read_csv(cnffile, dtype = {'Order':int, 'ZScore_Name':str, 'Quantile_Name':str, \
                                           'Data_Name':str, 'Start':int, 'Length':int, 'DType':str })
    tsz = df['Length'].sum()
    szstr = '%d Total State Vector Elements' % (tsz)
    print(szstr)

    nrw = df.shape[0]
    nsmp = -1
    stctr = 0

    # RN generator
    random.seed(sdchc)

    # Initialize MVN parameters
    mu0 = numpy.zeros((tsz,),dtype=numpy.float64)
    cv0 = numpy.zeros((tsz,tsz),dtype=numpy.float64)

    # Loop through groups to initialize
    print(qfile)
    print(zfile)
    for q in range(nrw):
        if (df['Length'].values[q] == 1):
            cv0[stctr,stctr] = 1.0
        else:
            cst = stctr
            cfn = stctr + df['Length'].values[q]
            diagvl = numpy.zeros( (df['Length'].values[q],), dtype=numpy.float64) + 0.6
            cvblk = numpy.zeros( (df['Length'].values[q],df['Length'].values[q]), dtype=numpy.float64) + 0.4 + \
                    numpy.diagflat( diagvl)
            cv0[cst:cfn,cst:cfn] = cvblk

        qvrnm = df['Quantile_Name'].values[q]
        fqs = Dataset(qfile,'r')
        if (df['Group'].values[q] == 'CloudFrac'):
            qtmp = fqs.variables[qvrnm][:,:,:]
        elif (df['Length'].values[q] > 1):
            qtmp = fqs.variables[qvrnm][:,:]
        else:
            qtmp = fqs.variables[qvrnm][:]
        fqs.close()
 
        zvrnm = df['ZScore_Name'].values[q]
        fzs = Dataset(zfile,'r')
        if (df['Group'].values[q] == 'CloudFrac'):
            ztmp = fzs.variables[zvrnm][:,:,:]
        elif (df['Length'].values[q] > 1):
            ztmp = fzs.variables[zvrnm][:,:]
        else:
            ztmp = fzs.variables[zvrnm][:]
        if nsmp < 0:
            nsmp = ztmp.shape[0]
            zscrarr = numpy.zeros((nsmp,tsz),dtype=numpy.float32)
            zlwrarr = numpy.zeros((nsmp,tsz),dtype=numpy.float32)
            zuprarr = numpy.zeros((nsmp,tsz),dtype=numpy.float32)
        fzs.close()

        yvrnm = df['Data_Name'].values[q]
        fys = Dataset(dfile,'r')
        if (df['Group'].values[q] == 'CloudFrac'):
            dttmp = fys.variables[yvrnm][:,:,:]
        elif (df['Length'].values[q] > 1):
            dttmp = fys.variables[yvrnm][:,:]
        else:
            dttmp = fys.variables[yvrnm][:]
        msgvl = fys.variables[yvrnm].missing_value
        fys.close()

        # Find z-score limits
        if (df['Group'].values[q] == 'CloudFrac'):
            cctr = 0
            for frw in range(3):
                for fcl in range(3):
                    ztmplwr, ztmpupr = calculate_VPD.std_norm_limits_from_obs_fill_msg(dttmp[:,frw,fcl], qtmp[frw,fcl,:], probs, msgvl)
                    cctr = cctr + 1
                    zlwrarr[:,stctr] = ztmplwr[:]
                    zuprarr[:,stctr] = ztmpupr[:]
                    zchk = ztmp[:,frw,fcl]
                    zmsg = zchk[zchk < -10.0]
                    zchk[zchk < -10.0] = random.uniform(size=zmsg.shape[0])
                    zscrarr[:,stctr] = zchk[:]
                    zdif = ztmpupr - ztmplwr
                    zsm = numpy.sum( (zdif < 0))
                    if zsm > 0:
                        zstr = 'Lower Upper Mismatch\n  %s (%d): %d' % (df['ZScore_Name'].values[q],cctr,zsm)
                        print(zstr)
                    stctr = stctr + 1
        elif (df['Length'].values[q] > 1):
            stidx = df['Start'].values[q] - 1
            fnidx = stidx + df['Length'].values[q]
            for k in range(stidx,fnidx):
                ztmplwr, ztmpupr = calculate_VPD.std_norm_limits_from_obs_fill_msg(dttmp[:,k], qtmp[k,:], probs, msgvl)
                zlwrarr[:,stctr] = ztmplwr[:]
                zuprarr[:,stctr] = ztmpupr[:]
                zchk = ztmp[:,k]
                zmsg = zchk[zchk < -10.0]
                zchk[zchk < -10.0] = random.uniform(size=zmsg.shape[0])
                zscrarr[:,stctr] = zchk[:]
                zdif = ztmpupr - ztmplwr
                zsm = numpy.sum( (zdif < 0))
                if zsm > 0:
                    zstr = 'Lower Upper Mismatch\n  %s (%d): %d' % (df['ZScore_Name'].values[q],k,zsm)
                    print(zstr)
                stctr = stctr + 1
        else:
            ztmplwr, ztmpupr = calculate_VPD.std_norm_limits_from_obs_fill_msg(dttmp, qtmp, probs, msgvl)
            zlwrarr[:,stctr] = ztmplwr[:]
            zuprarr[:,stctr] = ztmpupr[:]
            zchk = ztmp[:]
            zmsg = zchk[zchk < -10.0]
            zchk[zchk < -10.0] = random.uniform(size=zmsg.shape[0])
            zscrarr[:,stctr] = zchk[:]
            zdif = ztmpupr - ztmplwr
            zsm = numpy.sum( (zdif < 0))
            if zsm > 0:
                zstr = 'Lower Upper Mismatch\n  %s: %d' % (df['ZScore_Name'].values[q],zsm)
                print(zstr)
            stctr = stctr + 1

    print(nsmp)

    lgdns = stats.multivariate_normal.logpdf(zscrarr, mean=mu0, cov=cv0)
    cmpllk = numpy.sum(lgdns)
    lkstr = 'Initial Log-likelihood: %.4e' % (cmpllk)
    print(lkstr)

    mucr = mu0
    cvcr = cv0
    crlk = cmpllk
    lkdf = 1e8
    zfnl = zscrarr

    # Setup output
    emout = Dataset(outfile,'w') 

    dimiter = emout.createDimension('iteration',niter)
    dimstate = emout.createDimension('state',tsz)
    dimsmp = emout.createDimension('sample',nsmp)
    
    varlk = emout.createVariable('logLike','f8',['iteration'], fill_value = -9999)
    varlk.long_name = 'Complete information log likelihood'
    varlk.units = 'None'
    varlk.missing_value = -9999
    
    varmn = emout.createVariable('state_mean','f4',['iteration','state'], fill_value = -9999)
    varmn.long_name = 'Multivariate state mean vector'
    varmn.units = 'None'
    varmn.missing_value = -9999

    varcv = emout.createVariable('state_cov','f4',['iteration','state','state'], fill_value = -9999)
    varcv.long_name = 'Multivariate state covariance matrix'
    varcv.units = 'None'
    varcv.missing_value = -9999

    varest = emout.createVariable('state_samples','f4',['iteration','sample','state'], fill_value = -9999)
    varest.long_name = 'State variable expected values'
    varest.units = 'None'
    varest.missing_value = -9999
    
    emout.close()
    
    critr = 0
    while ( (critr < niter) and (lkdf > 1e3)):
        # MCMC
        prccr = linalg.inv(cvcr) 
        zfnl, zmn1 = calculate_VPD.trnc_norm_mcmc(zfnl, mucr, prccr, zlwrarr, zuprarr, \
                                                  niter = 450, nburn = 50, nvec = nsmp, nstate = tsz) 
        
        # Mean and Cov
        cvcr = numpy.cov(zmn1.T)
        mucr = numpy.mean(zmn1,axis=0)

        w, v = linalg.eig(cvcr)
        wsq = numpy.arange(w.shape[0])
        wsb = wsq[w < 1.5e-8] 
        if wsb.shape[0] > 0:
            s1 = 'Lifting %d eigenvalues' % (wsb.shape[0])
            print(s1)
            print(w[150:174])
            w[wsb] = 1.5e-8
            wdg = numpy.diagflat(w)
            cvcr = numpy.dot(v, numpy.dot(wdg,v.T))

        lgdns = stats.multivariate_normal.logpdf(zmn1, mean=mucr, cov=cvcr)
        cmpllk = numpy.sum(lgdns)
        lkdf = cmpllk - crlk
        lkstrcr = '''At EM Iteration %d,
    Log-likelihood: %.4e
    Log-like increase: %.4e 
    Minimimum Eigenvalue: %.6e''' % (critr,cmpllk,lkdf,numpy.amin(w))
        print(lkstrcr)

        crlk = cmpllk
        
        # Save results
        emout = Dataset(outfile,'r+')
        
        varlk = emout.variables['logLike']
        varlk[critr] = cmpllk
        
        varmn = emout.variables['state_mean']
        varmn[critr,:] = mucr 
 
        varcv = emout.variables['state_cov']
        varcv[critr,:,:] = cvcr 

        varest = emout.variables['state_samples']
        varest[critr,:,:] = zmn1 

        emout.close()
        
        critr = critr + 1
         
    return

def airs_raw_l2_summary(expdir, outfnm, nrep=10):
    # Extract desired AIRS L2 files directly from experiment results
    # expfl:   Name of file with experiment results
    # outfnm:  Ouptut file name
    # nrep:    Number of replicates of the reference AIRS granule

    # Experiment should have one directory per replicate

    nzairs = 100
    nzsrt = 101
    nsmpout = nrep * 45 * 30

    # Set up output (PSfc, temp profile and QC)
    qout = Dataset(outfnm,'w') 

    dimsmp = qout.createDimension('sample',nsmpout)
    dimlev = qout.createDimension('level',nzairs)

    varpsfc = qout.createVariable('PSurfStd','f4',['sample'], fill_value = -9999)
    varpsfc.long_name = 'Surface pressure'
    varpsfc.units = 'hPa'
    varpsfc.missing_value = -9999

    var2m = qout.createVariable('TSurfAir','f4',['sample'], fill_value = -9999)
    var2m.long_name = 'Near-surface air temperature'
    var2m.units = 'K'
    var2m.missing_value = -9999

    vart2qc = qout.createVariable('TSurfAir_QC','i2',['sample'], fill_value = -99)
    vart2qc.long_name = 'Near-surface air temperature QC'
    vart2qc.units = 'none'
    vart2qc.missing_value = -99

    vartmp = qout.createVariable('TAirSup','f4',['sample','level'], fill_value = -9999)
    vartmp.long_name = 'Air temperature'
    vartmp.units = 'K'
    vartmp.missing_value = -9999

    vartqc = qout.createVariable('TAirSup_QC','i2',['sample','level'], fill_value = -99)
    vartqc.long_name = 'Air temperature QC'
    vartqc.units = 'none'
    vartqc.missing_value = -99

    qout.close()

    for k in range(nrep):
        simdir = '%sindex_%d' % (expdir,k+1)
        print(simdir)
        if os.path.exists(simdir):
            flst = os.listdir(simdir)
            l2lst = []
            for j in range(len(flst)):
                if ('L2.RetSup' in flst[j]):
                    l2lst.append(flst[j])
                    
            # Sort by L2 run index
            xlst = []
            for j in range(len(l2lst)):
                l2prs = l2lst[j].split('.')
                lnl2 = len(l2prs)
                xstr = l2prs[lnl2-2]
                lnx = len(xstr)
                tstr = xstr[1:lnx]
                xlst.append(int(tstr))
            print(xlst)
            l2frm = pandas.DataFrame({'L2SupFile': l2lst, 'RunIndex': xlst})
            l2frm = l2frm.sort_values(by=['RunIndex'], ascending=[True])
            # Use only most recent 45
            l2ln = l2frm.shape[0]
            if l2ln > 45:
                lidxst = l2ln-45
                lidxfn = l2ln
            else:
                lidxst = 0
                lidxfn = l2ln
            l2frm = l2frm[lidxst:lidxfn]

            tmparr = numpy.zeros( (45,30), dtype=numpy.float32) - 9999.0
            for j in range(l2frm.shape[0]):
                l2fl = '%s/%s' % (simdir,l2frm['L2SupFile'].values[j])
                ncl2 = Dataset(l2fl)
                psfc = ncl2.variables['PSurfStd'][0,:]
                tprf = ncl2.variables['TAirSup'][0,:,:]
                tmpqc = ncl2.variables['TAirSup_QC'][0,:,:]
                t2m = ncl2.variables['TSurfAir'][0,:]
                t2mqc = ncl2.variables['TSurfAir_QC'][0,:]
                ncl2.close()

                ost = k*45*30 + j*30
                ofn = k*45*30 + (j+1)*30

                ncout = Dataset(outfnm,'r+')

                varpsfc = ncout.variables['PSurfStd']
                varpsfc[ost:ofn] = psfc

                vartmp = ncout.variables['TAirSup']
                vartmp[ost:ofn,:] = tprf

                varqc = ncout.variables['TAirSup_QC']
                varqc[ost:ofn,:] = tmpqc

                vart2m = ncout.variables['TSurfAir']
                vart2m[ost:ofn] = t2m

                var2qc = ncout.variables['TSurfAir_QC']
                var2qc[ost:ofn] = t2mqc

                ncout.close()

    return

def airscld_invtransf_stateconf_cloud9(rffl, qfl, gmmfl, outfl, stcnf, yrchc, rfmn, rfdy, rfgrn, scnrw, nrep = 10, \
                                       clearsky = False, l2dir = '/archive/AIRSOps/airs/gdaac/v6'):
    # Read in mixture model parameters and quantiles, draw random samples and set up SARTA input files
    # Use AIRS FOV cloud fraction information
    # Use state vector reference configuration
    # Use designated AIRS reference granule, and pull surface pressure temperature from there 
    # rffl:    Reference level file
    # qfl:     Template quantile file
    # gmmfl:   Gaussian mixture model results file
    # outfl:   Output file
    # stcnf:   State vector configuration file
    # yrchc:   Template Year Choice
    # rfmn:    Month for reference granule
    # rfdy:    Day for reference granule
    # rfgrn:   Reference granule number
    # scnrw:   Scan row for experiment
    # nrep:    Number of replicate granules
    # cloud:   Simulate clouds, use False for clear-sky only
    # l2dir:   Local AIRS Level 2 directory (to retrieve reference info)

    # RN Generator
    sdchc = 452546 + yrchc + rfmn*100
    random.seed(sdchc)
    cldprt = numpy.array([0.4,0.2,0.08])

    nszout = 45 * 30 * nrep
    sfrps = 45 * nrep
    nlvsrt = 98
    msgdbl = -9999.0

    # Read probs and pressure levels
    f = Dataset(rffl,'r')
    airs_sarta_levs = f.variables['level'][:]
    f.close()

    # Get reference granule info
    airsdr = '%s/%04d/%02d/%02d/airs2sup' % (l2dir,yrchc,rfmn,rfdy)
    if (os.path.exists(airsdr)):
        fllst = os.listdir(airsdr)
        l2str = 'AIRS.%04d.%02d.%02d.%03d' % (yrchc,rfmn,rfdy,rfgrn) 
        rffd = -1
        j = 0
        while ( (j < len(fllst)) and (rffd < 0) ):
            lncr = len(fllst[j])
            l4 = lncr - 4
            if ( (fllst[j][l4:lncr] == '.hdf') and (l2str in fllst[j])):
                l2fl = '%s/%s' % (airsdr,fllst[j])
                ncl2 = Dataset(l2fl)
                psfc = ncl2.variables['PSurfStd'][:,:]
                topg = ncl2.variables['topog'][:,:]
                ncl2.close()
                rffd = j
            j = j + 1
    else:
        print('L2 directory not found')

    # Surface replicates
    psfcvc = psfc[scnrw-1,:]
    topgvc = topg[scnrw-1,:]

    spres = numpy.tile(psfcvc,(sfrps,))
    salti = numpy.tile(topgvc,(sfrps,))

    # Altitude for H2O processing
    qin = Dataset(qfl,'r')
    lvs = qin.variables['level'][:]
    alts = qin.variables['Altitude_median'][:]
    qin.close()

    altrw = numpy.zeros((30,nlvsrt+3),dtype=numpy.float64)
    for i in range(30):
        # Set lowest levels to surface topog
        altrw[i,:] = alts[:]
        altrw[i, alts < topgvc[i]] = topgvc[i] 

    alth2o = numpy.tile(altrw,(sfrps,1))
    print(alth2o[15,80:100])
    print(alth2o[75,80:100])
    print(alth2o[80,80:100])
    print(alth2o.shape) 
    

    # Variable list, from configuration
    df = pandas.read_csv(stcnf, dtype = {'Order':int, 'ZScore_Name':str, 'Quantile_Name':str, \
                                         'Data_Name':str, 'Start':int, 'Length':int, 'DType':str })
    tsz = df['Length'].sum()
    szstr = '%d Total State Vector Elements' % (tsz)
    print(szstr)

    nrw = df.shape[0]
    nsmp = -1
    stctr = 0

    # Discrete/Continuous Indicator
    df['DiscCont'] = 'Continuous'
    typind = []
    stvrnms = [] 
    for q in range(nrw):
        if ( (df['Group'].values[q] == 'NumCloud') or (df['Group'].values[q] == 'CloudType') ):
            df['DiscCont'].values[q] = 'Discrete'
        for p in range(df['Length'].values[q]):
            typind.append(df['DiscCont'].values[q])
            cspt = df['Start'].values[q] + p
            vnm = '%s_%d' % (df['Data_Name'].values[q],cspt)
            stvrnms.append(vnm)

    # Quantile files 
    qin = Dataset(qfl,'r')
    prbs = qin.variables['probability'][:]
    nprb = prbs.shape[0]
    qsall = numpy.zeros((tsz,nprb))
    lvs = qin.variables['level'][:]
    alts = qin.variables['Altitude_median'][:]
    rhmd = qin.variables['RH_quantile'][:,103]
    nlvl = lvs.shape[0]
    cctr = 0
    for j in range(nrw):
        if (df['Length'].values[j] == 1):
            vr1 = df['Quantile_Name'].values[j] 
            qsall[cctr,:] = qin.variables[vr1][:]
            cctr = cctr + df['Length'].values[j] 
        elif (df['Group'].values[j] == 'CloudFrac'):
            for cl0 in range(3):
                for rw0 in range(3):
                    otst = cctr
                    otfn = cctr + 1
                    vr1 = df['Quantile_Name'].values[j] 
                    qsall[otst:otfn,:] = qin.variables[vr1][cl0,rw0,:]
                    cctr = cctr + 1
        else:
            inst = df['Start'].values[j] - 1
            infn = inst + df['Length'].values[j]
            otst = cctr
            otfn = cctr + df['Length'].values[j]
            vr1 = df['Quantile_Name'].values[j] 
            qsall[otst:otfn,:] = qin.variables[vr1][inst:infn,:]
            cctr = cctr + df['Length'].values[j] 
    qin.close()
    print('State medians')
    print(qsall[:,103])

    # Read GMM Results
    gmin = Dataset(gmmfl,'r')
    gmnms = gmin['State_Vector_Names'][:,:]
    gmmean = gmin['Mean'][:,:]
    gmpkcv = gmin['Packed_Covariance'][:,:]
    gmprps = gmin['Mixture_Proportion'][:]
    gmin.close()

    nmclps = gmnms.tolist()
    strvrs = list(map(calculate_VPD.clean_byte_list,nmclps))
    if sys.version_info[0] < 3:
        print('Version 2')
        strvrs = map(str,strvrs)
    nmix = gmmean.shape[0]
    nmxvar = gmmean.shape[1]

    mrgcv = numpy.zeros((nmix,nmxvar,nmxvar),dtype=numpy.float64)
    for j in range(nmix):
        mrgcv[j,:,:] = calculate_VPD.unpackcov(gmpkcv[j,:], nelm=nmxvar)

    # Component sizes
    dtall = numpy.zeros((nszout,nmxvar),dtype=numpy.float)
    cmpidx = numpy.zeros((nszout,),dtype=numpy.int16)
    csmp = random.multinomial(nszout,pvals=gmprps)
    cmsz = 0
    for j in range(nmix):
        cvfl = mrgcv[j,:,:]
        s1 = numpy.sqrt(numpy.diagonal(cvfl))
        crmt = calculate_VPD.cov2cor(cvfl)
        sdmt = numpy.diag(numpy.sqrt(cvfl.diagonal()))
        w, v = linalg.eig(crmt)
        print(numpy.amin(w))

        sdfn = cmsz + csmp[j]
        dtz = random.multivariate_normal(numpy.zeros((nmxvar,)),crmt,size=csmp[j])
        dttmp = numpy.tile(gmmean[j,:],(csmp[j],1)) + numpy.dot(dtz,sdmt)
        dtall[cmsz:sdfn,:] = dttmp[:,:]
        cmpidx[cmsz:sdfn] = j + 1

        cmsz = cmsz + csmp[j]

    # Re-shuffle
    ssq = numpy.arange(nszout)
    sqsmp = random.choice(ssq,size=nszout,replace=False)
    csmpshf = cmpidx[sqsmp]
    dtshf = dtall[sqsmp,:] 
    print(dtshf.shape) 

    ### Inverse Transform
    qout = numpy.zeros(dtshf.shape)
    for j in range(tsz):
        if typind[j] == 'Discrete':
            qout[:,j] = calculate_VPD.data_quantile_from_std_norm_discrete(dtshf[:,j],qsall[j,:],prbs,minval=qsall[j,0],maxval=qsall[j,nprb-1])
        else:
            qout[:,j] = calculate_VPD.data_quantile_from_std_norm(dtshf[:,j],qsall[j,:],prbs,minval=qsall[j,0],maxval=qsall[j,nprb-1])

    ### Prepare for SARTA
    varlstout = ['cngwat','cngwat2','cprbot','cprbot2','cprtop','cprtop2', \
                 'cpsize','cpsize2','cstemp','cstemp2','ctype','ctype2','salti','spres','stemp']

    # Convert to data frame
    smpfrm = pandas.DataFrame(data=qout,columns=stvrnms)
 
    dtout = numpy.zeros((nszout,len(varlstout)), dtype=numpy.float64)
    frmout = pandas.DataFrame(data=dtout,columns=varlstout)

    # Clear-sky?
    if clearsky:
        smpfrm['NumberSlab_Data_1'] = 0

    # Cloud Types
    frmout['ctype'] = (smpfrm['CType1_Data_1'] + 1.0) * 100.0 + 1.0
    frmout['ctype2'] = (smpfrm['CType2_Data_1'] + 1.0) * 100.0 + 1.0
    frmout.loc[(smpfrm.NumberSlab_Data_1 == 0),'ctype'] = msgdbl
    frmout.loc[(smpfrm.NumberSlab_Data_1 < 2),'ctype2'] = msgdbl 

    # Met/Sfc Components, arrays sized for SARTA and AIRS
    cctr = 0
    prhout = numpy.zeros((nszout,nlvsrt+3)) - 9999.0
    ptmpout = numpy.zeros((nszout,nlvsrt+3)) - 9999.0
    for j in range(nrw):
        if (df['Group'].values[j] == 'Temperature'):
            inst = df['Start'].values[j] - 1
            infn = inst + df['Length'].values[j]
            otst = cctr
            otfn = cctr + df['Length'].values[j]
            ptmpout[:,inst:infn] = qout[:,otst:otfn]
        elif (df['Group'].values[j] == 'RelHum'):
            inst = df['Start'].values[j] - 1
            infn = inst + df['Length'].values[j]
            otst = cctr
            otfn = cctr + df['Length'].values[j]
            prhout[:,inst:infn] = qout[:,otst:otfn]
            bsrh = rhmd[inst]
            for k in range(inst-1,-1,-1):
                if ma.is_masked(rhmd[k]):
                    prhout[:,k] = bsrh / 2.0
                    t2 = 'RH masked: %d' % (k)
                    print(t2)
                elif rhmd[k] < 0:
                    t2 = 'RH below 0: %d' % (k)
                    print(t2)
                    prhout[:,k] = bsrh
                else:
                    prhout[:,k] = rhmd[k]
                    bsrh = rhmd[k]
        elif (df['Group'].values[j] == 'Surface'):
            frmout['stemp'] = qout[:,cctr]
        cctr = cctr + df['Length'].values[j] 

    str1 = '''RH at Level 1: %.4e, %.4e ''' % (numpy.amin(prhout[:,0]),rhmd[0])
    str2 = '''RH at Level 2: %.4e, %.4e ''' % (numpy.amin(prhout[:,1]),rhmd[1])
    print(str1)
    print(str2)
    h2oout = calculate_VPD.calculate_h2odens(prhout,ptmpout,airs_sarta_levs,alth2o)

    # Surface from reference
    frmout['salti'] = salti
    # Need for clouds
    frmout['spres'] = spres 
    #smpfrm['SfcPres'] = spres 

    # Pressure Variables
    for i in range(nszout):
        if smpfrm['NumberSlab_Data_1'][smpfrm.index[i]] == 0:
            frmout.at[i,'cprbot'] = msgdbl 
            frmout.at[i,'cprtop'] = msgdbl
            frmout.at[i,'cprbot2'] = msgdbl 
            frmout.at[i,'cprtop2'] = msgdbl
        elif smpfrm['NumberSlab_Data_1'][smpfrm.index[i]] == 1:
            tmplgts = numpy.array( [smpfrm['CloudBot1Logit_Data_1'][smpfrm.index[i]], \
                                    smpfrm['DPCloud1Logit_Data_1'][smpfrm.index[i]] ] )
            frctmp = calculate_VPD.lgttoprp(tmplgts)
            frmout.at[i,'cprbot'] = spres[i] * (1.0 - frctmp[0])
            frmout.at[i,'cprtop'] = spres[i] * (1.0 - frctmp[0] - frctmp[1])
            frmout.at[i,'cprbot2'] = msgdbl 
            frmout.at[i,'cprtop2'] = msgdbl
        elif smpfrm['NumberSlab_Data_1'][smpfrm.index[i]] == 2:
            tmplgts = numpy.array( [smpfrm['CloudBot1Logit_Data_1'][smpfrm.index[i]], \
                                    smpfrm['DPCloud1Logit_Data_1'][smpfrm.index[i]], \
                                    smpfrm['DPSlabLogit_Data_1'][smpfrm.index[i]], \
                                    smpfrm['DPCloud2Logit_Data_1'][smpfrm.index[i]] ] )
            frctmp = calculate_VPD.lgttoprp(tmplgts)
            frmout.at[i,'cprbot'] = spres[i] * (1.0 - frctmp[0])
            frmout.at[i,'cprtop'] = spres[i] * (1.0 - frctmp[0] - frctmp[1])
            frmout.at[i,'cprbot2'] = spres[i] * (1.0 - frctmp[0] - frctmp[1] - frctmp[2])
            frmout.at[i,'cprtop2'] = spres[i] * (1.0 - frctmp[0] - frctmp[1] - frctmp[2] - frctmp[3])

    # Non-Gas Water
    frmout['cngwat'] = smpfrm['NGWater1_Data_1']
    frmout.loc[(smpfrm.NumberSlab_Data_1 == 0),'cngwat'] = msgdbl
    frmout['cngwat2'] = smpfrm['NGWater2_Data_1']
    frmout.loc[(smpfrm.NumberSlab_Data_1 < 2),'cngwat2'] = msgdbl

    # Temperature
    frmout['cstemp'] = smpfrm['CTTemp1_Data_1']
    frmout.loc[(smpfrm.NumberSlab_Data_1 == 0),'cstemp'] = msgdbl
    frmout['cstemp2'] = smpfrm['CTTemp2_Data_1']
    frmout.loc[(smpfrm.NumberSlab_Data_1 < 2),'cstemp2'] = msgdbl

    # Particle Size, from Sergio's paper
    # 20 for water, 80 for ice
             #'cpsize','cpsize2','cstemp','cstemp2','ctype','ctype2']
    frmout.loc[(frmout.ctype == 101.0),'cpsize'] = 20 
    frmout.loc[(frmout.ctype == 201.0),'cpsize'] = 80
    frmout.loc[(frmout.ctype < 0.0),'cpsize'] = msgdbl

    frmout.loc[(frmout.ctype2 == 101.0),'cpsize2'] = 20 
    frmout.loc[(frmout.ctype2 == 201.0),'cpsize2'] = 80
    frmout.loc[(frmout.ctype2 < 0.0),'cpsize2'] = msgdbl

    # Fractions, 3D Arrays
    cfrc1out = numpy.zeros((nszout,3,3)) - 9999.0
    cfrc2out = numpy.zeros((nszout,3,3)) - 9999.0
    cfrc12out = numpy.zeros((nszout,3,3)) - 9999.0
    for i in range(nszout):
        cldctr = 0
        if smpfrm['NumberSlab_Data_1'][smpfrm.index[i]] == 0:
            cfrc1out[i,:,:] = 0.0
            cfrc2out[i,:,:] = 0.0
            cfrc12out[i,:,:] = 0.0
        elif smpfrm['NumberSlab_Data_1'][smpfrm.index[i]] == 1:
            for q in range(3):
                for p in range(3):
                    cfcnm = 'TotCFrc_Data_%d' % (cldctr + 1)
                    cfrc1out[i,q,p] = smpfrm[cfcnm].values[i]
                    cldctr = cldctr + 1
            cfrc2out[i,:,:] = 0.0
            cfrc12out[i,:,:] = 0.0
        elif smpfrm['NumberSlab_Data_1'][smpfrm.index[i]] == 2:
            for q in range(3):
                for p in range(3):
                    cfcnm = 'TotCFrc_Data_%d' % (cldctr + 1)
                    lg2nm = 'CFrcLogit2_Data_%d' % (cldctr + 1)
                    lg12nm = 'CFrcLogit12_Data_%d' % (cldctr + 1)
                    tcfrc = smpfrm[cfcnm].values[i]
                    
                    zlgt2 = smpfrm[cfcnm].values[i]
                    zlgt12 = smpfrm[cfcnm].values[i]
                    tmplgts = numpy.array( [zlgt2, zlgt12] )
                    frctmp = calculate_VPD.lgttoprp(tmplgts)
 
                    cfrc1out[i,q,p] = tcfrc * (frctmp[2] + frctmp[1])
                    cfrc2out[i,q,p] = tcfrc * (frctmp[0] + frctmp[1])
                    cfrc12out[i,q,p] = tcfrc * (frctmp[1])

                    cldctr = cldctr + 1


    # Write Sample Output
    print(frmout[166:180])

    fldbl = numpy.array([-9999.],dtype=numpy.float64)
    flflt = numpy.array([-9999.],dtype=numpy.float32)
    flshrt = numpy.array([-99],dtype=numpy.int16)

    f = h5py.File(outfl,'w')
    for j in range(len(varlstout)): 
        dftmp = f.create_dataset(varlstout[j],data=frmout[varlstout[j]])
        dftmp.attrs['missing_value'] = -9999.
        dftmp.attrs['_FillValue'] = -9999.
    dfpt = f.create_dataset('ptemp',data=ptmpout)
    dfpt.attrs['missing_value'] = fldbl 
    dfpt.attrs['_FillValue'] = fldbl
    dfrh = f.create_dataset('relative_humidity',data=prhout)
    dfrh.attrs['missing_value'] = fldbl
    dfrh.attrs['_FillValue'] = fldbl
    dfgs = f.create_dataset('gas_1',data=h2oout)
    dfgs.attrs['missing_value'] = fldbl
    dfgs.attrs['_FillValue'] = fldbl
    dfcf1 = f.create_dataset('cfrac',data=cfrc1out)
    dfcf1.attrs['missing_value'] = fldbl
    dfcf1.attrs['_FillValue'] = fldbl
    dfcf2 = f.create_dataset('cfrac2',data=cfrc2out)
    dfcf2.attrs['missing_value'] = fldbl
    dfcf2.attrs['_FillValue'] = fldbl
    dfcf12 = f.create_dataset('cfrac12',data=cfrc12out)
    dfcf12.attrs['missing_value'] = fldbl
    dfcf12.attrs['_FillValue'] = fldbl
    dfcsmp = f.create_dataset('mixture_component',data=csmpshf)
    dfcsmp.attrs['missing_value'] = flshrt 
    dfcsmp.attrs['_FillValue'] = flshrt
    dflv = f.create_dataset('level',data=airs_sarta_levs)
    f.close()

    return

def extract_airs_supp(rffl, qfl, gmmfl, outfl, stcnf, yrchc, rfmn, rfdy, rfgrn, scnrw, \
                                       l2dir = '/archive/AIRSOps/airs/gdaac/v6'):
    # Use designated AIRS reference granule, and pull reference information 
    # dtdr:    Output directory
    # yrchc:   Template Year Choice
    # hrchc:   Template Hour Choice
    # rgchc:   Template Region Choice
    # rfmn:    Month for reference granule
    # rfdy:    Day for reference granule
    # rfgrn:   Reference granule number
    # scnrw:   Scan row for experiment
    # nrep:    Number of replicate granules
    # l2dir:   Local AIRS Level 2 directory (to retrieve reference info)

    #sfrps = 45 * nrep
    #nlvsrt = 98
    msgdbl = -9999.0

    # Read probs and pressure levels
    f = Dataset(rffl,'r')
    airs_sarta_levs = f['level'][:]
    f.close()

    # Get reference granule info
    airsdr = '%s/%04d/%02d/%02d/airs2sup' % (l2dir,yrchc,rfmn,rfdy)
    if (os.path.exists(airsdr)):
        fllst = os.listdir(airsdr)
        l2str = 'AIRS.%04d.%02d.%02d.%03d' % (yrchc,rfmn,rfdy,rfgrn) 
        rffd = -1
        j = 0
        while ( (j < len(fllst)) and (rffd < 0) ):
            lncr = len(fllst[j])
            l4 = lncr - 4
            if ( (fllst[j][l4:lncr] == '.hdf') and (l2str in fllst[j])):
                l2fl = '%s/%s' % (airsdr,fllst[j])
                ncl2 = Dataset(l2fl)
                psfc = ncl2.variables['PSurfStd'][:,:]
                topg = ncl2.variables['topog'][:,:]
                freqemis = ncl2.variables['freqEmis'][:,:,:]
                emisIR = ncl2.variables['emisIRStd'][:,:,:]
                lndfrc = ncl2.variables['landFrac'][:,:]
                tsrf = ncl2.variables['TSurfStd'][:,:]
                print('Emis Freq Dims')
                print(freqemis.shape)
                print('Emis IR Dims')
                print(emisIR.shape)
                print('Land Frac Dims')
                print(lndfrc.shape)
                latfor = ncl2.variables['Latitude'][:,:]
                lonfor = ncl2.variables['Longitude'][:,:]
                ncl2.close()
                rffd = j
            j = j + 1
    else:
        print('L2 directory not found')

    latout = latfor[scnrw-1,:]
    lonout = lonfor[scnrw-1,:]
    pscvc = psfc[scnrw-1,:]
    topgvc = topg[scnrw-1,:]
    emsv = emisIR[scnrw-1,:,:]
    frqemsv = freqemis[scnrw-1,:,:]
    lfrcout = lndfrc[scnrw-1,:]
    tsrfout = tsrf[scnrw-1,:]

    fldbl = numpy.array([-9999.],dtype=numpy.float64)
    flflt = numpy.array([-9999.],dtype=numpy.float32)
    flshrt = numpy.array([-99],dtype=numpy.int16)

    f = h5py.File(outfl,'w')
    dflt = f.create_dataset('Latitude',data=latout)
    dfln = f.create_dataset('Longitude',data=lonout)
    dfps = f.create_dataset('PSurfStd',data=pscvc)
    dftp = f.create_dataset('topog',data=topgvc)
    dfem = f.create_dataset('emisIRStd',data=emsv)
    dffq = f.create_dataset('freqEmis',data=frqemsv)
    dflf = f.create_dataset('landFrac',data=lfrcout)
    dtsf = f.create_dataset('TSurfStd',data=tsrfout)
    f.close()

    return

def airs_post_match_l2(flnm, tmidx, tmday, lats, lons, mskarr, rgnfrm, \
                       gmmdir, nsmp = 0, msgvl = -9999, \
                       l2srch = '/archive/AIRSOps/airs/gdaac/v6'):
    # Match AIRS Level 2 to region masks and execute posterior analysis 
    # flnm:    Name of output file (NetCDF expected)
    # tms:     Time index in output
    # tmday:   Datetime object with time information
    # lats:    Longitude variable array
    # lons:    Longitude variable array
    # mskarr:  Region mask array
    # rgnfrm:  Data frame with region indicators
    # gmmdir:  Directory with GMM results
    # nsmp:    Number of posterior samples to draw (optional)
    # msgvl:   Missing value
    # l2srch:  Level 2 search directory
  

    # Search AIRS Level 2
    airsdr = '%s/%04d/%02d/%02d/airs2ret' % (l2srch,tmday.year,tmday.month,tmday.day)
    asupdr = '%s/%04d/%02d/%02d/airs2sup' % (l2srch,tmday.year,tmday.month,tmday.day)

    dsclst = []
    asclst = []

    nlat = lats.shape[0]
    nlon = lons.shape[0]

    lonmn = lons[0] - 5.0
    lonmx = lons[nlon-1] + 5.0
    latmn = lats[0] - 5.0
    latmx = lats[nlat-1] + 5.0
    d0 = datetime.datetime(1993,1,1,0,0,0)
    ddif = tmday - d0
    bsdif = ddif.total_seconds()

    # Seed
    sdchc = 151444 + tmday.year*10 + int(bsdif)
    random.seed(sdchc)

    # Set up reference frame, with region mask info
    ltrp = numpy.repeat(lats,nlon)
    ltidx = numpy.repeat(numpy.arange(nlat),nlon)
    lnrp = numpy.tile(lons,nlat)
    lnidx = numpy.tile(numpy.arange(nlon),nlat)
    mskflt = mskarr.flatten()
    merfrm = pandas.DataFrame({'GridLonIdx': lnidx, 'GridLatIdx': ltidx, \
                               'GridLon': lnrp, 'GridLat': ltrp, 'RgnMask': mskflt})
    merfrm.loc[pandas.isnull(merfrm.RgnMask),'RgnMask'] = -99
    merfrm['RgnMask'] = merfrm['RgnMask'].astype('int32')

    if (os.path.exists(asupdr)):
        # Set up a list of files/granules that might match up
        fllst = os.listdir(asupdr)
        #print(fllst)

        for j in range(len(fllst)):
            lncr = len(fllst[j])
            l4 = lncr - 4
            if (fllst[j][l4:lncr] == '.hdf'):
                l2fl = '%s/%s' % (asupdr,fllst[j])
                ncl2 = Dataset(l2fl)
                slrzn = ncl2.variables['solzen'][:,:]
                l2lat = ncl2.variables['Latitude'][:,:]
                l2lon = ncl2.variables['Longitude'][:,:]
                l2tm = ncl2.variables['Time'][:,:]
                ncl2.close()

                # Check lat/lon ranges and asc/dsc
                l2tmdf = numpy.absolute(l2tm - bsdif)
                l2mntm = numpy.min(l2tmdf)

                # Within 4 hours
                if l2mntm < 14400.0:
                   ltflt = l2lat.flatten()
                   lnflt = l2lon.flatten()
                   latsb = ltflt[(ltflt >= latmn) & (ltflt <= latmx)]
                   lonsb = lnflt[(lnflt >= lonmn) & (lnflt <= lonmx)]
                   if ( (latsb.shape[0]  > 0) and (lonsb.shape[0] > 0) ):
                       asclst.append(fllst[j])
                       #sstr = '%s %.2f' % (fllst[j], l2mntm)
                       #print(sstr)

    # Region posterior model names
    hrcr = rgnfrm['Hour'].values[0]
    abrvcr = rgnfrm['Abbrev'].values[0]
    sstr = rgnfrm['Season'].values[0]
    rgnfl = '%s/PostGMM_%s_%s_%02dUTC.nc' % (gmmdir,abrvcr,sstr,hrcr)
    ncgm = Dataset(rgnfl)
    rtnms = ncgm.variables['state_names_retrieved'][:]
    lvs = ncgm.variables['level'][:]
    ncgm.close()
    nmclps = rtnms.tolist()
    strvrs = list(map(calculate_VPD.clean_byte_list,nmclps))

    # General AIRS level sequence
    lsqair = numpy.arange(34,97)
    lv850air = 90

    tairpc = 0
    for i in range(len(strvrs)):
        if ('TempPC' in strvrs[i]):
            tairpc = tairpc + 1 
    strpc = 'Number TAir PCs: %d'  % (tairpc)
    print(strpc)

    # Region GMM output, grab eigenvectors
    nrgn = rgnfrm.shape[0]
    nlev = lvs.shape[0]

    maxcmp = 0

    if tairpc > 0:
        taireig = numpy.zeros((nrgn,tairpc,nlev),dtype=numpy.float64)
        tairmnvc = numpy.zeros((nrgn,nlev),dtype=numpy.float64)

    for j in range(nrgn):
        hrcr = rgnfrm['Hour'].values[j]
        abrvcr = rgnfrm['Abbrev'].values[j]
        sstr = rgnfrm['Season'].values[j]
        rgnfl = '%s/PostGMM_%s_%s_%02dUTC.nc' % (gmmdir,abrvcr,sstr,hrcr)
        ncgm = Dataset(rgnfl)
        gmm_prp = ncgm.variables['mixture_proportion'][:]
        ncgm.close()

        nmxcmp = gmm_prp.shape[0]
        if nmxcmp > maxcmp:
            maxcmp = nmxcmp

        if tairpc > 0:
            ncgm = Dataset(rgnfl)
            tairmnvc[j,:] = ncgm.variables['temp_prof_mean'][:]
            taireig[j,:,:] = ncgm.variables['temp_eigenvector'][0:tairpc,:]
            ncgm.close()

            # Check temp prof
            tmpmx = numpy.nanmax(tairmnvc[j,:])
            for q1 in range(tairmnvc.shape[1]):
                if numpy.isnan(tairmnvc[j,q1]):
                    tairmnvc[j,q1] = tmpmx


    # Level 2 processing
    # Extract lat, lon, granules
    # Additional processing
    # 1. Cloud summaries
    # 2. PCA of vertical profiles (region specific)
    tmch = 0
    if (len(asclst) > 0):
        # Start matchups
        for j in range(len(asclst)):
            l2fl = '%s/%s' % (asupdr,asclst[j])
            ncl2 = Dataset(l2fl)
            l2lat = ncl2.variables['Latitude'][:,:]
            l2lon = ncl2.variables['Longitude'][:,:]
            l2tm = ncl2.variables['Time'][:,:]
            cfrcair = ncl2.variables['CldFrcStd'][:,:,:,:,:]
            cfrcaqc = ncl2.variables['CldFrcStd_QC'][:,:,:,:,:]
            tsfcqc = ncl2.variables['TSurfAir_QC'][:,:]
            tsfair = ncl2.variables['TSurfAir'][:,:]
            tsferr = ncl2.variables['TSurfAirErr'][:,:]
            psfc = ncl2.variables['PSurfStd'][:,:]
            tairsp = ncl2.variables['TAirSup'][:,:,:]
            ncldair = ncl2.variables['nCld'][:,:,:,:]
            ncl2.close()
      
            nairtrk = l2lat.shape[0]
            nairxtk = l2lat.shape[1]

            # Extract granule
            asplt = asclst[j].split('.')
            grnchc = asplt[4]

            frctot = cfrcair[:,:,:,:,0] + cfrcair[:,:,:,:,1]
            cldsmarr = numpy.zeros((nairtrk,nairxtk,4),frctot.dtype)
            ncldmx = numpy.zeros((nairtrk,nairxtk),ncldair.dtype)
            for q1 in range(nairtrk):
                for p1 in range(nairxtk):
                    cldsmarr[q1,p1,:] = calculate_VPD.cloud_frac_summary(frctot[q1,p1,:,:])
                    ncldmx[q1,p1] = numpy.amax(ncldair[q1,p1,:,:])

            # Data Frame
            tkidx = numpy.repeat(numpy.arange(nairtrk),nairxtk)
            xtidx = numpy.tile(numpy.arange(nairxtk),nairtrk)
            l2lnflt = l2lon.flatten().astype(numpy.float64)
            l2ltflt = l2lat.flatten().astype(numpy.float64)
            l2tmflt = l2tm.flatten().astype(numpy.float64)
            l2frm = pandas.DataFrame({'L2LonIdx': xtidx, 'L2LatIdx': tkidx, \
                                      'L2Lon': l2lnflt, 'L2Lat': l2ltflt, 'L2Time': l2tmflt})
            l2frm['GridLon'] = numpy.around(l2frm['L2Lon']/0.625) * 0.625
            l2frm['GridLat'] = numpy.around(l2frm['L2Lat']/0.5) * 0.5
            l2frm['Granule'] = int(grnchc)

            # Sfc info
            sfcspt = calculate_VPD.sfclvl(psfc,lvs)
            sfcspt = sfcspt + lsqair[0]

            tdftmp = tairsp[:,:,lv850air] - tsfair
            for q1 in range(nairtrk):
                for p1 in range(nairxtk):
                    if sfcspt[q1,p1] <= lv850air:
                        tdftmp[q1,p1] = tairsp[q1,p1,sfcspt[q1,p1]-2] - tsfair[q1,p1]
                        #str1 = 'Sfc below 850 hPa: %d, %d, %.4f' % (q1,p1,tdftmp[q1,p1])
                        #print(str1)

            ttmp = tsfair.flatten()
            tdftmp = tdftmp.flatten()
            ertmp = tsferr.flatten()

            if ttmp.dtype.byteorder == '>':
                l2frm['NSTRtrv'] = ttmp.byteswap().newbyteorder()
            else:
                l2frm['NSTRtrv'] = ttmp
            if ertmp.dtype.byteorder == '>':
                l2frm['NSTL2Err'] = ertmp.byteswap().newbyteorder()
            else:
                l2frm['NSTL2Err'] = ertmp
            if tdftmp.dtype.byteorder == '>':
                l2frm['TDif850'] = tdftmp.byteswap().newbyteorder()
            else:
                l2frm['TDif850'] = tdftmp
            qcmp = tsfcqc.flatten()
            if qcmp.dtype.byteorder == '>':
                l2frm['NSTRtrvQF'] = qcmp.byteswap().newbyteorder()
            else:
                l2frm['NSTRtrvQF'] = qcmp
            ncldtmp = ncldmx.flatten()
            if ncldtmp.dtype.byteorder == '>':
                l2frm['NCloud'] = ncldtmp.byteswap().newbyteorder()
            else:
                l2frm['NCloud'] = ncldtmp
            psftmp = psfc.flatten()
            if ncldtmp.dtype.byteorder == '>':
                l2frm['PSfc'] = psftmp.byteswap().newbyteorder()
            else:
                l2frm['PSfc'] = psftmp

            l2frm['CFrcMean'] = cldsmarr[:,:,0].flatten()
            l2frm['CFrcSD'] = cldsmarr[:,:,1].flatten()
            l2frm['NClr'] = cldsmarr[:,:,2].flatten()
            l2frm['NOvc'] = cldsmarr[:,:,3].flatten()

            # Set up temp PCs
            tpcnms = []
            for t in range(tairpc):
                pcnm = 'TempPC%d' % (t+1)
                tpcnms.append(pcnm)
                l2frm[pcnm] = numpy.zeros( (l2frm.shape[0],), dtype=cldsmarr.dtype) 

            l2mrg = pandas.merge(l2frm,merfrm,on=['GridLon','GridLat'])
            l2mrg = l2mrg[l2mrg['RgnMask'] >= 0]
            print(l2mrg.shape)
            nl2 = l2mrg.shape[0]
            if nl2 > 0:
                # PCA processing
                nlv = lsqair.shape[0]
                lsq = numpy.arange(nlv)
                for i in range(nl2):
                    rgidx = l2mrg['RgnMask'].values[i] - 1
                    atrk = l2mrg['L2LatIdx'].values[i]
                    ctrk = l2mrg['L2LonIdx'].values[i] 

                    tprftmp = tairsp[atrk,ctrk,lsqair] 
                    tprftmp = ma.masked_where(tprftmp < 0,tprftmp)
                    msq = ma.is_masked(tprftmp)
                    tprfscr = ma.filled(tprftmp, fill_value=tairmnvc[rgidx,:]) 

                    tpcsr = numpy.dot(taireig[rgidx,:,:],tprfscr)
                    for t in range(tairpc):
                        pcnm = 'TempPC%d' % (t+1)
                        l2mrg[pcnm].values[i] = tpcsr[t]

                    if ( (i % 100) == 0):
                        rgstr = 'Region %d' % (rgidx+1)
                        print(rgstr)
                        print(tprfscr[50:nlv])
                        print(tpcsr)
                # Append to master frame
                if tmch == 0:
                    mrg_out = l2mrg
                else:
                    mrg_out = mrg_out.append(l2mrg,ignore_index=True)
            if (nl2 > 50):
                print(l2mrg[20:30])
                print(l2mrg.columns)

            tmch = tmch + nl2 
            

    # Loop through regions and match
    # Region GMM output
    nrgn = rgnfrm.shape[0]
    totsdg = 0
    print(strvrs)
    for j in range(nrgn):
        hrcr = rgnfrm['Hour'].values[j]
        abrvcr = rgnfrm['Abbrev'].values[j]
        sstr = rgnfrm['Season'].values[j]
        rgnfl = '%s/PostGMM_%s_%s_%02dUTC.nc' % (gmmdir,abrvcr,sstr,hrcr)
        ncgm = Dataset(rgnfl)
        #rtnms = ncgm.variables['state_names_retrieved'][:]
        gmm_prp = ncgm.variables['mixture_proportion'][:]
        gmm_mux = ncgm.variables['mean_true'][:,:]
        gmm_muy = ncgm.variables['mean_retrieved'][:,:]
        gmm_varx = ncgm.variables['varcov_true'][:,:,:]
        gmm_varxy = ncgm.variables['varcov_cross'][:,:,:]
        gmm_vary = ncgm.variables['varcov_retrieved'][:,:,:]
        gmm_prcy = ncgm.variables['precmat_retrieved'][:,:,:]
        gmm_pstvarx = ncgm.variables['varcov_post_true'][:,:,:]
        ncgm.close()

        frmsb = mrg_out[mrg_out['RgnMask'] == (j+1)]
        print(frmsb.shape)
        nsdg = frmsb.shape[0]

        nmxcmp = gmm_prp.shape[0]
        nrtrv = gmm_muy.shape[1]
        nrtbs = nrtrv - tairpc
        nxprd = gmm_mux.shape[1]

        # Set up a data array
        print(abrvcr)
        ydattmp = numpy.zeros((nsdg,nrtrv),dtype=numpy.float64)
        for q in range(nrtbs):
            ydattmp[:,q] = frmsb[strvrs[q]]
        for q in range(tairpc):
            ydattmp[:,q+nrtbs] = frmsb[tpcnms[q]] 
        print(ydattmp[0:4,:]) 
     
        ## Apply GMM, from gmm_post_pred in airs_post_expt_support.R 
        # Densities
        f_y_c = numpy.zeros((nsdg,nmxcmp),dtype=numpy.float64)
        #p_c_y = numpy.zeros((nsdg,nmxcmp),dtype=numpy.float64)
        print('Computing f_y_c')
        for k in range(nmxcmp):
            w, v = linalg.eig(gmm_vary[k,:,:])
            wsq = numpy.arange(w.shape[0])
            wsb = wsq[w < 5.0e-5]
            if wsb.shape[0] > 0:
                s1 = 'Lifting %d eigenvalues' % (wsb.shape[0])
                print(s1)
                w[wsb] = 5.0e-5
                wdg = numpy.diagflat(w)
                gmm_vary[k,:,:] = numpy.dot(v, numpy.dot(wdg,v.T))
            w, v = linalg.eig(gmm_vary[k,:,:])
            print(numpy.amin(w))
            if nrtrv > 1:
                f_y_c[:,k] = stats.multivariate_normal.logpdf(ydattmp, mean=gmm_muy[k,:], cov=gmm_vary[k,:,:])
            elif ntrv == 1:
                # Univariate density
                ltr = 0
        # Adjust for possible underflow
        mxdns = numpy.amax(f_y_c,axis=1)
        mxarr = numpy.transpose(numpy.tile(mxdns,reps=(nmxcmp,1)))
        adjdns = f_y_c - mxarr

        # Compute the conditional probabilities, p_c_y
        print('computing p_c_y')
        prprep = numpy.tile(gmm_prp,reps=(nsdg,1))
        cmplk = prprep * numpy.exp(adjdns)
        sumlk = numpy.sum(cmplk,axis=1)
        sumrep = numpy.transpose(numpy.tile(sumlk,reps=(nmxcmp,1))) 
        cmpprb = cmplk / sumrep

        print('predicting E_X_Y')
        ex_y_c = numpy.zeros((nsdg,nxprd,nmxcmp),dtype=numpy.float64)
        ex_y = numpy.zeros((nsdg,nxprd),dtype=numpy.float64)
        for k in range(nmxcmp):
            muxrp = numpy.tile(gmm_mux[k,:],reps=(nsdg,1))
            muyrp = numpy.tile(gmm_muy[k,:],reps=(nsdg,1))
            ydevcr = ydattmp - muyrp
            prcdev = numpy.dot(gmm_prcy[k,:,:], numpy.transpose(ydevcr))
            cvxytmp = numpy.transpose(gmm_varxy[k,:,:])
            ex_y_c[:,:,k] = muxrp + numpy.transpose(numpy.dot(cvxytmp,prcdev))
        print(prcdev.shape) 
        print(muxrp.shape)
        print(muyrp.shape)            
        for k in range(nxprd):
            cmpmns = cmpprb * ex_y_c[:,k,:]
            ex_y[:,k] = numpy.sum(cmpmns,axis=1)
       
        print(ex_y[8:12,0])
        print('predicting Sigma_X_Y')
        Sigma_X_Y_C_bet = numpy.zeros((nsdg,nxprd,nxprd,nmxcmp),dtype=numpy.float64)
        Sigma_X_Y_C_wth = numpy.zeros((nsdg,nxprd,nxprd,nmxcmp),dtype=numpy.float64)
        Sigma_X_Y = numpy.zeros((nsdg,nxprd,nxprd),dtype=numpy.float64)
        for k in range(nmxcmp):
            wthcv = gmm_pstvarx[k,:,:]
            mndv = ex_y_c[:,:,k] - ex_y
            prbrp = numpy.repeat(cmpprb[:,k],nxprd*nxprd)
            prbrp = numpy.reshape(prbrp,(nsdg,nxprd,nxprd))
            wthrp = numpy.tile(wthcv.flatten(),nsdg)
            wthrp = numpy.reshape(wthrp,(nsdg,nxprd,nxprd))
            Sigma_X_Y_C_wth[:,:,:,k] = wthrp
            for i in range(nsdg):
                Sigma_X_Y_C_bet[i,:,:,k] = numpy.outer(mndv[i,:],mndv[i,:])
            Sigma_X_Y = Sigma_X_Y + prbrp * (Sigma_X_Y_C_wth[:,:,:,k] + Sigma_X_Y_C_bet[:,:,:,k])
        print(prbrp.shape)
        print(wthrp.shape)

        # Optionally sample
        # Posterior samples
        if nsmp > 0:
            smpsv = numpy.zeros((nsdg,nsmp,nxprd),dtype=numpy.float)
            skwsv = numpy.zeros((nsdg,nxprd),dtype=numpy.float)
            kursv = numpy.zeros((nsdg,nxprd),dtype=numpy.float)

            for i in range(nsdg):
                tmpsmp = numpy.zeros((nsmp,nxprd),dtype=numpy.float)
                cmpidx = numpy.zeros((nsmp,),dtype=numpy.int16)
                csmp = random.multinomial(nsmp,pvals = cmpprb[i,:])

                cmsz = 0
                for k in range(nmxcmp):
                    if csmp[k] > 0:
                        sdfn = cmsz + csmp[k]
                        dtz = random.multivariate_normal(numpy.zeros((nxprd,)), gmm_pstvarx[k,:,:], size=csmp[k]) 
                        dttmp = numpy.tile(ex_y_c[i,:,k],(csmp[k],1)) + dtz
                        tmpsmp[cmsz:sdfn,:] = dttmp[:,:]
                        cmpidx[cmsz:sdfn] = k + 1

                        cmsz = cmsz + csmp[k]
                # Re-shuffle
                ssq = numpy.arange(nsmp)
                sqsmp = random.choice(ssq,size=nsmp,replace=False)
                cmpshf = cmpidx[sqsmp]
                smpsv[i,:,:] = tmpsmp[sqsmp,:]
            for s1 in range(nxprd):
                skwsv[:,s1] = stats.skew(smpsv[:,:,s1],axis=1)
                kursv[:,s1] = stats.kurtosis(smpsv[:,:,s1],axis=1,fisher=True)
                #print(skwtmp.shape)
                #strskw = '  Skew %.3f \n' % (skwtmp[10])
                #print(strskw)

        # Create/update output arrays
        #   Region Indicator
        #   AIRS cross-track index
        #   AIRS along-track index
        #   Latitude
        #   Longitude
        #   Time
        #   Granule
        #   AIRS quality flag
        #   Predictor data array 
        #   Posterior mean array
        #   Posterior (co)variance array

        if totsdg == 0:
            rgout = numpy.zeros((nsdg,),dtype=numpy.int16)
            rgout[:] = j + 1 
            qfout = numpy.zeros((nsdg,),dtype=numpy.int16)
            qfout[:] = frmsb['NSTRtrvQF']
            lnidxout = numpy.zeros((nsdg,),dtype=numpy.int16)
            lnidxout[:] = frmsb['L2LonIdx']
            ltidxout = numpy.zeros((nsdg,),dtype=numpy.int16)
            ltidxout[:] = frmsb['L2LatIdx']
            grnout = numpy.zeros((nsdg,),dtype=numpy.int16)
            grnout[:] = frmsb['Granule']
            latout = numpy.zeros((nsdg,),dtype=numpy.float32)
            latout[:] = frmsb['L2Lat']
            lonout = numpy.zeros((nsdg,),dtype=numpy.float32)
            lonout[:] = frmsb['L2Lon']
            tmout = numpy.zeros((nsdg,),dtype=numpy.float64)
            tmout[:] = frmsb['L2Time']
            psfout = numpy.zeros((nsdg,),dtype=numpy.float32)
            psfout[:] = frmsb['PSfc']
            l2errout = numpy.zeros((nsdg,),dtype=numpy.float32)
            l2errout[:] = frmsb['NSTL2Err']
            prdmnout = numpy.zeros((nsdg,nxprd),dtype=numpy.float32)
            prdmnout[:,:] = ex_y
            if nsmp > 0:
                skwout = numpy.zeros((nsdg,nxprd),dtype=numpy.float32)
                skwout[:,:] = skwsv
                kurout = numpy.zeros((nsdg,nxprd),dtype=numpy.float32)
                kurout[:,:] = kursv
                smpout = numpy.zeros((nsdg,nsmp,nxprd),dtype=numpy.float32)
                smpout[:,:,:] = smpsv
            sigxyout = numpy.zeros((nsdg,nxprd,nxprd),dtype=numpy.float32)
            sigxyout[:,:] = Sigma_X_Y
            rtryout = numpy.zeros((nsdg,nrtrv),dtype=numpy.float32)
            rtryout[:,:] = ydattmp
            cmpprbout = numpy.zeros((nsdg,maxcmp),dtype=numpy.float32)
            cmpprbout[:,0:nmxcmp] = cmpprb
        else:
            rgtmp = numpy.zeros((nsdg,),dtype=numpy.int16)
            rgtmp[:] = j + 1 
            rgout = numpy.append(rgout,rgtmp)
            qftmp = frmsb['NSTRtrvQF']
            qfout = numpy.append(qfout,qftmp) 
            errtmp = frmsb['NSTL2Err']
            l2errout = numpy.append(l2errout,errtmp)
            lnidxtmp = frmsb['L2LonIdx']
            lnidxout = numpy.append(lnidxout,lnidxtmp) 
            ltidxtmp = frmsb['L2LatIdx']
            ltidxout = numpy.append(ltidxout,ltidxtmp) 
            grntmp = frmsb['Granule']
            grnout = numpy.append(grnout,grntmp)
            lontmp = frmsb['L2Lon']
            lonout = numpy.append(lonout,lontmp)
            lattmp = frmsb['L2Lat']
            latout = numpy.append(latout,lattmp)
            tmtmp = frmsb['L2Time']
            tmout = numpy.append(tmout,tmtmp)
            psftmp = frmsb['PSfc']
            psfout = numpy.append(psfout,psftmp)
            prdmnout = numpy.append(prdmnout,ex_y,axis=0)  
            if nsmp > 0:
                skwout = numpy.append(skwout,skwsv,axis=0)  
                kurout = numpy.append(kurout,kursv,axis=0)  
                smpout = numpy.append(smpout,smpsv,axis=0)
            rtryout = numpy.append(rtryout,ydattmp,axis=0) 
            sigxyout = numpy.append(sigxyout,Sigma_X_Y,axis=0)   
            cmpprbtmp = numpy.zeros((nsdg,maxcmp),dtype=numpy.float32)
            cmpprbtmp[:,0:nmxcmp] = cmpprb
            cmpprbout = numpy.append(cmpprbout,cmpprbtmp,axis=0)


        totsdg = totsdg + nsdg

    ## Prepare output file
    qout = Dataset(flnm,'w') 

    dimprd = qout.createDimension('state_retrieved',nrtrv)
    dimxtr = qout.createDimension('state_true',nxprd)
    dimsdg = qout.createDimension('sounding',totsdg)
    dimchr = qout.createDimension('charnm',30)
    dimmix = qout.createDimension('mixture_component',maxcmp)
    if nsmp > 0:
        dimsmp = qout.createDimension('posterior_sample',nsmp)

    str_out = netCDF4.stringtochar(numpy.array(strvrs,'S30'))
    print(str_out)
    varnms = qout.createVariable('state_names_retrieved','S1',['state_retrieved','charnm'])
    varnms[:] = str_out

    varrgn = qout.createVariable('region_indicator','i2',['sounding'], fill_value = -99)
    varrgn[:] = rgout
    varrgn.long_name = 'NCA CONUS region number'
    varrgn.units = 'None'
    varrgn.missing_value = -99

    varxidx = qout.createVariable('airs_x_index','i2',['sounding'], fill_value = -99)
    varxidx[:] = lnidxout
    varxidx.long_name = 'AIRS cross-track index (0-based)'
    varxidx.units = 'None'
    varxidx.missing_value = -99

    varyidx = qout.createVariable('airs_y_index','i2',['sounding'], fill_value = -99)
    varyidx[:] = ltidxout
    varyidx.long_name = 'AIRS along-track index (0-based)'
    varyidx.units = 'None'
    varyidx.missing_value = -99

    vargrn = qout.createVariable('airs_granule','i2',['sounding'], fill_value = -99)
    vargrn[:] = grnout
    vargrn.long_name = 'AIRS granule number'
    vargrn.units = 'None'
    vargrn.missing_value = -99

    varqf = qout.createVariable('airs_tsurfair_qc','i2',['sounding'], fill_value = -99)
    varqf[:] = qfout
    varqf.long_name = 'AIRS near-surface temperature quality flag'
    varqf.units = 'None'
    varqf.missing_value = -99

    varlon = qout.createVariable('longitude','f4',['sounding'], fill_value = -9999)
    varlon[:] = lonout
    varlon.long_name = 'AIRS FOR center longitude'
    varlon.units = 'degrees_east'
    varlon.missing_value = -9999

    varlat = qout.createVariable('latitude','f4',['sounding'], fill_value = -9999)
    varlat[:] = latout
    varlat.long_name = 'AIRS FOR center latitude'
    varlat.units = 'degrees_north'
    varlat.missing_value = -9999

    vartm = qout.createVariable('time','f8',['sounding'], fill_value = -9999)
    vartm[:] = tmout
    vartm.long_name = 'AIRS observation time'
    vartm.units = 'Seconds since 1993-01-01'
    vartm.missing_value = -9999

    varpsf = qout.createVariable('surface_pressure','f4',['sounding'], fill_value = -9999)
    varpsf[:] = psfout
    varpsf.long_name = 'AIRS FOR surface pressure'
    varpsf.units = 'hPa'
    varpsf.missing_value = -9999

    varl2er = qout.createVariable('airs_tsurfair_err','f4',['sounding'], fill_value = -9999)
    varl2er[:] = l2errout
    varl2er.long_name = 'AIRS Level 2 near-surface temperature error estimate'
    varl2er.units = 'K'
    varl2er.missing_value = -9999

    varmn = qout.createVariable('pred_post_mean','f4',['sounding','state_true'], fill_value = -9999)
    varmn[:] = prdmnout
    varmn.long_name = 'Posterior mean for true state'
    varmn.units = ''
    varmn.missing_value = -9999

    if nsmp > 0:
        varskw = qout.createVariable('pred_post_skew','f4',['sounding','state_true'], fill_value = -9999)
        varskw[:] = skwout
        varskw.long_name = 'Posterior skewness for true state'
        varskw.units = ''
        varskw.missing_value = -9999

        varkur = qout.createVariable('pred_post_kurtosis','f4',['sounding','state_true'], fill_value = -9999)
        varkur[:] = kurout
        varkur.long_name = 'Posterior kurtosis for true state'
        varkur.units = ''
        varkur.missing_value = -9999

        varsmp = qout.createVariable('pred_post_samples','f4',['sounding','posterior_sample','state_true'], fill_value = -9999)
        varsmp[:] = smpout
        varsmp.long_name = 'Posterior samples for true state'
        varsmp.units = ''
        varsmp.missing_value = -9999

    varmn = qout.createVariable('pred_post_var','f4',['sounding','state_true','state_true'], fill_value = -9999)
    varmn[:] = sigxyout
    varmn.long_name = 'Posterior (co)variance for true state'
    varmn.units = ''
    varmn.missing_value = -9999

    varrtr = qout.createVariable('airs_ret_covariate','f4',['sounding','state_retrieved'], fill_value = -9999)
    varrtr[:] = rtryout
    varrtr.long_name = 'Retrieved covariates'
    varrtr.units = ''
    varrtr.missing_value = -9999

    varprb = qout.createVariable('pred_post_prob','f4',['sounding','mixture_component'], fill_value = -9999)
    varprb[:] = cmpprbout
    varprb.long_name = 'Mixture component posterior probabilities'
    varprb.units = 'None'
    varprb.missing_value = -9999

    qout.close()

    return


def airs_add_isd_val(flnm, vlnm, tmidx, tmday, rgnfrm, gmmdir):
    # Read validation matchups for AIRS outputs 
    # flnm:    Name of output file (NetCDF expected)
    # vlnm:    Location of validation data (CSV)
    # tms:     Time index in output
    # tmday:   Datetime object with time information
    # rgnfrm:  Data frame with region indicators
    # gmmdir:  Directory with GMM results

    flflt = numpy.array([-9999.], dtype=numpy.float32)
    flshr = numpy.array([-99], dtype=numpy.int16)

    # Read validation 
    #vlfrm = pandas.read_csv(vlnm, dtype = {'isd_temperature':float, 'N':int}, na_values = 'NA')
    vlfrm = pandas.read_csv(vlnm, na_values = 'NA')
    #print(vlfrm[0:10])
    #vlfrm['isd_temperature'] = df['isd_temperature'].astype(
    print(vlfrm.dtypes)

    isdtout = ma.array(vlfrm['isd_temperature'],dtype=numpy.float32)
    isdtout = ma.masked_invalid(isdtout)
    # Convert to K
    isdtout = isdtout + 273.15
    print(isdtout[0:10])
    isdtout = ma.filled(isdtout, fill_value = flflt) 
   
    vlctout = ma.array(vlfrm['N'],dtype=numpy.int16)
    vlctout = ma.masked_invalid(vlctout)
    vlctout = ma.filled(vlctout, fill_value = flshr)

    # Merge with existing file
    ncout = Dataset(flnm,'r+')

    dm1 = ncout.dimensions['sounding']

    if ('ISD' in ncout.groups):
        str1 = 'ISD group present'
        vlgrp = ncout.groups['ISD']
    else:
        vlgrp = ncout.createGroup('ISD')

    if ('ISD_temperature' in vlgrp.variables):
        vartisd = vlgrp.variables['ISD_temperature']
        vartisd[:] = isdtout
    else:
        vrtisd = vlgrp.createVariable('ISD_temperature','f4',['sounding'], fill_value = flflt[0])
        vrtisd[:] = isdtout
        vrtisd.long_name = 'Near-surface temperature at validation sites'
        vrtisd.units = 'K'
        vrtisd.missing_value = flflt[0]    

    if ('ISD_count' in vlgrp.variables):
        varnisd = vlgrp.variables['ISD_count']
        varnisd[:] = isdtout
    else:
        vrnisd = vlgrp.createVariable('ISD_count','i2',['sounding'], fill_value = flshr[0])
        vrnisd[:] = vlctout
        vrnisd.long_name = 'Number of validation observation sites'
        vrnisd.units = 'none'
        vrnisd.missing_value = flshr[0]    

    ncout.close()

    return

def qsummary(df, grpvr, vlvr):
    # Quantile summary for grouped data frame
    tmpdt = df[vlvr]
    dtvld = tmpdt[numpy.isfinite(tmpdt)]
    nmtch = dtvld.shape[0]
    plvs = numpy.array([10.0,25.0,50.0,75.0,90.0])
    dtqs = numpy.percentile(dtvld,q=plvs)
    dfout = pandas.DataFrame({'NSmp' : nmtch, 'Q10' : dtqs[0], 'Q25' : dtqs[1], 'Q50' : dtqs[2], \
                              'Q75' : dtqs[3], 'Q90' : dtqs[4]}, index=[0])
    return dfout


