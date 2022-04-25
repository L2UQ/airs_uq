import os
os.environ["OMP_NUM_THREADS"] = '8'
os.environ["OPENBLAS_NUM_THREADS"] = '8'
os.environ["MKL_NUM_THREADS"] = '8'
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset, date2num
from pickle import load
from glob import glob
from datetime import datetime
from scipy import stats
from numpy import random, ndarray, linalg

def read_AIRS_file(filename):
    f = Dataset(filename)
    '''
    airs_cc_radiances = f.variables['airs_cc_radiances'][:]
    airs_h2o = f.variables['airs_h2o'][:]
    airs_h2o_Err = f.variables['airs_h2o_Err'][:]
    airs_h2o_QC = f.variables['airs_h2o_QC'][:]
    airs_temperature = f.variables['airs_temperature'][:]
    airs_temperature_Err = f.variables['airs_temperature_Err'][:]
    airs_temperature_QC = f.variables['airs_temperature_QC'][:]
    cf = f.variables['cf'][:]
    plat = f.variables['plat'][:]
    plon = f.variables['plon'][:]
    salti = f.variables['salti'][:]
    sarta_h2o = f.variables['sarta_h2o'][:]
    sarta_landfrac = f.variables['sarta_landfrac'][:]
    sarta_radiances = f.variables['sarta_radiances'][:]
    sarta_temperature = f.variables['sarta_temperature'][:]
    spres = f.variables['spres'][:]
    stemp = f.variables['stemp'][:]
    '''
    var_names =[]
    var_values =[]
    var_info = []

    for name, info in f.variables.iteritems():
        var_names.append(name)
        var_value = f.variables[name][:]
        var_dim = var_value.shape[0]
        if name[0:6] == 'airs_h' or name[0:6] == 'airs_t':
            dummy_array = np.zeros([var_dim, 1])-9999.
            var_value = np.append(var_value, dummy_array, axis=1)
        var_values.append(var_value)
        var_info.append(info)
    return np.array(var_names), var_values, var_info

def calculate_es(temp):
    return 611*np.exp(17.67*(temp-273.15)/(temp-29.65))

def calculate_QV_and_VPD(h2o, ptemp, plev, palts):
    ### Calculate specific humidity (QV) and vapor pressure deficit (VPD)
    ###     h2o    - 2D array of molecular densities of water vapor (SARTA's default)
    ###     ptemp  - 2D array of temperatures (K)
    ###     plev   - 1D array of pressure levels (hPa)
    ###     palts  - 2D array of altitudes (m)
    # to be replaced later
    ###### Assumes 2D h2o and temperature
    nxy = h2o.shape[0]
    nz = h2o.shape[1]
    QV  = np.zeros((nxy, nz))-9999.
    RH  = np.zeros((nxy, nz))-9999.
    vpd  = np.zeros((nxy, nz))-9999.
    for ixy in np.arange(nxy):
        for iz in np.arange(1, nz):
        #for iz in np.arange(1, nz-3):
            if np.min([h2o[ixy, iz], plev[iz], ptemp[ixy, iz]]) != -9999.:
                # QV calculation, needs to account for layer thicknesses
                if iz > 0:
                    QV[ixy, iz] = h2o[ixy, iz]/(6.02214179*(10**21))/(28.97/18.00*plev[iz]/8.314/ptemp[ixy, iz]*(palts[ixy,iz-1] - palts[ixy,iz]))
                else:
                    # top layer thickness default 10^5
                    QV[ixy, iz] = 0.01*h2o[ixy, iz]/(6.02214179*(10**21))/(28.97/18.00*plev[iz]/8.314/ptemp[ixy, iz]*1.0e5)
                mmrs = 0.622*calculate_es(ptemp[ixy, iz])/(plev[iz]*100.)
                RH[ixy, iz] = QV[ixy, iz]/mmrs*100.
                vpd[ixy, iz] = calculate_es(ptemp[ixy, iz])*(100-RH[ixy, iz])/100./100.
        # Top level
        for iz in range(0,1):
            if np.min([h2o[ixy, iz], plev[iz], ptemp[ixy, iz]]) != -9999.:
                mmrs = 0.622*calculate_es(ptemp[ixy, iz])/(plev[iz]*100.)
                RH[ixy, iz] = 0.4 * RH[ixy, iz+1]
                QV[ixy, iz] = mmrs * RH[ixy, iz] / 100.
                vpd[ixy, iz] = calculate_es(ptemp[ixy, iz])*(100-RH[ixy, iz])/100./100.
    return QV, RH, vpd

def near_sfc_temp(ptemp, plev, sprs, passqual = False, qual = None):
    ### Calculate near-surface temperature from pressure levels and temperature profile
    ###     ptemp    - 2D array of temperatures (K)
    ###     plev     - 1D array of pressure levels (hPa)
    ###     sprs     - 1D array of surface pressure
    ###     passqual - Return near-surface quality information? Yes, if True
    ###     qual     - Optional quality flag variable
    ### Regress Temp on logP for lowest two levels
    ### Procedure is based on that defined in the AIRS Layers, Levels, and Trapezoids document

    nxy = ptemp.shape[0]
    nz = ptemp.shape[1]
    tsfc =  np.zeros((nxy),dtype=ptemp.dtype)-9999.
    if passqual:
        qualout = np.zeros((nxy),dtype=qual.dtype)
    psq = np.arange(nz)

    for ixy in np.arange(nxy):
        psb = psq[plev <= sprs[ixy]]
        pln = psb.shape[0]
        # Interpolate between levels if appropriate
        if pln == psq.shape[0]:
            pln2 = pln-2
            ps2 = psb[pln2:pln]
        else:
            # Check for AIRS 5 mb surface criteria
            pdf = np.absolute(sprs[ixy] - plev[pln-1])
            if pdf <= 5.0:
                pln2 = pln-2
                ps2 = psb[pln2:pln]
            else:
                # Interpolate case
                plnm1 = pln-1
                plnp1 = pln+1
                ps2 = psq[plnm1:plnp1]

        # Fit: interpolate in pressure
        slpout, itcptout, r2, pval, stderr = stats.linregress(plev[ps2],ptemp[ixy,ps2])
        # Predict
        tsfc[ixy] = itcptout + slpout * (sprs[ixy])
        if tsfc[ixy] < 0.0:
            tsfc[ixy] = -9999.0
        if passqual:
            qualout[ixy] = qual[ixy,pln-1]

    if passqual:
        return tsfc, qualout
    else:
        return tsfc

def near_sfc_qv_rh(qvin, sftemp, plev, sprs, passqual = False, qual = None):
    ### Calculate near-surface spec humidity and relative humidity from pressure levels, temperature profile, and specific humidity profile
    ###     qvin     - 2D array of temperatures (kg/kg)
    ###     sftemp   - 1D array of surface temperatures (K)
    ###     plev     - 1D array of pressure levels (hPa)
    ###     sprs     - 1D array of surface pressure
    ###     passqual - Return near-surface quality information? Yes, if True
    ###     qual     - Optional quality flag variable
    ### Use specific humidity from vertical level nearest surface

    nxy = qvin.shape[0]
    nz = qvin.shape[1]
    qvsfc = np.zeros((nxy),dtype=qvin.dtype)-9999.
    rhsfc = np.zeros((nxy),dtype=qvin.dtype)-9999.
    if passqual:
        qualout = np.zeros((nxy),dtype=qual.dtype)
    psq = np.arange(nz)

    for ixy in np.arange(nxy):
        psb = psq[plev <= sprs[ixy]]
        pln = psb.shape[0]
        pln1 = pln-1
        pslv = psb[pln1]
        # Extract
        qvsfc[ixy] = qvin[ixy,pslv]
        if sftemp[ixy] > 0.0:
            mmrs = 0.622*calculate_es(sftemp[ixy])/(sprs[ixy]*100.)
            rhsfc[ixy] = qvsfc[ixy]/mmrs
        if passqual:
            qualout[ixy] = qual[ixy,pslv]

    if passqual:
        return qvsfc, rhsfc, qualout
    else:
        return qvsfc, rhsfc

def calculate_h2odens(rh, ptemp, plev, palts):
    ### Calculate H2O molecular density (molecules * cm^-3) from relative humidity profiles
    ###     rh     - 2D array of relative humidity (unitless)
    ###     ptemp  - 2D array of temperatures (K)
    ###     plev   - 1D array of pressure levels (hPa)
    ###     palts  - 2D array of altitudes (m)
    # to be replaced later
    ###### Assumes 2D h2o and temperature
    nxy = rh.shape[0]
    nz = rh.shape[1]
    QV  = np.zeros((nxy, nz))-9999.
    h2odns  = np.zeros((nxy, nz))-9999.
    for ixy in np.arange(nxy):
        for iz in np.arange(0, nz):
            mmrs = 0.622*calculate_es(ptemp[ixy, iz])/(plev[iz]*100.)
            QV[ixy, iz] = rh[ixy, iz] * mmrs
            if np.min([rh[ixy, iz], plev[iz], ptemp[ixy, iz]]) != -9999.:
                # QV calculation, needs to account for layer thicknesses
                if iz > 0:
                    h2odns[ixy, iz] = QV[ixy,iz] * (6.02214179 * (10**21)) * (28.97/18.00*plev[iz]/8.314/ptemp[ixy, iz]*(palts[ixy,iz-1] - palts[ixy,iz]))
                else:
                    # top layer thickness default 10^4
                    h2odns[ixy, iz] = QV[ixy,iz] * (6.02214179 * (10**21)) * (28.97/18.00*plev[iz]/8.314/ptemp[ixy, iz]*1.0e4)
    return h2odns

def create_variable(f, variable_name, long_name, unit, variable_value):
    outvar = f.createVariable(variable_name, 'f8', ['nt', 'nxy', 'nz'], fill_value=-9999)
    outvar[:] = np.expand_dims(variable_value, axis=0)
    outvar.long_name = long_name
    outvar.units = unit
    outvar.missing_value = -9999.
    return


def write_AIRS_file(new_filename, var_names, var_values, var_info,
                    airs_QV, airs_RH, airs_VPD, sarta_QV, sarta_RH, sarta_VPD):
    f_out = Dataset(new_filename, 'w')
    nfreq = 2378
    nxy, nz = airs_QV.shape

    f_out.createDimension('nfreq', nfreq)
    dim2 = f_out.createDimension('nxy', nxy)
    dim3 = f_out.createDimension('nz', nz)
    dim1 = f_out.createDimension('nt', None)

    f_out.createVariable('time', 'f8', ['nt'])
    if new_filename[19] == 'A':
        hour = 21
    else:
        hour = 9
    f_out.variables['time'][:] = date2num(datetime(year=int(new_filename[14:18]), month=int(new_filename[19:21]), day=int(new_filename[22:24]),hour=hour), units='days since 2012-06-01')
    f_out.variables['time'].units = 'days since 2012-06-01'
    create_variable(f_out,  'airs_QV', 'specific_humidity', 'kg kg-1', airs_QV)
    create_variable(f_out,  'airs_RH', 'relative_humidity', '%', airs_RH)
    create_variable(f_out,  'airs_VPD', 'vapor_pressure_deficit', 'mb', airs_VPD)
    create_variable(f_out,  'sarta_QV', 'specific_humidity', 'kg kg-1', sarta_QV)
    create_variable(f_out,  'sarta_RH', 'relative_humidity', '%', sarta_RH)
    create_variable(f_out,  'sarta_VPD', 'vapor_pressure_deficit', 'mb', sarta_VPD)

    nvar = len(var_names)
    for ivar in np.arange(nvar):
        if var_names[ivar][-9:] == 'radiances':
            dim = ['nt', 'nxy', 'nfreq']
        elif var_names[ivar][4:6] == '_h' or var_names[ivar][4:6] == '_t' or var_names[ivar][5:7] == '_h' or var_names[ivar][5:7] == '_t':
            dim = ['nt', 'nxy', 'nz']
        else:
            dim = ['nt', 'nxy']

        outvar = f_out.createVariable(var_names[ivar], var_info[ivar].datatype, dim, fill_value=-9999.)
        outvar.setncatts({k: var_info[ivar].getncattr(k) for k in var_info[ivar].ncattrs()})
        outvar[:] = np.expand_dims(var_values[ivar], axis=0)
        outvar.missing_value = -9999.
    f_out.close()

def quantile_msgdat(vcdat, probs, msgval=-9999.):
    # Compute quantiles with missing data
    if (np.amax(probs) <= 1.0):
        prb100 = 100.0 * probs
    else:
        prb100 = probs

    dtsb = vcdat[vcdat != msgval]

    if dtsb.shape[0] > 0:
        qsout = np.percentile(dtsb,q=prb100)
    else:
        qsout = np.zeros(probs.shape) + msgval
    return qsout

def quantile_msgdat_discrete(vcdat, probs, msgval=-99):
    # Compute quantiles with missing data, discrete version
    if (np.amax(probs) <= 1.0):
        prb100 = 100.0 * probs
    else:
        prb100 = probs

    dtsb = vcdat[vcdat != msgval]

    if dtsb.shape[0] > 0:
        qsout = np.percentile(dtsb,q=prb100,interpolation='nearest')
    else:
        qsout = np.zeros(probs.shape) + msgval
    return qsout

def std_norm_quantile_from_obs(vcdat, obsqs, probs, msgval=-9999.):
    # Compute transform from observed quantiles to standard normal quantiles
    # Based on R function nrmrng from hydrology applications in vic_spec_fns.R

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    ndat = vcdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[vcdat != msgval]
    nvld = vsq.shape[0]
    zout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        ptst = np.append(0.0,np.append(probs,1.0))
        etst = np.append(-np.inf,np.append(obsqs,np.inf))
        qsq = np.arange(ptst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(vcdat[vsq],(ntst,1))
        etmt = np.transpose(np.tile(etst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (etmt < dtmt)
        hiind = (dtmt < etmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)
        #if smlw.shape[0] > 520:
        #    for j in range(505,510):
        #        str1 = 'Data[%d]: %.3f,  Lwind: %d, Hiind: %d' % (j,vcdat[vsq[j]],smlw[j],smhi[j])
        #        str2 = '  Quantiles: %.3f, %.3f' % (etst[smlw[j]],etst[smhi[j]])
        #        print(str1)
        #        print(str2)

        # Find probability spot
        prbdif = ptst[smhi] - ptst[smlw]
        pspt = ptst[smlw] + prbdif * random.uniform(size=nvld)
        #print(pspt[505:510])

        zout[vsq] = stats.norm.ppf(pspt)
    return zout


def data_quantile_from_std_norm(zdat, obsqs, probs, minval=-np.inf, maxval=np.inf, msgval=-9999.):
    # Inverse quantile transform: Transform from z-score back to data scale
    # Based on R function nrmrng from hydrology applications in vic_spec_fns.R

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    ndat = zdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[zdat != msgval]
    nvld = vsq.shape[0]
    qout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        # qtst, practical limits of z-score
        qtst = np.append(-99.0,np.append(qprb,99.0))
        etst = np.append(minval,np.append(obsqs,maxval))
        qsq = np.arange(qtst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(zdat[vsq],(ntst,1))
        qtmt = np.transpose(np.tile(qtst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (qtmt < dtmt)
        hiind = (dtmt < qtmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)

        #print('Sum Low')
        #print(smlw[0:10])
        #print('Sum High')
        #print(smhi[0:10])

        # Interpolate
        wtvc = (zdat[vsq] - qtst[smlw]) / (qtst[smhi] - qtst[smlw])
        qtmp = (1.0-wtvc) * etst[smlw] + wtvc * etst[smhi]

        qout[vsq] = qtmp[:]
    return qout

def data_quantile_from_std_norm_discrete(zdat, obsqs, probs, minval=-np.inf, maxval=np.inf, msgval=-99):
    # Inverse quantile transform: Transform from z-score back to data scale, discrete case
    # Based on R function nrmrng from hydrology applications in vic_spec_fns.R

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    ndat = zdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[zdat != msgval]
    nvld = vsq.shape[0]
    qout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        # qtst, practical limits of z-score
        qtst = np.append(-99.0,np.append(qprb,99.0))
        etst = np.append(minval,np.append(obsqs,maxval))
        qsq = np.arange(qtst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(zdat[vsq],(ntst,1))
        qtmt = np.transpose(np.tile(qtst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (qtmt < dtmt)
        hiind = (dtmt < qtmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)

        #print('Sum Low')
        #print(smlw[0:10])
        #print('Sum High')
        #print(smhi[0:10])

        # Assign at random
        wtvc = (zdat[vsq] - qtst[smlw]) / (qtst[smhi] - qtst[smlw])
        u1 = random.uniform(size=nvld)

        #vsqlw = vsq[wtvc < u1]
        #vsqhi = vsq[wtvc > u1]
        qtmplw = etst[smlw]
        qtmphi = etst[smhi]
        qtmp = qtmplw
        qtmp[wtvc > u1] = qtmphi[wtvc > u1]
        qout[vsq] = qtmp[:]

    return qout

def lgtzs(props):
    # Transform a collection of K proportions into K-1 logit components (log odds)
    # Order is relevant and retained

    nprp = props.shape[0]
    nlgt = nprp - 1
    cprb = np.cumsum(props[0:nlgt])
    zlgt = np.log(props[0:nlgt] / (1.0-cprb) )

    return zlgt

def lgttoprp(lgts):
    # Transform a collection of K-1 logit transforms to K proportions that sum to 1
    nlgt = lgts.shape[0]
    nprp = nlgt + 1
    prbout = np.zeros((nprp,))
    cprp = 0.0
    for j in range(nlgt):
        prbout[j] = (cprp + np.exp(lgts[j])) / (1.0 + np.exp(lgts[j])) - cprp
        cprp = cprp + prbout[j]
    prbout[nprp-1] = 1.0 - cprp
    return prbout

def std_norm_quantile_from_obs_fill_msg(vcdat, obsqs, probs, msgval=-9999.):
    # Compute transform from observed quantiles to standard normal quantiles
    # Fill missing values with a random standard normal value

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    ndat = vcdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[vcdat != msgval]
    msq = dsq[vcdat == msgval]
    nvld = vsq.shape[0]
    nmsg = msq.shape[0]
    zout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        ptst = np.append(0.0,np.append(probs,1.0))
        etst = np.append(-np.inf,np.append(obsqs,np.inf))
        qsq = np.arange(ptst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(vcdat[vsq],(ntst,1))
        etmt = np.transpose(np.tile(etst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (etmt < dtmt)
        hiind = (dtmt < etmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)
        #print(smhi[505:510])
        #if smlw.shape[0] > 520:
        #    for j in range(505,510):
        #        str1 = 'Data[%d]: %.3f,  Lwind: %d, Hiind: %d' % (j,vcdat[vsq[j]],smlw[j],smhi[j])
        #        str2 = '  Quantiles: %.3f, %.3f' % (etst[smlw[j]],etst[smhi[j]])
        #        print(str1)
        #        print(str2)

        # Find probability spot
        prbdif = ptst[smhi] - ptst[smlw]
        pspt = ptst[smlw] + prbdif * random.uniform(size=nvld)
        #print(pspt[505:510])

        zout[vsq] = stats.norm.ppf(pspt)

        # Missing
        psptm = random.uniform(size=nmsg)
        zout[msq] = stats.norm.ppf(psptm)
    return zout

def cov2cor(cvmt):
    d = 1.0 / np.sqrt(cvmt.diagonal())
    d1 = np.diag(d)
    t1 = np.dot(d1,cvmt)
    crmt = np.dot(t1,d1)
    return crmt

def unpackcov(pckmat,nelm):
    # Unpack a vectorized lower-triangle of a covariance matrix
    x0 = 1 + np.zeros((nelm,nelm))
    xpck = np.triu(x0)
    x2 = ndarray.flatten(xpck)
    x2[x2 == 1.0] = pckmat
    x2.shape = (nelm,nelm)
    diagsv = np.diagonal(x2)
    x2l = np.tril(np.transpose(x2),-1)
    xout = x2l + x2
    return xout

def clean_byte_list(btlst):
    clean = [x for x in btlst if x != None]
    strout = b''.join(clean).decode('utf-8')
    return strout

def std_norm_limits_from_obs_fill_msg(vcdat, obsqs, probs, msgval=-9999.):
    # Obtain upper and lower z-score limits based on empirical quantiles
    # Fill missing values with a random standard normal value

    nprb = probs.shape[0]
    qprb = stats.norm.ppf(probs)
    qmin = qprb[0] * 1.5
    qmax = qprb[nprb-1] * 1.5

    mct = ma.count_masked(vcdat)
    if mct > 0:
        vcdat = vcdat.filled(fill_value=msgval)
    ndat = vcdat.shape[0]
    dsq = np.arange(ndat)
    vsq = dsq[vcdat != msgval]
    msq = dsq[vcdat == msgval]
    nvld = vsq.shape[0]
    nmsg = msq.shape[0]
    lwrzout = np.zeros((ndat,)) + msgval
    uprzout = np.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        #print("All observations missing, no transformation performed")
        lwrzout[msq] = qmin
        uprzout[msq] = qmax
    else:
        ptst = np.append(0.0,np.append(probs,1.0))
        etst = np.append(-np.inf,np.append(obsqs,np.inf))
        qsq = np.arange(ptst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = np.tile(vcdat[vsq],(ntst,1))
        etmt = np.transpose(np.tile(etst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (etmt < dtmt)
        hiind = (dtmt < etmt)

        smlw = np.sum(lwind,axis=0) - 1
        smhi = ntst - np.sum(hiind,axis=0)

        # Find probability spot
        lwrzout[vsq] = stats.norm.ppf(ptst[smlw])
        uprzout[vsq] = stats.norm.ppf(ptst[smhi])

        # Missing
        lwrzout[msq] = qmin
        uprzout[msq] = qmax

        lwrzout[lwrzout < qmin] = qmin
        uprzout[uprzout > qmax] = qmax

    return lwrzout, uprzout

def trnc_norm_mcmc(datarr, mnvec, prcmat, lwrlim, uprlim, niter, nburn, nvec, nstate):
    # Perform MCMC sampling of full information states with truncated normal updates
    # datarr:   Data array (nvec x nstate)
    # mnvec:    Mean vector (nstate)
    # prcmat:   Precsion matrix, inverse covariance (nstate x nstate)
    # lwrlim:   Truncated normal, lower limits (nvec x nstate)
    # uprlim:   Truncated normal, upper limits (nvec x nstate)
    # niter:    Number of MCMC iterations
    # nburn:    Number of burn in iterations
    # nvec:     Data Sample size
    # nstate:   Multivariate dimension

    murp = np.tile(mnvec, (nvec,1))
    datcur = np.zeros(datarr.shape, dtype=datarr.dtype)
    datcur[:,:] = datarr[:,:]
    datmn = np.zeros(datarr.shape, dtype=datarr.dtype)

    cndvrs = 1.0 / (np.diagonal(prcmat))
    cndsds = np.sqrt(cndvrs)

    mctr = 0
    for t in range(niter):
        istr = 'MCMC Iteration %d' % (t)
        if ( ( t % 10) == 0):
            print(istr)
        # Loop through state elements
        for q in range(nstate):
            zdev = datcur - murp
            #print(cndvrs[q])
            cijvec = (-1.0) * cndvrs[q] * prcmat[:,q]
            cijvec[q] = 0.0
            mndv = np.dot(zdev,cijvec)
            cmn = murp[:,q] + mndv

            # Standardize
            stdlw = (lwrlim[:,q] - cmn) / cndsds[q]
            stdhi = (uprlim[:,q] - cmn) / cndsds[q]
            plwr = stats.norm.cdf(stdlw)
            pupr = stats.norm.cdf(stdhi)

            #str1 = 'plwr extremes: %.3e, %.3e' % (np.amin(plwr),np.amax(plwr))
            #str2 = 'pupr extremes: %.3e, %.3e' % (np.amin(pupr),np.amax(pupr))
            #print(str1)
            #print(str2)
            # Sample in prob space
            prbdif = pupr - plwr
            #str2 = '%d prbdif extremes: %.3e, %.3e' % (q,np.amin(prbdif),np.amax(prbdif))
            #print(str2)
            asq = np.arange(nvec)

            asb = asq[prbdif < 0]
            if asb.shape[0] > 0:
                str2 = '%d prbdif < 0' % (q)
                print(str2)
                print(asb)

            pspt = plwr + prbdif * random.uniform(size=nvec)
            # Adjust to limit extremes
            pspt[pspt < 10e-8] = 1.0e-8
            pspt[pspt > (1.0 - 1.0e-8)] = 1.0 - 1.0e-8

            # Transform back
            stdspt = stats.norm.ppf(pspt)
            znew = cndsds[q] * stdspt + cmn
            datcur[:,q] = znew

        # Update mean
        if t >= nburn:
            mctr = mctr + 1
            #print(np.amin(datcur))
            #print(np.amax(datcur))
            datmn = datmn + (datcur - datmn) / mctr
            #print(np.amin(datmn))
            #print(np.amax(datmn))

    return datcur, datmn

def cloud_frac_summary(cfrcarr):
    # Compute summary of 3x3 cloud fraction array
    # cfrcarr:  Cloud fraction array (3x3) 

    # returns array of summary variables

    cldflt = cfrcarr.flatten()
    cldmn = np.mean(cldflt)
    cldsd = np.std(cldflt)
    clrvc = cldflt[cldflt == 0.0]
    ovcvc = cldflt[cldflt == 1.0]

    cldout = np.array([cldmn,cldsd,clrvc.shape[0],ovcvc.shape[0]])
    return cldout

def sfclvl(psfc, levarr): 
    # Return array with surface level indicator: the lowest vertical level above the surface pressure
    # Assume psfc is 2D (lat, lon)

    nlt = psfc.shape[0]
    nln = psfc.shape[1]

    nz = levarr.shape[0]
    psq = np.arange(nz)
    slvs = np.zeros((nlt,nln),dtype=np.int16)

    for j in range(nlt):
        for i in range(nln):
            psb = psq[levarr <= psfc[j,i]]
            if psb.shape[0] == 0:
                str1 = 'Sfc pos ?: %d, %d, %.4f' % (j,i,psfc[j,i])
                print(str1)
                slvs[j,i] = -99
            else:
                slvs[j,i] = psb[-1]
    return slvs

