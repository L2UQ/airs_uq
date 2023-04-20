import pandas
from netCDF4 import Dataset
import numpy
import numpy.ma as ma
from numpy import random, ndarray, linalg
from scipy import stats

def qsummary(df,grpvr,vlvr):
    # Summarize with quantiles
    tmpdt = df[vlvr]
    dtvld = tmpdt[numpy.isfinite(tmpdt)]
    nmtch = dtvld.shape[0] 
    dtmn = numpy.mean(dtvld)
    dtvr = numpy.var(dtvld)
    dfout = pandas.DataFrame({'NSmp' : nmtch, \
                              'Mean' : dtmn, 'Variance' : dtvr}, \
                               index=[0])
    return dfout   


def clean_byte_list(btlst):
    clean = [x for x in btlst if x != None]
    strout = b''.join(clean).decode('utf-8')
    return strout

def cloud_frac_summary(cfrcarr):
    # Compute summary of 3x3 cloud fraction array
    # cfrcarr:  Cloud fraction array (3x3) 

    # returns array of summary variables

    cldflt = cfrcarr.flatten()
    cldmn = numpy.mean(cldflt)
    cldsd = numpy.std(cldflt)
    clrvc = cldflt[cldflt == 0.0]
    ovcvc = cldflt[cldflt == 1.0]

    cldout = numpy.array([cldmn,cldsd,clrvc.shape[0],ovcvc.shape[0]])
    return cldout

def sfclvl(psfc, levarr): 
    # Return array with surface level indicator: the lowest vertical level above the surface pressure
    # Assume psfc is 2D (lat, lon)

    nz = levarr.shape[0]
    psq = numpy.arange(nz)
    slvs = numpy.zeros((1,),dtype=numpy.int16)

    psb = psq[levarr <= psfc]
    if psb.shape[0] == 0:
        str1 = 'Sfc pos ?: %d, %d, %.4f' % (j,i,psfc[j,i])
        print(str1)
        slvs[0] = -99
    else:
        slvs[0] = psb[-1]

    return slvs

def airs_nsat_sdg_gmm(gmmfl, supfl, atrk, xtrk):
    # Read in AIRS Level 2 products and 
    # produce conditional GMM parameters for
    # near-surface air temperature

    print(gmmfl)
    ncgm = Dataset(gmmfl)
    rtnms = ncgm.variables['state_names_retrieved'][:]
    lvs = ncgm.variables['level'][:]
    ncgm.close()
    nmclps = rtnms.tolist()
    strvrs = list(map(clean_byte_list,nmclps))

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
    nlev = lvs.shape[0]

    maxcmp = 0

    ncgm = Dataset(gmmfl)
    gmm_prp = ncgm.variables['mixture_proportion'][:]
    ncgm.close()

    nmxcmp = gmm_prp.shape[0]
    if nmxcmp > maxcmp:
        maxcmp = nmxcmp

    if tairpc > 0:
        ncgm = Dataset(gmmfl)
        tairmnvc = ncgm.variables['temp_prof_mean'][:]
        taireig = ncgm.variables['temp_eigenvector'][0:tairpc,:]
        ncgm.close()

        # Check temp prof
        tmpmx = numpy.nanmax(tairmnvc)
        for q1 in range(tairmnvc.shape[0]):
            if numpy.isnan(tairmnvc[q1]):
                tairmnvc[q1] = tmpmx

    # Level 2 processing
    # 1. Cloud summaries
    # 2. PCA of vertical profiles 
    ncl2 = Dataset(supfl)
    cfrcair = ncl2.variables['CldFrcStd'][atrk,xtrk,:,:,:]
    cfrcaqc = ncl2.variables['CldFrcStd_QC'][atrk,xtrk,:,:,:]
    tsfcqc = ncl2.variables['TSurfAir_QC'][atrk,xtrk]
    tsfair = ncl2.variables['TSurfAir'][atrk,xtrk]
    tsferr = ncl2.variables['TSurfAirErr'][atrk,xtrk]
    psfc = ncl2.variables['PSurfStd'][atrk,xtrk]
    tairsp = ncl2.variables['TAirSup'][atrk,xtrk,:]
    ncldair = ncl2.variables['nCld'][atrk,xtrk,:,:]
    ncl2.close()
      
    frctot = cfrcair[:,:,0] + cfrcair[:,:,1]
    #cldsmarr = numpy.zeros((nairtrk,nairxtk,4),frctot.dtype)
    #ncldmx = numpy.zeros((nairtrk,nairxtk),ncldair.dtype)
    cldsmarr = cloud_frac_summary(frctot)
    ncldmx = numpy.amax(ncldair)


    # Sfc info
    sfcspt = sfclvl(psfc,lvs)
    sfcspt = sfcspt + lsqair[0]

    tdftmp = tairsp[lv850air] - tsfair
    if sfcspt <= lv850air:
        tdftmp = tairsp[sfcspt[0]-2] - tsfair[0]

    l2frm = pandas.DataFrame({'L2LonIdx': xtrk, 'L2LatIdx': atrk}, index = [0])

    l2frm['NSTRtrv'] = tsfair
    l2frm['TDif850'] = tdftmp
    l2frm['NCloud'] = ncldmx
    l2frm['PSfc'] = psfc
    l2frm['CFrcMean'] = cldsmarr[0]
    l2frm['CFrcSD'] = cldsmarr[1]
    l2frm['NClr'] = cldsmarr[2]
    l2frm['NOvc'] = cldsmarr[3]
    tpcnms = []
    for t in range(tairpc):
        pcnm = 'TempPC%d' % (t+1)
        tpcnms.append(pcnm)
        l2frm[pcnm] = 0.0 

    # PCA processing
    nlv = lsqair.shape[0]
    lsq = numpy.arange(nlv)

    tprftmp = tairsp[lsqair] 
    tprftmp = ma.masked_where(tprftmp < 0,tprftmp)
    msq = ma.is_masked(tprftmp)
    tprfscr = ma.filled(tprftmp, fill_value=tairmnvc[:]) 

    tpcsr = numpy.dot(taireig[:,:],tprfscr)
    print(tpcsr.shape)
    for t in range(tairpc):
        pcnm = 'TempPC%d' % (t+1)
        l2frm[pcnm].values[:] = tpcsr[t]

    # GMM calculations
    ncgm = Dataset(gmmfl)
    gmm_prp = ncgm.variables['mixture_proportion'][:]
    gmm_mux = ncgm.variables['mean_true'][:,:]
    gmm_muy = ncgm.variables['mean_retrieved'][:,:]
    gmm_varx = ncgm.variables['varcov_true'][:,:,:]
    gmm_varxy = ncgm.variables['varcov_cross'][:,:,:]
    gmm_vary = ncgm.variables['varcov_retrieved'][:,:,:]
    gmm_prcy = ncgm.variables['precmat_retrieved'][:,:,:]
    gmm_pstvarx = ncgm.variables['varcov_post_true'][:,:,:]
    ncgm.close()


    nmxcmp = gmm_prp.shape[0]
    nrtrv = gmm_muy.shape[1]
    nrtbs = nrtrv - tairpc
    nxprd = gmm_mux.shape[1]

    # Set up a data array
    ydattmp = numpy.zeros((nrtrv,),dtype=numpy.float64)
    for q in range(nrtbs):
        ydattmp[q] = l2frm[strvrs[q]]
    for q in range(tairpc):
        ydattmp[q+nrtbs] = l2frm[tpcnms[q]] 
    print(ydattmp) 
     
    ## Apply GMM, from gmm_post_pred in airs_post_expt_support.R 
    # Densities
    f_y_c = numpy.zeros((nmxcmp,),dtype=numpy.float64)
    print('Computing f_y_c')
    for k in range(nmxcmp):
        w, v = linalg.eig(gmm_vary[k,:,:])
        wsq = numpy.arange(w.shape[0])
        wsb = wsq[w < 5.0e-5]
        if wsb.shape[0] > 0:
            s1 = 'Lifting %d eigenvalues' % (wsb.shape[0])
            #print(s1)
            w[wsb] = 5.0e-5
            wdg = numpy.diagflat(w)
            gmm_vary[k,:,:] = numpy.dot(v, numpy.dot(wdg,v.T))
        w, v = linalg.eig(gmm_vary[k,:,:])
        #print(numpy.amin(w))
        if nrtrv > 1:
            f_y_c[k] = stats.multivariate_normal.logpdf(ydattmp, mean=gmm_muy[k,:], cov=gmm_vary[k,:,:])
        elif ntrv == 1:
            # Univariate density
            ltr = 0
    # Adjust for possible underflow
    mxdns = numpy.amax(f_y_c)
    mxarr = numpy.transpose(numpy.tile(mxdns,reps=(nmxcmp,1)))
    adjdns = f_y_c - mxarr
    print('Adjdens shape')
    print(adjdns.shape)

    # Compute the conditional probabilities, p_c_y
    print('computing p_c_y')
    prprep = numpy.tile(gmm_prp,reps=(1))
    cmplk = prprep * numpy.exp(adjdns)
    sumlk = numpy.sum(cmplk)
    sumrep = numpy.transpose(numpy.tile(sumlk,reps=(1))) 
    cmpprb = cmplk / sumrep
    print(cmpprb)
    print(cmpprb.shape)

    print('predicting E_X_Y')
    ex_y_c = numpy.zeros((nxprd,nmxcmp),dtype=numpy.float64)
    ex_y = numpy.zeros((nxprd),dtype=numpy.float64)
    for k in range(nmxcmp):
        muxrp = numpy.tile(gmm_mux[k,:],reps=(1))
        muyrp = numpy.tile(gmm_muy[k,:],reps=(1))
        ydevcr = ydattmp - muyrp
        prcdev = numpy.dot(gmm_prcy[k,:,:], numpy.transpose(ydevcr))
        cvxytmp = numpy.transpose(gmm_varxy[k,:,:])
        ex_y_c[:,k] = muxrp + numpy.transpose(numpy.dot(cvxytmp,prcdev))
    print(prcdev.shape) 
    print(muxrp.shape)
    print(muyrp.shape)            
    for k in range(nxprd):
        cmpmns = cmpprb * ex_y_c[k,:]
        print(cmpmns.shape)
        ex_y[k] = numpy.sum(cmpmns,axis=1)
    print(ex_y) 

    print('predicting Sigma_X_Y')
    Sigma_X_Y_C_bet = numpy.zeros((nxprd,nxprd,nmxcmp),dtype=numpy.float64)
    Sigma_X_Y_C_wth = numpy.zeros((nxprd,nxprd,nmxcmp),dtype=numpy.float64)
    Sigma_X_Y = numpy.zeros((nxprd,nxprd),dtype=numpy.float64)
    for k in range(nmxcmp):
        wthcv = gmm_pstvarx[k,:,:]
        mndv = ex_y_c[:,k] - ex_y
        prbrp = numpy.repeat(cmpprb[:,k],nxprd*nxprd)
        #print(cmpprb[:,k])
        #print(prbrp.shape)
        prbrp = numpy.reshape(prbrp,(nxprd,nxprd))
        wthrp = numpy.tile(wthcv.flatten(),1)
        wthrp = numpy.reshape(wthrp,(nxprd,nxprd))
        Sigma_X_Y_C_wth[:,:,k] = wthrp
        Sigma_X_Y_C_bet[:,:,k] = numpy.outer(mndv,mndv)
        Sigma_X_Y = Sigma_X_Y + prbrp * (Sigma_X_Y_C_wth[:,:,k] + Sigma_X_Y_C_bet[:,:,k])
#        print(prbrp.shape)
#        print(wthrp.shape)

    # Output data frame
    mxcmps = numpy.arange(1,nmxcmp+1)
    gmmfrm = pandas.DataFrame({'Component': mxcmps, 'CondProb': cmpprb[0,:], 'CompCondMean': ex_y_c[0,:], 'CompCondVar': Sigma_X_Y_C_wth[0,0,:]})

    cmbfrm = pandas.DataFrame({'Component': numpy.array([-1], dtype=mxcmps.dtype), 'CondProb': numpy.array([1.0], dtype=cmpprb.dtype), \
                               'CompCondMean': ex_y[0], 'CompCondVar': Sigma_X_Y[0,0] })
    print(cmbfrm)

    return gmmfrm

