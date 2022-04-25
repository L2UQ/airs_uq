# Quantile utilities for processing MERRA/AIRS data 

import matplotlib as mpl
mpl.use('agg')
import pandas
import pylab
from matplotlib import pyplot, colors
import matplotlib.ticker as mticker
import numpy
from scipy import stats
#from statsmodels.graphics.mosaicplot import mosaic
import numpy.ma as ma
from netCDF4 import Dataset
from numpy import random, linalg
import datetime
import os, sys
import calculate_VPD
import properscoring

def uq_airs_post_dens(outfile, modfile, sdgidx, fhdr, fgwdth=8, fghght=6):
    # Validation, posterior density figure
    # outfile:   Output file with figures 
    # modfile:   UQ model results file
    # cnffile:   State vector configuration file
    # fhdr:      Figure header

    fin = Dataset(modfile,'r')
    #latl2 = fin.variables['latitude'][:]
    #lonl2 = fin.variables['longitude'][:]
    nstl2 = fin.variables['airs_ret_covariate'][sdgidx,0]
    pstmn = fin.variables['pred_post_mean'][sdgidx,0]
    pstvr = fin.variables['pred_post_var'][sdgidx,0,0]
    nstqc = fin.variables['airs_tsurfair_qc'][sdgidx]
    rgnid = fin.variables['region_indicator'][sdgidx]
    l2err = fin.variables['airs_tsurfair_err'][sdgidx]
    uqsmp = fin.variables['pred_post_samples'][sdgidx,:,0]       

    tmpsq = numpy.arange(260,300.5,0.5)

    if ('ISD' in fin.groups):
        vlgrp = fin.groups['ISD']
        if ('ISD_temperature' in vlgrp.variables):
            tvlisd = vlgrp.variables['ISD_temperature'][sdgidx]

            nsmp = uqsmp.shape[0]
            ssq = numpy.arange(nsmp)

            airsz = (tvlisd - nstl2) / l2err
            pitairs = stats.norm.cdf(airsz)
            crpsairs = properscoring.crps_gaussian(tvlisd, mu=nstl2, sig=l2err) 

            ssb = ssq[ uqsmp <= tvlisd]
            pituq = (ssb.shape[0] + 0.5) / (nsmp + 1.0)
            crpsuq = properscoring.crps_ensemble(tvlisd, uqsmp[:])

    fin.close()

    airsdns = stats.norm.pdf(tmpsq,loc=nstl2,scale=l2err)
    uqkde = stats.gaussian_kde(uqsmp)
    uqdns = uqkde.evaluate(tmpsq)

    c3 = ["#0066FF","#B2387E","#009933"]

    mxdns = numpy.amax(numpy.append(airsdns,uqdns)) * 0.95
    lblst = ['','','','']
    lblst[0] = 'AIRS PIT: %.3f' % (pitairs)
    lblst[1] = 'AIRS CRPS: %.2f K' % (crpsairs)
    lblst[2] = 'UQ PIT: %.3f' % (pituq)
    lblst[3] = 'UQ CRPS: %.2f K' % (crpsuq)
    lbclr = [c3[0],c3[0],c3[1],c3[1]]

    fig = pyplot.figure(figsize=(fgwdth,fghght))

    nmlst = ['AIRS','UQ']

    p1 = pyplot.subplot(1,1,1)

    pl1, = p1.plot(tmpsq,airsdns,'-',color=c3[0])
    pl2, = p1.plot(tmpsq,uqdns,'-',color=c3[1])
    plst = [pl1, pl2]
    p1.plot([tvlisd,tvlisd],[0.0,0.01],'-',color=c3[2],linewidth=2)
    p1.set_xlim(260, 300)
    #p1.set_ylim(-5,101.0)
    p1.xaxis.grid(color='#777777',linestyle='dotted')
    p1.yaxis.grid(color='#777777',linestyle='dotted')
    #p1.set_xlabel(nmlst[0],size=14)
    #p1.set_ylim(-0.2,12)
    p1.set_xlabel('Temperature [K]',size=11)
    p1.set_ylabel('Density',size=11)
    #p1.set_yticks(lspts)
    #p1.set_yticklabels(pspts)
    for lb in p1.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p1.yaxis.get_ticklabels():
        lb.set_fontsize(10)
    # Additional labeling
    if nstl2 < 280:
        xlbspt = 290
    else:
        xlbspt = 265
    for p in range(len( lblst)):
        pyplot.text(xlbspt,mxdns-0.006*p,lblst[p], \
                    horizontalalignment='left', \
                    color=lbclr[p],fontsize=10)    
         
    pyplot.title(fhdr,size=12)
    fig.subplots_adjust(bottom=0.1,top=0.92,left=0.12,right=0.92, \
                        hspace=0.15,wspace=0.15)
    fig.savefig(outfile)
    pyplot.close()

    return
   



def quantile_summary_figs(outfile, qfile, cnffile, fhdr, fgwdth=15, fghght=7):
    # Summary figures of marginal distributions 
    # outfile:   Output file with figures 
    # qfile:     Quantile file
    # cnffile:   State vector configuration file
    # fhdr:      Figure header

    df = pandas.read_csv(cnffile, dtype = {'Order':int, 'Group':str, 'ZScore_Name':str, 'Quantile_Name':str, \
                                           'Data_Name':str, 'Start':int, 'Length':int, 'DType':str })
    tsz = df['Length'].sum()
    szstr = '%d Total State Vector Elements' % (tsz)
    print(szstr)

    nrw = df.shape[0]

    # Loop through groups to initialize
    #for q in range(nrw):
    #    qvrnm = df['Quantile_Name'].values[q]
    #    fqs = Dataset(qfile,'r')
    #    if (df['Group'].values[q] == 'CloudFrac'):
    #        qtmp = fqs.variables[qvrnm][:,:,:]
    #    elif (df['Length'].values[q] > 1):
    #        qtmp = fqs.variables[qvrnm][:,:]
    #    else:
    #        qtmp = fqs.variables[qvrnm][:]
    #    fqs.close()

    # Probs and levels
    fqs = Dataset(qfile,'r')
    lvs = fqs['level'][:]
    prbs = fqs['probability'][:]
    fqs.close()

    prbsq = numpy.arange(prbs.shape[0])

    q50spt = prbsq[prbs == 0.5]
    q05spt = prbsq[prbs == 0.05]
    q95spt = prbsq[prbs == 0.95]

    pspts = numpy.array([1000,500,200,100,50,20,10,1,0.5,0.1,0.05,0.01])
    lspts = 7.0 - numpy.log(pspts)
    lvrv = 7.0 - numpy.log(lvs) 

    ## Work through state vector groups
    c3 = ["#0066FF","#B2387E","#39BEB1","#ACA4E2"]
    fig = pyplot.figure(figsize=(fgwdth,fghght))

    #  
    #nmlst = ['CFrac1','CNGWat1']
    #xsb = dffl.loc[(dffl.CFrac1 >= 0.0),nmlst[0]]
    #ysb = dffl.loc[(dffl.CFrac1 >= 0.0),nmlst[1]]

    tpvrnm = df.loc[(df.Group == 'Temperature'),'Quantile_Name'].values[0]
    sfvrnm = df.loc[(df.Group == 'Surface'),'Quantile_Name'].values[0]
    fqs = Dataset(qfile,'r')
    qtmp = fqs.variables[tpvrnm][:,:]
    qsfc = fqs.variables[sfvrnm][:]
    fqs.close()

    p1 = pyplot.subplot(2,4,1)

    p1.plot(qtmp[:,q50spt],lvrv,'-',c=c3[0])
    p1.plot(qtmp[:,q05spt],lvrv,'--',c=c3[0])
    p1.plot(qtmp[:,q95spt],lvrv,'--',c=c3[0])
    #p1.set_xlim(-0.1,1.1)
    #p1.set_ylim(-5,101.0)
    p1.xaxis.grid(color='#777777',linestyle='dotted')
    p1.yaxis.grid(color='#777777',linestyle='dotted')
    #p1.set_xlabel(nmlst[0],size=14)
    p1.set_ylim(-0.2,12)
    p1.set_xlabel('Temperature',size=11)
    p1.set_ylabel('Pressure',size=11)
    p1.set_yticks(lspts)
    p1.set_yticklabels(pspts)
    for lb in p1.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p1.yaxis.get_ticklabels():
        lb.set_fontsize(10)
         
    pyplot.title('Temperature Quantiles',size=12)

    # RH
    rhvrnm = df.loc[(df.Group == 'RelHum'),'Quantile_Name'].values[0]
    fqs = Dataset(qfile,'r')
    qrh = fqs.variables[rhvrnm][:,:]
    fqs.close()

    p2 = pyplot.subplot(2,4,5)

    p2.plot(qrh[:,q50spt],lvrv,'-',c=c3[1])
    p2.plot(qrh[:,q05spt],lvrv,'--',c=c3[1])
    p2.plot(qrh[:,q95spt],lvrv,'--',c=c3[1])
    #p1.set_xlim(-0.1,1.1)
    p2.set_ylim(-0.2,6)
    p2.set_xlabel('RH',size=11)
    p2.set_ylabel('Pressure',size=11)
    p2.xaxis.grid(color='#777777',linestyle='dotted')
    p2.yaxis.grid(color='#777777',linestyle='dotted')
    #p1.set_xlabel(nmlst[0],size=14)
    #p1.set_ylabel(nmlst[1],size=14)
    p2.set_yticks(lspts[lspts < 6])
    p2.set_yticklabels(pspts[lspts < 6])
    for lb in p2.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p2.yaxis.get_ticklabels():
        lb.set_fontsize(10)
         
    pyplot.title('RH Quantiles',size=12)

    # Cloud Slabs
    nslbvrnm = df.loc[(df.Group == 'NumCloud'),'Quantile_Name'].values[0]
    fcld = Dataset(qfile,'r')
    qnslb = fqs.variables[nslbvrnm][:]
    fcld.close()

    nslbvl = numpy.array([0,1,2],dtype=numpy.int16)
    pnslb = numpy.zeros((3,),dtype=numpy.float32)
    psb0 = prbs[qnslb == nslbvl[0]]
    pnslb[0] = psb0[psb0.shape[0]-1]
    psb2 = prbs[qnslb == nslbvl[2]]
    pnslb[2] = 1.0 -  psb2[0]
    pnslb[1] = 1.0 - pnslb[0] - pnslb[2]
    print(pnslb)

    ct1vrnm = df.loc[(df.Group == 'CloudType'),'Quantile_Name'].values[0]
    ct2vrnm = df.loc[(df.Group == 'CloudType'),'Quantile_Name'].values[1]
    print(ct1vrnm)
    print(ct2vrnm)
    fcld = Dataset(qfile,'r')
    qctp1 = fqs.variables[ct1vrnm][:]
    qctp2 = fqs.variables[ct2vrnm][:]
    fcld.close()

    ctpvl = numpy.array([0,1],dtype=numpy.int16)
    pctyp1 = numpy.zeros((2,),dtype=numpy.float32)
    pctyp2 = numpy.zeros((2,),dtype=numpy.float32)
    psb1 = prbs[qctp1 == ctpvl[0]]
    pctyp1[0] = psb1[psb1.shape[0]-1]
    pctyp1[1] = 1.0 - pctyp1[0]
    psb2 = prbs[qctp2 == ctpvl[0]]
    pctyp2[0] = psb1[psb2.shape[0]-1]
    pctyp2[1] = 1.0 - pctyp2[0]


    xcts = numpy.arange(10)
    pctbr = numpy.array([pnslb[0],pnslb[1],pnslb[2],0.0, \
                         pctyp1[0],pctyp1[1],0.0, \
                         pctyp2[0],pctyp2[1],0.0])
    xlbs = ['0','1','2','','L','I','','L','I']

    p3 = pyplot.subplot(2,4,2)
    pyplot.bar(xcts+0.5,pctbr,color=c3[2])

    p3.plot([3.5, 3.5],[0.0,1.05],'-',c='#000000')
    p3.plot([6.5, 6.5],[0.0,1.05],'-',c='#000000')
    p3.text(1,1.0,'Number\nSlab',size=10,horizontalalignment='center',verticalalignment='top')
    p3.text(4.5,1.0,'Type\n1',size=10,horizontalalignment='center',verticalalignment='top')
    p3.text(7.5,1.0,'Type\n2',size=10,horizontalalignment='center',verticalalignment='top')
    p3.set_ylim(0.0,1.05)
    p3.set_xlim(-0.5,10.0)
    p3.set_xlabel('',size=11)
    p3.set_ylabel('Proportion',size=11)
    p3.xaxis.grid(color='#777777',linestyle='dotted')
    p3.yaxis.grid(color='#777777',linestyle='dotted')
    p3.set_xticks(xcts+0.5)
    p3.set_xticklabels(xlbs)
    for lb in p3.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p3.yaxis.get_ticklabels():
        lb.set_fontsize(10)
    pyplot.title('Cloud Types',size=12)

    # Cloud Pressure
    cp1vrnm = df.loc[(df.Group == 'CloudPres'),'Quantile_Name'].values[0]
    fqs = Dataset(qfile,'r')
    qcpr1 = fqs.variables[cp1vrnm][:]
    fqs.close()

    cp2vrnm = df.loc[(df.Group == 'CloudPres'),'Quantile_Name'].values[1]
    fqs = Dataset(qfile,'r')
    qcpr2 = fqs.variables[cp2vrnm][:]
    fqs.close()

    cp3vrnm = df.loc[(df.Group == 'CloudPres'),'Quantile_Name'].values[2]
    fqs = Dataset(qfile,'r')
    qcpr3 = fqs.variables[cp3vrnm][:]
    fqs.close()

    cp4vrnm = df.loc[(df.Group == 'CloudPres'),'Quantile_Name'].values[3]
    fqs = Dataset(qfile,'r')
    qcpr4 = fqs.variables[cp4vrnm][:]
    fqs.close()

    clpr = ['#A36C00','#985EBD']
    p4 = pyplot.subplot(2,4,6)
    pl1, = p4.plot(qcpr1,prbs,'-',c=clpr[0])
    pl2, = p4.plot(qcpr2,prbs,'--',c=clpr[0])
    pl3, = p4.plot(qcpr3,prbs,'-',c=clpr[1])
    pl4, = p4.plot(qcpr4,prbs,'--',c=clpr[1])
    p4.set_ylim(-0.05,1.05)
    p4.set_xlabel('Cloud Pressure Logit',size=11)
    p4.set_ylabel('CDF',size=11)
    p4.xaxis.grid(color='#777777',linestyle='dotted')
    p4.yaxis.grid(color='#777777',linestyle='dotted')
    for lb in p4.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p4.yaxis.get_ticklabels():
        lb.set_fontsize(10)
    leg = pyplot.legend([pl1, pl2, pl3, pl4],['CldBot','DPCld1','DPSlab','DPCld2'], loc = 'upper left',labelspacing=0.25,borderpad=0.25)
    for t in leg.get_texts():
        t.set_fontsize(10)
         
    pyplot.title('Cloud Slab Pressure Logit',size=12)
 
    # Cloud Fraction
    cfcvrnm = df.loc[(df.Group == 'CloudFrac'),'Quantile_Name'].values[0]
    fqs = Dataset(qfile,'r')
    qcfrc = fqs.variables[cfcvrnm][:]
    fqs.close()

    qcfrcmd = numpy.median(qcfrc,axis=[0,1])
    qcfrcmx = numpy.amax(qcfrc,axis=(0,1))
    qcfrcmn = numpy.amin(qcfrc,axis=(0,1))

    p5 = pyplot.subplot(2,4,3)
    p5.plot(qcfrcmn,prbs,'--',c=c3[2])
    p5.plot(qcfrcmx,prbs,'--',c=c3[2])
    p5.plot(qcfrcmd,prbs,'-',c=c3[2])
    p5.set_xlim(-0.1,1.1)
    p5.set_ylim(-0.05,1.05)
    p5.set_xlabel('Total Cloud Frac',size=11)
    p5.set_ylabel('CDF',size=11)
    p5.xaxis.grid(color='#777777',linestyle='dotted')
    p5.yaxis.grid(color='#777777',linestyle='dotted')
    for lb in p5.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p5.yaxis.get_ticklabels():
        lb.set_fontsize(10)
         
    pyplot.title('Cloud Fraction',size=12)
 
    # Cloud Frac 2 Logit
    cfc2vrnm = df.loc[(df.Group == 'CloudFrac'),'Quantile_Name'].values[1]
    fqs = Dataset(qfile,'r')
    qcfc2 = fqs.variables[cfc2vrnm][:]
    fqs.close()
    qcfc2md = numpy.median(qcfc2,axis=[0,1])

    cfc12vrnm = df.loc[(df.Group == 'CloudFrac'),'Quantile_Name'].values[2]
    fqs = Dataset(qfile,'r')
    qcfc12 = fqs.variables[cfc12vrnm][:]
    fqs.close()
    qcfc12md = numpy.median(qcfc12,axis=[0,1])

    p6 = pyplot.subplot(2,4,7)
    pl1, = p6.plot(qcfc2md,prbs,'-',c=c3[2])
    pl2, = p6.plot(qcfc12md,prbs,'--',c=c3[2])
    p6.set_ylim(-0.05,1.05)
    p6.set_xlabel('Cloud Fraction Logit',size=11)
    p6.set_ylabel('CDF',size=11)
    p6.xaxis.grid(color='#777777',linestyle='dotted')
    p6.yaxis.grid(color='#777777',linestyle='dotted')
    for lb in p6.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p6.yaxis.get_ticklabels():
        lb.set_fontsize(10)
    leg = pyplot.legend([pl1, pl2],['Slab 2','Overlap'], loc = 'upper left',labelspacing=0.25,borderpad=0.25)
    for t in leg.get_texts():
        t.set_fontsize(10)
         
    pyplot.title('Cloud Fraction Logit',size=12)
 
    # Cloud Top Temp
    ct1vrnm = df.loc[(df.Group == 'CloudTemp'),'Quantile_Name'].values[0]
    fqs = Dataset(qfile,'r')
    qctt1 = fqs.variables[ct1vrnm][:]
    fqs.close()

    ct2vrnm = df.loc[(df.Group == 'CloudTemp'),'Quantile_Name'].values[1]
    fqs = Dataset(qfile,'r')
    qctt2 = fqs.variables[ct2vrnm][:]
    fqs.close()

    p7 = pyplot.subplot(2,4,4)
    pl1, = p7.plot(qctt1,prbs,'-',c=c3[0])
    pl2, = p7.plot(qctt2,prbs,'--',c=c3[0])
    p7.set_ylim(-0.05,1.05)
    p7.set_xlabel('Cloud Top Temperature [K]',size=11)
    p7.set_ylabel('CDF',size=11)
    p7.xaxis.grid(color='#777777',linestyle='dotted')
    p7.yaxis.grid(color='#777777',linestyle='dotted')
    for lb in p7.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p7.yaxis.get_ticklabels():
        lb.set_fontsize(10)
    leg = pyplot.legend([pl1, pl2],['Slab 1','Slab 2'], loc = 'lower right',labelspacing=0.25,borderpad=0.25)
    for t in leg.get_texts():
        t.set_fontsize(10)
         
    pyplot.title('Cloud Top Temperature',size=12)
 
    # Cloud WC
    cwt1vrnm = df.loc[(df.Group == 'CloudWater'),'Quantile_Name'].values[0]
    fqs = Dataset(qfile,'r')
    qcngw1 = fqs.variables[cwt1vrnm][:]
    fqs.close()

    cwt2vrnm = df.loc[(df.Group == 'CloudWater'),'Quantile_Name'].values[1]
    fqs = Dataset(qfile,'r')
    qcngw2 = fqs.variables[cwt2vrnm][:]
    fqs.close()

    p8 = pyplot.subplot(2,4,8)
    pl1, = p8.plot(qcngw1,prbs,'-',c=c3[1])
    pl2, = p8.plot(qcngw2,prbs,'--',c=c3[1])
    p8.set_ylim(-0.05,1.05)
    p8.set_xlabel('Cloud Non-gas Water [g m^-2]',size=11)
    p8.set_ylabel('CDF',size=11)
    p8.xaxis.grid(color='#777777',linestyle='dotted')
    p8.yaxis.grid(color='#777777',linestyle='dotted')
    for lb in p8.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p8.yaxis.get_ticklabels():
        lb.set_fontsize(10)
    leg = pyplot.legend([pl1, pl2],['Slab 1','Slab 2'], loc = 'lower right',labelspacing=0.25,borderpad=0.25)
    for t in leg.get_texts():
        t.set_fontsize(10)
         
    pyplot.title('Cloud Non-gas Water',size=12)
 
    fig.subplots_adjust(bottom=0.1,top=0.92,left=0.08,right=0.92, \
                        hspace=0.35,wspace=0.35)
    pyplot.suptitle(fhdr)
    fig.savefig(outfile)
    pyplot.close()

    return
   
def state_corr_matrix(outfile, stfile, cnffile, fhdr, fgwdth=10, fghght=7):
    # Summary figures of marginal distributions 
    # outfile:   Output file with figures 
    # stfile:     Quantile file
    # cnffile:   State vector configuration file
    # fhdr:      Figure header

    df = pandas.read_csv(cnffile, dtype = {'Order':int, 'Group':str,  'ZScore_Name':str, 'Quantile_Name':str, \
                                           'Data_Name':str, 'Start':int, 'Length':int, 'DType':str })
    tsz = df['Length'].sum()
    szstr = '%d Total State Vector Elements' % (tsz)
    print(szstr)

    nrw = df.shape[0]
    df['CumLen'] = numpy.cumsum(df['Length'])

    drws =  [0,1,3,6,9,11,15]
    xdrws = [1,1,0,1,0, 1, 0] 
    xspts = []
    xlbs  = []
    xspts2 = []
    xlbs2  = []
    yspts = []
    ylbs  = []
    for j in range(len(drws)):
        if j == 0:
            xspts.append(df['CumLen'].values[drws[j]] / 2)
            xlbs.append(df['Group'].values[drws[j]])
        else:
            xspts.append(df['CumLen'].values[drws[j]-1] + (df['Length'].values[drws[j]] / 2))
            xlbs.append(df['Group'].values[drws[j]])
        if xdrws[j] == 1:
            xspts2.append(xspts[j])
            xlbs2.append(xlbs[j])
    for j in range(len(drws)-1,-1,-1):
        yspts.append(tsz - xspts[j])
        ylbs.append(xlbs[j])

    ## Work through state vector groups
    c3 = ["#0066FF","#B2387E","#39BEB1","#ACA4E2"]

    fqs = Dataset(stfile,'r')
    lglk = fqs.variables['logLike'][:]
    lksq = numpy.arange(lglk.shape[0])
    lkspt = lksq[lglk == numpy.amax(lglk)]
    cvmt = fqs.variables['state_cov'][lkspt[0],:,:]
    fqs.close()

    
    crmt = calculate_VPD.cov2cor(cvmt)

    igrd2, ggrd2 = numpy.meshgrid(numpy.arange(-0.5,tsz+0.5,1.0),numpy.arange(tsz+0.5,-0.5,-1.0))
    clst = ['#004cff','#6d87ff','#a4b3ff','#ccd2ff','#ffffff', \
            '#f5a29b','#f47667','#e25037','#be0000']
    cmpbr2 = colors.LinearSegmentedColormap.from_list("BlueRedCor",clst)

    fig = pyplot.figure(figsize=(fgwdth,fghght))

    p1 = pyplot.subplot(1,1,1)

    p1.yaxis.set_major_locator(mticker.FixedLocator(yspts))
    p1.xaxis.set_major_locator(mticker.FixedLocator(xspts2))
    p1.set_yticklabels(ylbs)
    p1.set_xticklabels(xlbs2)

    p1.set_xlim(-1,tsz+1)
    p1.set_ylim(-1,tsz+1)
    p1.xaxis.grid(color='#898989',linestyle='dotted')
    p1.yaxis.grid(color='#898989',linestyle='dotted')
    ccs = pyplot.pcolormesh(igrd2, ggrd2, crmt, cmap=cmpbr2, vmin=-1.0,vmax=1.0)
    p1.set_xlabel('State Vector Element',size=11)
    p1.set_ylabel('State Vector Element',size=11)
    for lb in p1.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p1.yaxis.get_ticklabels():
        lb.set_fontsize(10)
         
    pyplot.text(0.5,1.02,fhdr,horizontalalignment='center',fontsize=12, \
                transform=p1.transAxes)
    cax = pylab.axes([0.9,0.25,0.025,0.5])
    cbar = pyplot.colorbar(ccs,ticks=[-1.0,-0.5,0.0,0.5,1.0],cax=cax,orientation='vertical')
    cbar.ax.set_yticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
 
    fig.subplots_adjust(bottom=0.15,top=0.9,left=0.1,right=0.85, \
                        hspace=0.2,wspace=0.25)
    fig.savefig(outfile,bbox_inches="tight")
    pyplot.close()

    return

