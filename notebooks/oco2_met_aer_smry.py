# Load and summarize OCO-2 aerosol info

import earthaccess
import xarray as xr
import s3fs
import numpy
import pandas
import datetime
import matplotlib
from matplotlib import pyplot
from matplotlib import colors
import matplotlib.ticker as mticker
import math

# Surface choice
sfctxt = 'Ocean'
sfcchc = 0
seastxt = '2018-08'

# LtCO2 and LtMet from OCO-2
# OCO2_L2_Lite_FP
short_name = 'OCO2_L2_Lite_FP'
version = '11.1r'
start_time = '2018-08-01'
end_time = '2018-08-31'

results = earthaccess.search_data(
    short_name=short_name,
    version=version,
    cloud_hosted=True,
    temporal=(start_time,end_time)
)

s3_urls_v11 = [granule.data_links(access="direct")[0] for granule in results]
nlite = len(s3_urls_v11)
print(nlite)

auth = earthaccess.login(strategy="netrc")

daac = 'GES_DISC'
temp_s3_credentials = earthaccess.get_s3_credentials(daac)
def begin_s3_direct_access(url: str=daac):
    response = earthaccess.get_s3_credentials(daac)
    return s3fs.S3FileSystem(key=response['accessKeyId'],
                             secret=response['secretAccessKey'],
                             token=response['sessionToken'],
                             client_kwargs={'region_name':'us-west-2'})
fs = begin_s3_direct_access()

# Loop through files
nsdglt = 0
for i in range(nlite):
    ltnc = xr.open_dataset(fs.open(s3_urls_v11[i]),
                           decode_cf=True,engine='h5netcdf')
    ltsdg = ltnc.sounding_id.values[:]
    ltlat = ltnc.latitude.values[:]
    ltlon = ltnc.longitude.values[:]
    ltxco2 = ltnc.xco2.values[:]
    ltflg = ltnc.xco2_quality_flag.values[:]
    ltnc.close()

    ltncrtr = xr.open_dataset(fs.open(s3_urls_v11[i]),
                              decode_cf=True,group="Retrieval",engine='h5netcdf')
    sfctp = ltncrtr.surface_type.values[:]
    ltncrtr.close()
    # Surface type: 0=water, 1=land

    # Orbit info
    ltncsdg = xr.open_dataset(fs.open(s3_urls_v11[i]),
                              decode_cf=True,group="Sounding",engine='h5netcdf')
    orbit = ltncsdg.orbit.values[:]
    ltncsdg.close()

    ltsdg = ltsdg.astype(numpy.int64)
    sfctpi = sfctp.astype(numpy.int16)
    orbit = orbit.astype(numpy.int32)
    ltflgi = ltflg.astype(numpy.int16)

    ltfrm = pandas.DataFrame({'SoundingID': ltsdg, 'Orbit': orbit, 'SfcType':sfctpi,
                              'Latitude': ltlat, 'Longitude': ltlon, 'XCO2': ltxco2, 'V11QFlag': ltflgi})
    # QF subset
    ltfrm = ltfrm[ltfrm['V11QFlag'] == 0]
    #print(ltfrm.dtypes)
    
    if nsdglt == 0:
        lt_all = ltfrm
    else:
        lt_all = pandas.concat([lt_all,ltfrm], ignore_index=True) 
    nsdglt = nsdglt + ltsdg.shape[0]

## Subset and group
def qsummary(df,grpvr,vrlst):
    # Summarize with quantiles
    nmtch = df.shape[0] 
    dfout = pandas.DataFrame({'NSmp' : nmtch}, index=[0])
    #dfout[grpvr] = df[grpvr].values[0]
    for j in range(len(vrlst)):
        tmpdt = df[vrlst[j]]
        dtvld = tmpdt[numpy.isfinite(tmpdt)]
        dtvld = dtvld[dtvld != 0.0]
        vrnm = '%s_Med' % (vrlst[j])
        dfout[vrnm] = numpy.median(dtvld)

    return dfout

print(nsdglt)

# Sfc choice
sfcfrm = lt_all[lt_all['SfcType'] == sfcchc]

grpfrm = sfcfrm.groupby(['Orbit'])
sfcqs = grpfrm.apply(qsummary,include_groups=False,grpvr='Orbit',vrlst=['XCO2','Latitude','Longitude'])
sfcqs.reset_index(drop=False,inplace=True)
print(sfcqs.shape)

# OCO2_L2_Met, use same start time/end time
short_name = 'OCO2_L2_Met'
version = '11r'

results = earthaccess.search_data(
    short_name=short_name,
    version=version,
    cloud_hosted=True,
    temporal=(start_time,end_time)
)

s3_urls_l2met = [granule.data_links(access="direct")[0] for granule in results]
print(len(s3_urls_l2met))

# Create a data frame with L2Met info
l2sfrm = pandas.DataFrame({'S3File': s3_urls_l2met})
l2sfrm['ModeOrbStr'] = l2sfrm['S3File'].str.extract(r'(L2Met[A-Z]{2}_[0-9]{5}[a-z]{1})')
l2sfrm.dropna(subset=['ModeOrbStr'],inplace=True)
l2sfrm['OrbStr'] = l2sfrm['ModeOrbStr'].str.replace('L2Met[A-Z]{2}_', '', regex=True)
l2sfrm['Orbit'] = l2sfrm['OrbStr'].str.replace('[a-z]{1}','',regex=True)
l2sfrm['Orbit'] = l2sfrm['Orbit'].astype(numpy.int32)
print(l2sfrm[0:20])

# Merge Lite and L2
mrgorb = pandas.merge(sfcqs,l2sfrm, on='Orbit', how='inner')
print(mrgorb.shape)

# Loop L2Met collection
nl2met = mrgorb.shape[0]
nbtch = 0

import h5py

aerlst = []
nsdgaer = 0
# Aerosol Gaussian params have dimension [nfrm,ncrs,naer,4] 
#  - extract 2nd (height) and 4th (AOD) elements
for k in range(nl2met): 
    s3nmcr = mrgorb['S3File'].values[k]
    print(s3nmcr)
    l2h5 = h5py.File(fs.open(s3nmcr),'r')
    l2sdg = l2h5['/SoundingGeometry/sounding_id'][:,:]
    aergaus = l2h5['/Aerosol/composite_aod_gaussian_met'][:,:,:,:]
    if k == 0:
        aernms = l2h5['/Metadata/CompositeAerosolTypes'][:]
        naer = aernms.shape[0]
        for j in range(naer):
            aerlst.append(aernms[j].decode('utf-8'))
    l2h5.close()

    nfrm = l2sdg.shape[0]
    ncrs = l2sdg.shape[1]

    frmsq = numpy.arange(nfrm)
    crssq = numpy.arange(ncrs)
    frmspt = numpy.repeat(frmsq,ncrs)
    crsspt = numpy.tile(crssq,nfrm)

    # Set up data frame
    l2gfrm = pandas.DataFrame({'SoundingID': l2sdg.flatten(), 'FrameIdx': frmspt, 'FPIdx': crsspt})
    for j in range(naer):
        nmaod = '%s_LogAOD' % (aerlst[j])
        nmht = '%s_Ht' % (aerlst[j])
        l2gfrm[nmaod] = numpy.log(aergaus[:,:,j,3].flatten())
        l2gfrm[nmht] = aergaus[:,:,j,1].flatten()
    print(l2gfrm[12:20])

    # Subset lite and merge
    ltgrn = lt_all[ lt_all['Orbit'] == mrgorb['Orbit'].values[k]]
    aermrg = pandas.merge(l2gfrm, ltgrn, on='SoundingID', how='inner')
    aermrg['L2MetS3'] = s3nmcr

    if nsdgaer == 0:
        aer_all = aermrg
    else:
        aer_all = pandas.concat([aer_all,aermrg], ignore_index=True) 
    nsdgaer = nsdgaer + aermrg.shape[0]

print(aernms)
print(l2sdg.dtype)
print(aer_all.shape)

# Summarize and plot
# Latitude binning
ltfrq = 15.0
aer_all['GridLat'] = numpy.floor(aer_all['Latitude']/ltfrq) * ltfrq + ltfrq / 2.0

fig, axes = pyplot.subplots(nrows=3,ncols=2,figsize=(8,9))
print(axes.shape)

for j in range(naer):
    nmaod = '%s_LogAOD' % (aerlst[j])
    aer_all.boxplot(column=[nmaod], by="GridLat", ax=axes.flatten()[j])

fig.delaxes(axes[2,1]) # remove empty subplot
tstr = '%s %s Log AOD' % (sfctxt, seastxt)
fig.suptitle(tstr,fontsize=12)

pyplot.tight_layout() 
pltnm = 'OCO2_L2Met_LogAOD_%s_%s.png' % (sfctxt,seastxt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

# Summarize heights
fig, axes = pyplot.subplots(nrows=3,ncols=2,figsize=(8,9))
for j in range(naer):
    nmaod = '%s_LogAOD' % (aerlst[j])
    nmht = '%s_Ht' % (aerlst[j])
    aer_sbst = aer_all[ aer_all[nmaod] >= -6.0]
    aer_sbst.boxplot(column=[nmht], by="GridLat", ax=axes.flatten()[j])

fig.delaxes(axes[2,1]) # remove empty subplot
tstr = '%s %s Aerosol Profile Height' % (sfctxt, seastxt)
fig.suptitle(tstr,fontsize=12)

pyplot.tight_layout() 
pltnm = 'OCO2_L2Met_AerHt_%s_%s.png' % (sfctxt,seastxt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

## Aerosol profile plotting

# Color scheme
typlst = ['DU','SS','BC','OC','SO']
lbs = ['Dust','Sea Salt','Black Carbon','Organic Carbon','Sulfate']
clst = ["#9C6E3D","#008597", "#777777","#51833B","#9963A4"]
brkpt = [0.7, 0.9, 0.6, 0.6, 0.6]
typfrm = pandas.DataFrame({'AerAbbrv': typlst, 'AerLabel': lbs, 'AerColor': clst, 'AerBrk': brkpt})

# MERRA Levels
mrlvs = pandas.read_csv("MERRA2_SigLev.csv", \
                        dtype = {'LevIdx':int, 'SigmaCrd':float, 'SigStd':float}, \
                        encoding='utf-8-sig')
nlv = mrlvs.shape[0]

# Profile arrays
nexmp = 100
prfarr = numpy.zeros( (naer,2,nexmp,nlv), dtype=numpy.float32)
for j in range(naer):
    nmaod = '%s_LogAOD' % (aerlst[j])
    nmht = '%s_Ht' % (aerlst[j])
    print(nmaod)
    aer_sbst = aer_all[ aer_all[nmaod] >= -6.0]
    
    aerbrk = typfrm['AerBrk'].values[j]
    print(aerbrk)
    aerlw = aer_sbst[aer_sbst[nmht] >= aerbrk]
    print(aerlw.shape)
    if aerlw.shape[0] > nexmp:
        nlw = nexmp
        stplw = int(math.floor(aerlw.shape[0] / nexmp))
        print(stplw)
        sqlw = numpy.arange( 0, stplw*nexmp, stplw)
    else:
        nlw = aerlw.shape[0]
        sqlw = numpy.arange(aerlw.shape[0])
    for i in range(sqlw.shape[0]):
        s3nmcr = aerlw['L2MetS3'].values[sqlw[i]]
        sdidx = aerlw['FrameIdx'].values[sqlw[i]]
        fpidx = aerlw['FPIdx'].values[sqlw[i]]
        l2h5 = h5py.File(fs.open(s3nmcr),'r')
        prfarr[j,0,i,:] = l2h5['/Aerosol/composite_aod_profile_met'][sdidx,fpidx,j,:]
        l2h5.close()
    
    aerhi = aer_sbst[aer_sbst[nmht] <= aerbrk]
    print(aerhi.shape)
    if aerhi.shape[0] > nexmp:
        nhi = nexmp
        stphi = int(math.floor(aerhi.shape[0] / nexmp))
        print(stphi)
        sqhi = numpy.arange( 0, stphi*nexmp, stphi)
    else:
        nhi = aerhi.shape[0]
        sqhi = numpy.arange(aerhi.shape[0])
    for i in range(sqhi.shape[0]):
        s3nmcr = aerhi['L2MetS3'].values[sqhi[i]]
        sdidx = aerhi['FrameIdx'].values[sqhi[i]]
        fpidx = aerhi['FPIdx'].values[sqhi[i]]
        l2h5 = h5py.File(fs.open(s3nmcr),'r')
        prfarr[j,1,i,:] = l2h5['/Aerosol/composite_aod_profile_met'][sdidx,fpidx,j,:]
        l2h5.close()

    print(numpy.amax(prfarr[j,0,:,:]))
    print(numpy.amax(prfarr[j,1,:,:]))

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    lv3 = int(math.floor(lv/3))
    return tuple(int(value[i:i+lv3], 16) for i in range(0, lv, lv3))

fig = pyplot.figure(figsize=(8,15))
for j in range(naer):

    rgbvl = hex_to_rgb(clst[j])
    r1 = rgbvl[0] / 256.0
    g1 = rgbvl[1] / 256.0
    b1 = rgbvl[2] / 256.0

    pspt = 2*j + 1
    p1 = pyplot.subplot(naer,2,pspt)
    for i in range(nexmp):
        p1.plot(prfarr[j,0,i,:],1.0-mrlvs['SigmaCrd'],'-',linewidth=0.4,color=(r1,g1,b1,0.35))
    p1.set_ylim(-0.05,1.05)
    p1.set_xlim(0,0.028)
    p1.yaxis.set_major_locator(mticker.FixedLocator(numpy.arange(0.0,1.2,0.2)))
    p1.set_yticklabels(['1.0','0.8','0.6','0.4','0.2','0.0'])
    p1.xaxis.grid(color='#898989',linestyle='dotted')
    p1.yaxis.grid(color='#898989',linestyle='dotted')
    xlb = '%s Aerosol Mixing Ratio' % (typlst[j])
    p1.set_xlabel(xlb)
    p1.set_ylabel(r'$\sigma = \frac{p}{p_s}$')
    tstr = '%s Low Profile Height' % (typlst[j])
    pyplot.title(tstr,fontsize=11)
    for lb in p1.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p1.yaxis.get_ticklabels():
        lb.set_fontsize(10)

    pspt = 2*j + 2
    p2 = pyplot.subplot(naer,2,pspt)
    for i in range(nexmp):
        p2.plot(prfarr[j,1,i,:],1.0-mrlvs['SigmaCrd'],'-',linewidth=0.4,color=(r1,g1,b1,0.35))
    p2.set_ylim(-0.05,1.05)
    p2.set_xlim(0,0.028)
    p2.yaxis.set_major_locator(mticker.FixedLocator(numpy.arange(0.0,1.2,0.2)))
    p2.set_yticklabels(['1.0','0.8','0.6','0.4','0.2','0.0'])
    p2.xaxis.grid(color='#898989',linestyle='dotted')
    p2.yaxis.grid(color='#898989',linestyle='dotted')
    xlb = '%s Aerosol Mixing Ratio' % (typlst[j])
    p2.set_xlabel(xlb)
    p2.set_ylabel(r'$\sigma = \frac{p}{p_s}$')
    tstr = '%s High Profile Height' % (typlst[j])
    pyplot.title(tstr,fontsize=11)
    for lb in p1.xaxis.get_ticklabels():
        lb.set_fontsize(10)
    for lb in p1.yaxis.get_ticklabels():
        lb.set_fontsize(10)

tstr = '%s %s Aerosol Profile Examples' % (sfctxt, seastxt)
fig.suptitle(tstr,fontsize=12)

pyplot.tight_layout() 
pltnm = 'OCO2_L2Met_AerProf_%s_%s.png' % (sfctxt,seastxt)
pyplot.savefig(pltnm,bbox_inches="tight",dpi=200)
pyplot.close()

