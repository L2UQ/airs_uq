# Map QFlg behavior over replicates

library(ncdf4)
library(colorspace)
library(ggplot2)
library(reshape2)

source("mapfunctions.R")

theme_mat = theme_bw() 
theme_mat$axis.title.x$size = 12
theme_mat$axis.text.x$size = 12
theme_mat$axis.title.y$size = 12
theme_mat$axis.text.y$size = 12
theme_mat$plot.title$size = 14
theme_mat$plot.title$hjust = 0.5

ar7 = rainbow_hcl(7,c=65,l=55)
r5 = diverge_hcl(5,h=c(260,0),c=80,l=c(40,80),power=1)

nact = function(dtarr) {
  dtvc = as.vector(dtarr)
  natot = length(dtvc[is.na(dtvc)])
  return(natot)
}

#smpal = sequential_hcl(51,h=135,c.=c(0,75),l=c(100,25),power=1.0)

rgname = 'Midwest'
rgn = 'MW'
hrs = 21
#scnrws = c(1,4,21)
scnrws = c(24,34,45)
yr = 2018
season = "DJF"


for (i in seq(1,length(scnrws))) {
    exptfl = sprintf("CONUS_AIRS_%s_%04d_%02dUTC_%s_SR%02d_SimSARTAStates_Mix_CloudFOV_scanRow=%d_UQ_Output.h5",
                     season,yr,hrs,rgn,scnrws[i],scnrws[i])
    nc1 = nc_open(exptfl)
    lat = ncvar_get(nc1,"latitude")
    lon = ncvar_get(nc1,"longitude")
    sprs = ncvar_get(nc1,"spres")
    tairs = ncvar_get(nc1,"airs_ptemp")
    lev = as.vector(ncvar_get(nc1,"level"))
    cfrc = ncvar_get(nc1,"cfrac")
    nc_close(nc1)

    # Land Fraction
    supfl = sprintf("CONUS_AIRS_%s_%04d_%02dUTC_%s_SR%02d_SupportVars.h5",
                     season,yr,hrs,rgn,scnrws[i],scnrws[i])
    nc2 = nc_open(supfl)
    lfrc = as.vector(ncvar_get(nc2,"landFrac"))
    nc_close(nc2)
    
      
    # Scan Row indices
    scnrps = seq(2,90,by=3)
    scnidx = scnrws[i]*3 - 1
    
    sfcfl = sprintf("CONUS_AIRS_%s_%04d_%02dUTC_%s_SR%02d_Sfc_UQ_Output.h5",
                    season,yr,hrs,rgn,scnrws[i])
    nc2 = nc_open(sfcfl)
    nstret = as.vector(ncvar_get(nc2,"TSfcAir_Retrieved"))
    tflg =  as.vector(ncvar_get(nc2,"TSfcAir_QC"))
    nc_close(nc2)
    
    cfcmn = apply(cfrc,1,mean)
    cfcarr = matrix(cfcmn,nrow=30,ncol=45*10)
    cfccmn = apply(cfcarr,1,mean)
    nstarr = matrix(nstret,nrow=30,ncol=45*10)
    qflgarr = matrix(tflg,nrow=30,ncol=45*10)
    naspts = apply(nstarr,1,FUN=nact)
    prsarr = matrix(sprs,nrow=30,ncol=45*10)
    gdind = matrix(as.integer(qflgarr < 2),nrow=30,ncol=45*10)
    gdpct = apply(gdind,1,sum) * 1.0 / ncol(gdind)
    
    msfrm = data.frame(ScnIdx = seq(1,30),Longitude = lon[scnrps,scnidx], LandFrac=lfrc,
                       Latitude = lat[scnrps,scnidx], GoodPct = gdpct, PSfc = prsarr[,1], CFrac=cfccmn)
    msfrm$ScanRow = scnrws[i]
    
    if (i == 1) {
        fmsfrm = msfrm
    }
    else {
        fmsfrm = rbind(fmsfrm,msfrm)
    }
}

# Map It
# Read Map Info, create lat/lon labels
usst = centermap2("States.csv",center=-88,intrarg = FALSE)
lnlb = lonlbs(c(-100,-95,-90,-85,-80),center=-88.0)
ltlb = latlbs(seq(35,50,by=5))

fmsfrm = fklon(fmsfrm,lonvar = "Longitude",center=-88)
fmsfrm$ScanRow = factor(fmsfrm$ScanRow)
fmsfrm$MsgStatus = "None"
fmsfrm$MsgStatus[fmsfrm$NACount > 0] = "All"

tstr = sprintf("Near-Sfc Temp QF Good: %s %s %02d UTC",season,rgname,hrs)
ggeo = ggplot(fmsfrm,aes(x=fk360,y=Latitude)) + geom_point(aes(color=GoodPct)) + 
  geom_path(aes(x=fk360,y=Y,group=group2), data=usst) +
  scale_x_continuous("",breaks=lnlb$fk360,labels=parse(text=lnlb$labxpr),limits=c(-14,14)) + 
  scale_y_continuous("",breaks=ltlb$origlat,labels=parse(text=ltlb$labxpr),limits=c(34,52)) + 
  scale_color_gradientn("PropGood",colors = r5,limits=c(0,1)) + 
  ggtitle(tstr) + theme_mat + coord_equal()
pnm = sprintf("MapQFGood_%s_%s_%02dUTC.pdf",season,rgn,hrs)
pdf(pnm,width = 8,height=6)
print(ggeo)
dev.off()

ggeo = ggplot(fmsfrm,aes(x=fk360,y=Latitude)) + geom_point(aes(color=LandFrac)) + 
  geom_path(aes(x=fk360,y=Y,group=group2), data=usst) +
  scale_x_continuous("",breaks=lnlb$fk360,labels=parse(text=lnlb$labxpr),limits=c(-14,14)) + 
  scale_y_continuous("",breaks=ltlb$origlat,labels=parse(text=ltlb$labxpr),limits=c(34,52)) + 
  scale_color_gradientn("LndFrc",colors = r5,limits=c(0,1)) + 
  ggtitle(tstr) + theme_mat + coord_equal()
pnm = sprintf("MapLandFrac_%s_%s_%02dUTC.pdf",season,rgn,hrs)
pdf(pnm,width = 8,height=6)
print(ggeo)
dev.off()

# True cloud fraction?
ggeo = ggplot(fmsfrm,aes(x=fk360,y=Latitude)) + geom_point(aes(color=CFrac)) + 
  geom_path(aes(x=fk360,y=Y,group=group2), data=usst) +
  scale_x_continuous("",breaks=lnlb$fk360,labels=parse(text=lnlb$labxpr),limits=c(-14,14)) + 
  scale_y_continuous("",breaks=ltlb$origlat,labels=parse(text=ltlb$labxpr),limits=c(34,52)) + 
  scale_color_gradientn("LndFrc",colors = r5,limits=c(0,1)) + 
  ggtitle(tstr) + theme_mat + coord_equal()
pnm = sprintf("MapCloudFrac_%s_%s_%02dUTC.pdf",season,rgn,hrs)
pdf(pnm,width = 8,height=6)
print(ggeo)
dev.off()
