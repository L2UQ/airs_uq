library(ggplot2)
library(reshape2)
library(plyr)
library(maps)

valmn = function(dtfrm) {
    meanout = mean(dtfrm$value,na.rm=TRUE)
    medout = median(dtfrm$value,na.rm=TRUE)
    frameout = data.frame(mean=meanout,median=medout)
    return(frameout)
}

chckgrp = function(mpfrm) {
    mpfrm = mpfrm[order(mpfrm$group,mpfrm$order),]
	mpfrm$dif1 = c(NA,diff(mpfrm$fk360))
	return(mpfrm)
}

chckgrp2 = function(mpfrm) {
  mpfrm = mpfrm[order(mpfrm$plyidx,mpfrm$plyord),]
  mpfrm$dif1 = c(NA,diff(mpfrm$fk360))
  return(mpfrm)
}

grpct = function(dfrm) {
	frmout = data.frame(ctgrp = length(unique(dfrm$group2)))
	return(frmout)
}

spltna = function(dfrm) {
    # Split at NA's in the difference variable
    sq1 = seq(1,length(dfrm$dif1))	
	tsq = sq1[is.na(dfrm$dif1)]
	if (length(tsq) > 1) {
 	    tls = c(tsq[seq(2,length(tsq))] - 1, length(sq1))
 	}
 	else {
 		tls = length(sq1)
 	}
	dfrm$group2 = paste(dfrm$group,1,sep="_")
	for (i in seq(1,length(tsq))) {
		dfrm$group2[tsq[i]:tls[i]] = paste(dfrm$group[tsq[i]:tls[i]] ,i,sep="_")		
	}
	return(dfrm)
}

spltna2 = function(dfrm) {
  # Split at NA's in the difference variable
  sq1 = seq(1,length(dfrm$dif1))	
  tsq = sq1[is.na(dfrm$dif1)]
  if (length(tsq) > 1) {
    tls = c(tsq[seq(2,length(tsq))] - 1, length(sq1))
  }
  else {
    tls = length(sq1)
  }
  dfrm$group2 = paste(dfrm$plyidx,1,sep="_")
  for (i in seq(1,length(tsq))) {
    dfrm$group2[tsq[i]:tls[i]] = paste(dfrm$plyidx[tsq[i]:tls[i]] ,i,sep="_")		
  }
  return(dfrm)
}


centermap = function(database,center=180,intrarg=TRUE) {
	# Make a map centered on a chosen longitude, using map_data
	require(ggplot2)
	frm1 = map_data(database,interior=intrarg)
	frm1$fk360 = (frm1$long - center) %% 360.0
	frm1$fk360[frm1$fk360 > 180] = frm1$fk360[frm1$fk360 > 180] - 360
	
	frm2 = ddply(frm1,c("group"),.fun=chckgrp)
	frm2$dif1[frm2$dif1 < -100 & !is.na(frm2$dif1)] = NA
	frm2$dif1[frm2$dif1 > 100 & !is.na(frm2$dif1)] = NA
	frm3 = ddply(frm2,c("group"),.fun=spltna)
	return(frm3)
}

centermap2 = function(flnm,center=180,intrarg=TRUE) {
  # Make a map centered on a chosen longitude, using map_data
  require(ggplot2)
  frm1 = read.csv(flnm,header = TRUE)
  frm1$fk360 = (frm1$X - center) %% 360.0
  frm1$fk360[frm1$fk360 > 180] = frm1$fk360[frm1$fk360 > 180] - 360
  
  frm2 = ddply(frm1,c("plyidx"),.fun=chckgrp2)
  frm2$dif1[frm2$dif1 < -100 & !is.na(frm2$dif1)] = NA
  frm2$dif1[frm2$dif1 > 100 & !is.na(frm2$dif1)] = NA
  frm3 = ddply(frm2,c("plyidx"),.fun=spltna2)
  return(frm3)
}

lonlbs = function(lonbrks,center=180,prec=0) {
	# Return nicely formatted labels, with arbitrary map center
	# Assume lonbrks are (-180,180) or (0,360)
	frm1 = data.frame(origlon=lonbrks)
	frm1$fk360 = (frm1$origlon - center) %% 360.0
	frm1$fk360[frm1$fk360 > 180] = frm1$fk360[frm1$fk360 > 180] - 360
    frm1$labnum = frm1$origlon
    frm1$labnum[frm1$origlon < 0] = abs(frm1$origlon[frm1$origlon < 0])
    frm1$labnum[frm1$origlon > 180] = frm1$origlon[frm1$origlon > 180] - 360
    frm1$labdir = "* W"
    frm1$labdir[frm1$origlon == 0 | frm1$origlon == 180.0 | frm1$origlon == -180.0] = ""
    frm1$labdir[frm1$origlon > 0 & frm1$origlon < 180.0] = "* E"
    if (prec == 0) {
        frm1$lablab = paste(sprintf("%d",frm1$labnum),"* degree",frm1$labdir)
    }
    else if (prec == 1) {
      frm1$lablab = paste(sprintf("%.1f",frm1$labnum),"* degree",frm1$labdir)
    }
    frm1$labxpr = as.expression(frm1$lablab)
    frm1 = frm1[order(frm1$fk360),]
    return(frm1)
    
}

latlbs = function(latbrks,prec=0) {
	# Return nicely formatted labels
	# Assume latbrks are (-90,90)
	frm1 = data.frame(origlat=latbrks)
    frm1$labnum = frm1$origlat
    frm1$labnum[frm1$origlat < 0] = abs(frm1$origlat[frm1$origlat < 0])
    frm1$labdir = "* N"
    frm1$labdir[frm1$origlat == 0] = ""
    frm1$labdir[frm1$origlat < 0] = "* S"
    if (prec == 0) {
        frm1$lablab = paste(sprintf("%d",frm1$labnum),"* degree",frm1$labdir)
    }
    else if (prec == 1) {
        frm1$lablab = paste(sprintf("%.1f",frm1$labnum),"* degree",frm1$labdir)
    }
    frm1$labxpr = as.expression(frm1$lablab)
    frm1 = frm1[order(frm1$origlat),]
    return(frm1)
}

fklon = function(dfrm,lonvar,center=180) {
	# Convert a data frame with a longitude coordinate to a system centered at center
	dfrm$fk360 = (dfrm[,lonvar] - center) %% 360.0
	dfrm$fk360[dfrm$fk360 > 180] = dfrm$fk360[dfrm$fk360 > 180] - 360
    return(dfrm)
}
