# Scripts supporting AIRS retrieval experiment post-processing

library(mclust)
library(ncdf4)
library(colorspace)
library(ggplot2)
library(plyr)
library(reshape2)
library(matrixcalc)

grpsmry = function(dtfrm, vrnm = "value") {
  dtsb = dtfrm[,vrnm]
  dtsb = dtsb[!is.na(dtsb)]
  sfrm = data.frame(Mean = mean(dtsb), StdDev = sd(dtsb))
  return(sfrm)
}

vec_outer = function(vec) {
    # Quick vector outer product
    crsprd = outer(vec,vec,FUN = "*")
    return(crsprd)
}

cldsmry = function(cldarr) {
  # Summarize a 3x3 cloud fraction array
  cldvc = as.vector(cldarr)
  cldmn = mean(cldarr)
  cldsd = sd(cldarr)
  nclr = length(cldarr[cldarr == 0.0])
  novc = length(cldarr[cldarr == 1.0])
  frmout = data.frame(CFrcMean=cldmn,CFrcSD=cldsd,NClr=nclr,NOvc=novc)
  return(frmout)
}

expt_tprf_arr = function(dfrm, expdir = NULL, rpvrs = c("MinRow","ModRow","MaxRow")) {
  # Output array of retrieved temperature profiles from simulation experiment
  # dfrm   - data frame with identifying information
  # rpvrs  - variable names for replicate experiments

  nrp = length(rpvrs)
  for (k in seq(1,nrp)) {
    # CONUS_AIRS_DJF_2018_21UTC_NE_SR17_SimSARTAStates_Mix_CloudFOV_scanRow=17_UQ_Output.h5
    scrw = dfrm[1,rpvrs[k]]
    if (is.null(expdir)) {
        ofl = sprintf("CONUS_AIRS_%s_%d_%02dUTC_%s_SR%02d_SimSARTAStates_Mix_CloudFOV_scanRow=%d_UQ_Output.h5",
                      dfrm$Season[1],dfrm$Year[1],dfrm$Hour[1],dfrm$Abbrev[1],scrw,scrw)
    }
    else {
        ofl = sprintf("%s/CONUS_AIRS_%s_%d_%02dUTC_%s_SR%02d_SimSARTAStates_Mix_CloudFOV_scanRow=%d_UQ_Output.h5",
                      expdir,dfrm$Season[1],dfrm$Year[1],dfrm$Hour[1],dfrm$Abbrev[1],scrw,scrw)
    }
    nc1 = nc_open(ofl)
    tmprt = ncvar_get(nc1,"airs_ptemp")
    tmpqc = ncvar_get(nc1,"airs_ptemp_qc")
    psfc = ncvar_get(nc1,"spres")
    lev = as.vector(ncvar_get(nc1,"level"))
    nc_close(nc1)


    # Array of retrieved temps
    if (k == 1) {
      tmparr = tmprt
    }
    else {
      tmparr = cbind(tmparr,tmprt)
    }
  }
  #dimnames(tmparr)
  tmparr[tmparr < 0] = NA
  return(list(tmparr,lev))

}

expt_tmp_diag = function(dfrm, expdir = NULL, rpvrs = c("MinRow","ModRow","MaxRow")) {
  # Constuct dataset of possible retrieved NST diagnostics
  # dfrm   - data frame with identifying information
  # rpvrs  - variable names for replicate experiments

  nrp = length(rpvrs)
  for (k in seq(1,nrp)) {
    # CONUS_AIRS_DJF_2018_21UTC_NE_SR17_SimSARTAStates_Mix_CloudFOV_scanRow=17_UQ_Output.h5
    scrw = dfrm[1,rpvrs[k]]
    if (is.null(expdir)) {
        ofl = sprintf("CONUS_AIRS_%s_%d_%02dUTC_%s_SR%02d_SimSARTAStates_Mix_CloudFOV_scanRow=%d_UQ_Output.h5",
                      dfrm$Season[1],dfrm$Year[1],dfrm$Hour[1],dfrm$Abbrev[1],scrw,scrw)
    }
    else {
        ofl = sprintf("%s/CONUS_AIRS_%s_%d_%02dUTC_%s_SR%02d_SimSARTAStates_Mix_CloudFOV_scanRow=%d_UQ_Output.h5",
                      expdir,dfrm$Season[1],dfrm$Year[1],dfrm$Hour[1],dfrm$Abbrev[1],scrw,scrw)
    }
    nc1 = nc_open(ofl)
    nsttr = as.vector(ncvar_get(nc1,"sarta_nst"))
    nstairs = as.vector(ncvar_get(nc1,"airs_nst"))
    ncld = as.vector(ncvar_get(nc1,"numCloud"))
    cfrcstd = ncvar_get(nc1,"CldFrcStd")
    pcld = ncvar_get(nc1,"PCldTopStd")
    tmprt = ncvar_get(nc1,"airs_ptemp")
    tmpqc = ncvar_get(nc1,"airs_ptemp_qc")
    psfc = ncvar_get(nc1,"spres")
    lev = as.vector(ncvar_get(nc1,"level"))
    nc_close(nc1)

    # CONUS_AIRS_DJF_2018_21UTC_NW_SR16_Sfc_UQ_Output.h5
    if (is.null(expdir)) {
        sffl = sprintf("CONUS_AIRS_%s_%d_%02dUTC_%s_SR%02d_Sfc_UQ_Output.h5",
                       dfrm$Season[1],dfrm$Year[1],dfrm$Hour[1],dfrm$Abbrev[1],scrw)
    }
    else {
        sffl = sprintf("%04d/NearSurface/CONUS_AIRS_%s_%d_%02dUTC_%s_SR%02d_Sfc_UQ_Output.h5",
                       dfrm$Year[1],dfrm$Season[1],dfrm$Year[1],dfrm$Hour[1],dfrm$Abbrev[1],scrw)
    }

    nc1 = nc_open(sffl)
    nsttr = as.vector(ncvar_get(nc1,"TSfcAir_True"))
    nstairs = as.vector(ncvar_get(nc1,"TSfcAir_Retrieved"))
    tsfcqc = as.vector(ncvar_get(nc1,"TSfcAir_QC"))
    nc_close(nc1)

    print(table(tsfcqc))
    cfrcstd[cfrcstd < 0] = NA
    cfrcsm = cfrcstd[1,,,] + cfrcstd[2,,,]
    cldfrm1 = adply(cfrcsm,.margins = 3,.fun = cldsmry,.progress = "text")

    # Data Frame with various diagnostic variables
    tinv = tmprt[91,] - nstairs
    lvsq = seq(1,length(lev))
    for (i in seq(1,length(tinv)) ) {
        if (psfc[i] <= lev[92]) {
            lmn = max(lvsq[lev <= psfc[i]])
            tinv[i] = tmprt[lmn-3,i] - nstairs[i]
        }
    }

    sdsq = seq(1,length(nsttr))
    exprp = rep(1:10,each=45*30)
    frmexp = data.frame(SdgID = sdsq, ExpRep = exprp, NSTTrue = nsttr, QFlgNST = tsfcqc, NSTRtrv = nstairs,
                        TDif850 = tinv, NCloud = ncld, CFrcMean = cldfrm1$CFrcMean, CFrcSD = cldfrm1$CFrcSD,
                        NClr = cldfrm1$NClr, NOvc = cldfrm1$NOvc, PSfc = psfc)
    frmexp$ScnRow = scrw
    if (k == 1) {
      frmout = frmexp
    }
    else {
      frmout = rbind(frmout,frmexp)
    }
  }
  return(frmout)

}

gmm_cond_mom = function(gmmobj, trumat, retmat, outfile = NULL) {
    # Assemble GMM conditional moments for true state elements given retrieved
    # gmmobj:  A densityMClust object with the joint GMM estimates
    # trumat:  Matrix/vector of true states
    # retmat:  Matrix of retrieved quantities
    # outfile: File name for output (HDF5)

    K = ncol(gmmobj$z)   # number of modes in the mixture
    N = nrow(retmat)     # number of obs
    dx1 = N; dx2 = ncol(trumat) # dimensions of required matrices
    dy1 = N; dy2 = ncol(retmat)
    print(dx2)
    xsq = seq(1,dx2)
    ysq = seq(dx2+1,dx2+dy2)

    # Set up arrays to hold parameters.
    SigmaXX = array(0,c(dx2,dx2,K)) # 3-dimensional array [xdim, xdim, component]
    SigmaXY = array(0,c(dx2,dy2,K)) # 3-dimensional array [xdim, ydim, component]
    SigmaYY = array(0,c(dy2,dy2,K)) # 3-dimensional array [ydim, ydim, component]
    SigmaYYinv = array(0,c(dy2,dy2,K)) # 3-dimensional array [ydim, ydim, component]
    xbar = array(NA, c(dx2,K)) # 2-dimensional array [component, xdim]
    ybar = array(NA, c(dy2,K)) # 2-dimensional array [component, ydim]
    # Cxpst will hold the cov(X|Y) for each component, k=1,2,...,K
    Cxpst = array(NA, c(dx2,dx2,K)) # 3-dimensional array [component, xdim, xdim]

    for (k in 1:K) {
      print(paste("k = ", k, sep = ""),quote=F)

      # Use fitted GMM mean and cov estimates (alternative is weighted empirical estimates)
      # Need to compute mu_k and sigma_k on the scale of X and Y.
      # densityMclust returns them on the scale of pcaX$x[,1:nPCX]
      # and pcaY$x[,1:nPCY]. Mclust computes its component means
      # and variances from weighted observations with weights
      # given by d$z.

      xbar[,k] = gmmobj$parameters$mean[xsq,k]
      ybar[,k] = gmmobj$parameters$mean[ysq,k]

      SigmaXX[,,k] = gmmobj$parameters$variance$sigma[xsq,xsq,k]
      SigmaXY[,,k] = gmmobj$parameters$variance$sigma[xsq,ysq,k]
      SigmaYY[,,k] = gmmobj$parameters$variance$sigma[ysq,ysq,k]
      if (is.matrix(SigmaYY[,,k])) {
        SigmaYYinv[,,k] = matrix.inverse(SigmaYY[,,k])
      } else {
        SigmaYYinv[,,k] = matrix.inverse(as.matrix(SigmaYY[,,k]))
      }


      # The density value of the nth datum under mixture component k is the multi-
      # variate normal density value for mean ybar[k,] and covariance matrix
      # SigmaYY[k,,].
      #for (n in 1:N) {
      #  f_y__c[n,k] = dmvnorm(Y[n,], mean=ybar[k,], sigma=as.matrix(SigmaYY[k,,]))
      #}

      # C[k,,] is the conditional (posterior) covariance for component k.
      # Only needs to be computed once (since it is the same no matter which
      # Yn), so do it here.
      if (dx2 == 1) {
        Cxpst[,,k] = SigmaXX[,,k] - ((t(SigmaXY[,,k]) %*% SigmaYYinv[,,k]) %*% SigmaXY[,,k])
      } else {
        Cxpst[,,k] = SigmaXX[,,k] - ((SigmaXY[,,k] %*% SigmaYYinv[,,k]) %*% t(SigmaXY[,,k]))
      }
    }

    # Save results
    uqModelParams = list(prob=gmmobj$parameters$pro,ybar=ybar,xbar=xbar,SigmaXX=SigmaXX,
                         SigmaXY=SigmaXY,SigmaYY=SigmaYY,SigmaYYinv=SigmaYYinv,Cxpst=Cxpst)

    if (!(is.null(outfile))) {
        stnmsx = unlist(dimnames(trumat)[2])
        stnmsy = unlist(dimnames(retmat)[2])

        # Write to NetCDF if provided
        dimK = ncdim_def("component",units="",vals = seq(1,K),create_dimvar = FALSE)
        dimchr = ncdim_def("charnm",units="",vals = seq(1,30),create_dimvar = FALSE)
        dimstx = ncdim_def("state_true",units="",vals = seq(1,dx2),create_dimvar = FALSE)
        dimsty = ncdim_def("state_retrieved",units="",vals = seq(1,dy2),create_dimvar = FALSE)

        varmux = ncvar_def("mean_true",units="",dim=list(dimstx,dimK), -9999, prec="double",
                           longname = "Component means for true state sub-vector")
        varmuy = ncvar_def("mean_retrieved",units="",dim=list(dimsty,dimK), -9999, prec="double",
                           longname = "Component means for retrieved state sub-vector")
        varprb = ncvar_def("mixture_proportion", units="",dim=list(dimK), -9999, prec="double",
                           longname = "Component mixture proportions")
        varcvx = ncvar_def("varcov_true",units="",dim=list(dimstx,dimstx,dimK), -9999, prec="double",
                           longname = "Component covariance matrix for true state sub-vector")
        varcvxy = ncvar_def("varcov_cross",units="",dim=list(dimstx,dimsty,dimK), -9999, prec="double",
                           longname = "Component cross-covariance matrix")
        varcvy = ncvar_def("varcov_retrieved",units="",dim=list(dimsty,dimsty,dimK), -9999, prec="double",
                           longname = "Component covariance matrix for retrieved state sub-vector")
        varprcy = ncvar_def("precmat_retrieved",units="",dim=list(dimsty,dimsty,dimK), -9999, prec="double",
                            longname = "Component inverse covariance matrix for retrieved state sub-vector")
        varpstcvx = ncvar_def("varcov_post_true",units="",dim=list(dimstx,dimstx,dimK), -9999, prec="double",
                           longname = "Component posterior/conditional covariance matrix for true state sub-vector")
        varnmx = ncvar_def("state_names_true",units="",dim=list(dimchr,dimstx),NULL,prec="char",
                            longname = "Names of true state variables")
        varnmy = ncvar_def("state_names_retrieved",units="",dim=list(dimchr,dimsty),NULL,prec="char",
                           longname = "Names of retrieved state variables")

        ncout = nc_create(outfile, vars=list(varmux, varmuy, varprb, varcvx, varcvxy, varcvy,
                                             varprcy, varpstcvx, varnmx, varnmy), force_v4 = TRUE )

        ncvar_put(ncout,varprb,vals = gmmobj$parameters$pro)
        ncvar_put(ncout,varmux,vals = xbar)
        ncvar_put(ncout,varmuy,vals = ybar)
        ncvar_put(ncout,varcvx,vals = SigmaXX)
        ncvar_put(ncout,varcvxy,vals = SigmaXY)
        ncvar_put(ncout,varcvy,vals = SigmaYY)
        ncvar_put(ncout,varprcy,vals = SigmaYYinv)
        ncvar_put(ncout,varpstcvx,vals = Cxpst)
        ncvar_put(ncout,varnmx,vals = stnmsx)
        ncvar_put(ncout,varnmy,vals = stnmsy)
        nc_close(ncout)

    }

    return(uqModelParams)


}

gmm_post_pred = function(mixModParms,retmat) {
  # Construct posterior mixture densities
  # mixModParms:  A list of GMM parameters
  #               list(prob, ybar, xbar, SigmaXX,
  #                    SigmaXY, SigmaYY, SigmaYYinv, Cxpst)
  # retmat:       Matrix of retrieved quantities (predictors)
  # Note that the mixture component dimension is intended to by the last dimension in relevant arrays

  N = nrow(retmat)     # number of obs
  K = dim(mixModParms$ybar)[2]
  dy1 = N; dy2 = ncol(retmat)
  dx2 = dim(mixModParms$xbar)[1]

  # Densities
  f_y__c = matrix(NA,N,K)
  p_c__y = matrix(NA,N,K)

  print('computing f_y__c',quote=F)
  for (k in 1:K) {
    if (dy2 > 1) {
         f_y__c[,k] = dmvnorm(retmat, mean=mixModParms$ybar[,k], sigma=mixModParms$SigmaYY[,,k],log=TRUE)
    } else if (dy2 == 1) {
           f_y__c[,k] = dnorm(as.vector(retmat), mean=as.numeric(mixModParms$ybar[k]),
                              sd=sqrt(as.numeric(mixModParms$SigmaYY[k])),log=TRUE)
    }
  }

  # Adjust for possible underflow
  mxdns = apply(f_y__c,1,FUN = max,na.rm = TRUE)
  mxarr = array(rep(mxdns,K),c(N,K))
  adjdns = f_y__c - mxarr

  # Compute the conditional probabilities, p_c__y
  print('computing p_c__y',quote=F)
  prprep = array(rep(mixModParms$prob,each=N),c(N,K))
  cmplk = prprep * exp(adjdns)
  sumlk = apply(cmplk,1,FUN = sum)
  sumrep = array(rep(sumlk,K),c(N,K))
  cmpprb = cmplk / sumrep

  print('predicting E_X__Y',quote=F)
  E_X__Y_C = array(0,c(N,dx2,K))
  E_X__Y = array(0,c(N,dx2))

  for (k in 1:K) {
    muxrp = matrix(rep(mixModParms$xbar[,k],each=N),nrow=N,ncol=dx2)
    muyrp = matrix(rep(mixModParms$ybar[,k],each=N),nrow=N,ncol=dy2)
    #print(dim(muxrp))
    #print(dim(mixModParms$SigmaXY[,,k]))
    #print(dim(mixModParms$SigmaYY[,,k]))
    E_X__Y_C[,,k] = muxrp + t(mixModParms$SigmaXY[,,k] %*% mixModParms$SigmaYYinv[,,k] %*% t(retmat - muyrp))
  }
  for (j in 1:dx2) {
    cmpmns = cmpprb * E_X__Y_C[,j,]
    E_X__Y[,j] = apply(cmpmns,1,sum)
  }

  print('predicting Sigma_X__Y',quote=F)
  Sigma_X__Y_C_bet = array(0,c(N,dx2,dx2,K))
  Sigma_X__Y_C_wth = array(0,c(N,dx2,dx2,K))
  Sigma_X__Y = array(0,c(N,dx2,dx2))
  for (k in 1:K) {
    wthcv = mixModParms$Cxpst[,,k]
    mndv = E_X__Y_C[,,k] - E_X__Y
    prbrp = array(rep(cmpprb[,k],dx2*dx2),c(N,dx2,dx2))
    Sigma_X__Y_C_wth[,,,k] = array(rep(wthcv,each=N),c(N,dx2,dx2))
    Sigma_X__Y_C_bet[,,,k] = apply(mndv,1,FUN = vec_outer)
    Sigma_X__Y = Sigma_X__Y + prbrp * (Sigma_X__Y_C_wth[,,,k] + Sigma_X__Y_C_bet[,,,k])
  }

  # Save results
  predinfo = list(condprob=cmpprb,postmean=E_X__Y,postcov=Sigma_X__Y)
  return(predinfo)
}

gmm_mix_sample = function(mixprop,mixmean,mixvar,nsz = 1000) {
    # Quick GMM samples
    ncmp = length(mixprop)
    cmpsmp = sample(1:ncmp,size=nsz,replace = TRUE,prob=mixprop)
    mus = mixmean[cmpsmp]
    sds = sqrt(mixvar[cmpsmp])
    ys = rnorm(nsz,mean=mus,sd=sds)
}
