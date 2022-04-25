# airs_uq
Supporting software for simulation based uncertainty quantification for the AIRS Level 2 algorithm

The AIRS Monte Carlo simulation experiments are carried out in several steps. Supporting software is
linked where available. Simulations are defined for a *geophysical template*, a specific set of atmospheric and
observing conditions. For AIRS, a template is typically referenced by dates/season, region of interest, and
time of day.

**Pre-processing (merra\_access)**

* Assemble reference data from MERRA2 reanalysis  
MERRA2 variables include vertical profiles of temperature and humidity, along with cloud states.
* Augment reanalysis with two-slab cloud properties  
Cloud state information is processed through a cloud slab identification algorithm. The methodology
and corresponding forward model are based on the two-slab SARTA model of [DeSouza-Machado et al.,
2018](https://doi.org/10.5194/amt-11-529-2018)
* Match MERRA2 (coarse resolution) locations to AIRS retrievals (FOV, fine resolution) for cloud
fraction information

**Template setup (airs\_uq)**

Scripts utilize subroutines from the [calculate\_VPD](lib/calculate_VPD.py) and [quantile\_airs](lib/quantile_airs.py) modules as
well as R support routines in [airs\_post\_expt\_support.R](lib/airs_post_expt_support.R)

***

### Additional AIRS Documentation

Simulation experiments use Version 6 of the AIRS retrieval algorithm

* [Algorithm Theoretical Basis Document](https://docserver.gesdisc.eosdis.nasa.gov/public/project/AIRS/L2_ATBD.pdf)
* [Data User's Guide](https://docserver.gesdisc.eosdis.nasa.gov/public/project/AIRS/V7_L2_Product_User_Guide.pdf)
* [AIRS Levels, Layers, and Trapezoids](https://docserver.gesdisc.eosdis.nasa.gov/repository/Mission/AIRS/3.3_ScienceDataProductDocumentation/3.3.4_ProductGenerationAlgorithms/V6_L2_Levels_Layers_Trapezoids.pdf)
