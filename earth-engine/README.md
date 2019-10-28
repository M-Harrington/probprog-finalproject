## Folder to place Earth Engine Files

Structure roughly assumed to be one or a couple self contained files (can run on their own) main files for 
running the heavy part of the analysis.  Smaller files (module-like) for testing performing other things. 

Note: seasons correspond to 

* January (PMR): Dec, Jan
* April/ May (PRM): Feb, Mar, Apr, May
* August (MO):  June, July, August
* November (PMK): Sept, Nov

#### TODO:
- ~~Import well data~~
- ~~Define ROI~~
- For updating, fix some old code to be compatible with Landsat 5 instead of Landsat 7
- Bring in Landsat 5 EVI and NDWI layers 
- Add convolutions
- Update classification over larger number of time periods (on only the smaller area)
- Export Classified pixels
- Build Pyro Model
