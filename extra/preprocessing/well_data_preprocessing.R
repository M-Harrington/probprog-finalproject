##Pretty similar to preprocess_well, but adapted for prob programming final project

# Check if packages are installed, installs and loads them if necessary
if (!require("pacman")) install.packages("pacman")
pacman::p_load(sf, tidyverse, gstat, sp,magrittr)


#load and fix names
data.wells <- read.csv("C://users/matttt/Dropbox/India Water (Meir, Ryan, Jacob, Johannes)/Data/water_level/data_season.csv")
data.wells <- tbl_df(data.wells)
names(data.wells) <- tolower(names(data.wells))
colnames(data.wells)[4] <- "pomr"                 #Post Monsoon r.
colnames(data.wells)[5] <- "prm"                  #Pre Monsoon
colnames(data.wells)[6] <- "mo"                   #Monsoon
colnames(data.wells)[7] <- "pomk"                 #Post Monsoon k.

### change to wide format, create a variable time for easy sorting (lazy way)
data.well.long <- gather(data.wells, season, well.level, one_of('pomr', 'prm', 'mo', 'pomk'))

time_func <- function(season){
  if (is.na(season)){
    return(NA)}
  else if (season == 'prm'){
    return(2)
  }
  else if (season == 'mo'){
    return(3)
  }
  else if (season == 'pomk'){
    return(4)
  }
  else if(season == 'pomr'){
    return(1)
  }
}
data.well.long <- data.well.long %>% mutate(season.int = 0)
data.well.long[,'season.int'] <- data.well.long[,'season'] %>% apply(1, time_func)

### Sort by object
data.well.long <- data.well.long %>% arrange(objectid_well, year_obs, season.int)

# create more specific time variable, 1-96 for obs that are 1994-2016
data.well.long <- data.well.long %>% mutate(time = (year_obs-1994)*4 + season.int)

# #drop all na's (this wasn't good, still checking for potential errors, look for flags)
# data.well.long <- data.well.long %>% filter(!is.na(well.level))

# get rid of unneeded variables
data.well.long <- data.well.long %>% select(-one_of("shape_area", "shape_len", "area_re", "aq.count","pcnt.aquifer",
                                                    "pa_order", "state", "newcode14", "newcode43", "season.int",
                                                    "district_cgwb", "block_cgwb", "block_name","site_name","teh_name",
                                                    "hydro_loc", "cgwb_code"))


#select only wells within roi
data.well.long2<- data.well.long[(data.well.long$lat <= 26.77769880073871) &(data.well.long$lat>=25.65411556477628) & (data.well.long$lon <=71.87240405627415) & (data.well.long$lon >= 71.18575854846165),]

data.well.long2 %<>% select(objectid_well,year_obs,lat,lon, district,season,well.level,time)

# save file
setwd("C:\\Users\\Matttt\\Documents\\Python Scripts")
write.csv(data.well.long2,"pp_well_data.csv")
