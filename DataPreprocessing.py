'''# Well and EVI preprocessing
Taking evi rasters and put them into one dataset, meanwhile compute distances between wells and pixels'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

import os
from scipy.spatial.distance import cdist

os.chdir("C:\\Users\\Matttt\\Documents\\Python Scripts\\ProbProg-ExtraScripts")
path=[]; evi_datasets = []; date_list=[]
# concatenate evi rasters
for file_name in os.listdir("evi_layers/"):
    path = "evi_layers/"+file_name

    #open and parse names/dates
    evi_tmp= xr.open_rasterio(path, parse_coordinates=None)
    date= file_name.split('.')[0].split("k")[1]
    date=dt.datetime.strptime(date,'%Y-%m-%d')
    date_list.append(date)


    evi_datasets.append(evi_tmp.assign_coords(date=date).sel(band=1))

evi_dataset=xr.concat(evi_datasets,dim='date')
xcoord= evi_dataset.isel(date=[0]).x.values  #optionally for testing x=[0,1,2,3],y=[0,1,2,3],
ycoord= evi_dataset.isel(date=[0]).y.values


c_list=[(a,b) for a in xcoord for b in ycoord]
evi_coords = np.array(c_list)

#load, remove nas
wells= pd.read_csv("pp_well_data.csv",index_col=0)
wells= wells[wells['well.level'].notna()]
wells['well.level'] = wells['well.level']*-1

#need wells to have same time dimensions as image files
#pomk= 08-15,11-15; pomr= 11-15,01-15; prm= 01-15,05-15; mo= 05-15,08-15;

def timefunc(well_time):
    '''Well Time is a row from the wells matrix with season and year for simplicity
    returns the translated time'''
    if well_time['season'] == "pomr":
        img_time = "11-15"
    elif well_time['season'] == "prm":
        img_time = "01-15"
    elif well_time['season'] == "mo":
        img_time = "05-15"
    else:
        img_time= "08-15"

    img_time=dt.datetime.strptime(str(well_time['year_obs'])+"-"+img_time, '%Y-%m-%d')
    return img_time

wells['time_long']= wells.apply(timefunc, axis=1)

#Match up time dimensions
wells=wells[wells.time_long.isin(evi_dataset.date.values)]
wells_x=wells.loc[:,['lat','lon','time_long']]
wells_y=wells.loc[:,['well.level','time_long']]

#chop up dataframe into appropriate tensor shape (probably would have been better to just reshape using xarray)
g = wells_x.groupby('time_long').cumcount()
L = (wells_x.set_index(['time_long',g])
       .unstack(fill_value=0)
       .stack().groupby(level=0)
       .apply(lambda x: x.values.tolist())
       .tolist())

#remove zero entries
x_wells =[None]*len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    x_wells[i]=temp

##Same thing for y_wells
#chop up dataframe into appropriate tensor shape
g = wells_y.groupby('time_long').cumcount()
L = (wells_y.set_index(['time_long',g])
       .unstack(fill_value=0)
       .stack().groupby(level=0)
       .apply(lambda x: x.values.tolist())
       .tolist())

#remove zero entries
y_wells =[None]*len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    y_wells[i]=temp

da_nans=evi_dataset.where(evi_dataset>0)

#create coordinate database
da_stacked=da_nans.stack(dim=['y','x','date'])
nona_coords = da_stacked[da_stacked.notnull()].dim

#Change to pandas dataframe
nona_coords_pd = pd.DataFrame(nona_coords.values)
farm_coords = pd.DataFrame(nona_coords_pd.iloc[:,0].tolist())
farm_coords.rename(columns={0:"y", 1:"x", 2:"date"}, inplace=True)

farm_coords['evi_value'] = da_stacked[da_stacked.notnull()]
farm_coords.head()

### Farm dataset
farm_x=farm_coords.loc[:,['y','x','date']]
farm_y=farm_coords.loc[:,['evi_value','date']]

#chop up dataframe into appropriate tensor shape
g = farm_x.groupby('date').cumcount()
L = (farm_x.set_index(['date',g])
       .unstack(fill_value=0)
       .stack().groupby(level=0)
       .apply(lambda x: x.values.tolist())
       .tolist())

#remove zero entries
x_farm =[None]*len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    x_farm[i]=temp

##Same thing for y_wells
#chop up dataframe into appropriate tensor shape
g = farm_y.groupby('date').cumcount()
L = (farm_y.set_index(['date',g])
       .unstack(fill_value=0)
       .stack().groupby(level=0)
       .apply(lambda x: x.values.tolist())
       .tolist())

#remove zero entries
y_farm =[None]*len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    y_farm[i]=temp

'''### Export to file using pickle
Note the order of lists for unpacking'''

import pickle
os.chdir("C://Users/Matttt/Documents/probprog-finalproject/preprocessing")

#Save to file
file = open('dataset.pkl','wb')

pickle.dump(x_farm, file)
pickle.dump(y_farm, file)
pickle.dump(x_wells, file)
pickle.dump(y_wells, file)

file.close()

def date_func(date):
    if date.month== 11:
        season= 1
    if date.month== 1:
        season= 2
    if date.month== 8:
        season= 3
    return season

date_list_season=[]
for i in range(len(date_list)):
    date_list_season.append(date_func(date_list[i]))

date_list_season
