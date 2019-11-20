import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt

import pickle
import os

path = []
evi_datasets = []
date_list = []
for file_name in os.listdir("evi_layers/"):
    path = "evi_layers/" + file_name

    evi_tmp = xr.open_rasterio(path, parse_coordinates=None)
    date = file_name.split(".")[0].split("k")[1]
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    date_list.append(date)

    evi_datasets.append(evi_tmp.assign_coords(date=date).sel(band=1))

evi_dataset = xr.concat(evi_datasets, dim="date")
xcoord = evi_dataset.isel(date=[0])
ycoord = evi_dataset.isel(date=[0]).y.values


c_list = [(a, b) for a in xcoord for b in ycoord]
evi_coords = np.array(c_list)

wells = pd.read_csv("pp_well_data.csv", index_col=0)
wells = wells[wells["well.level"].notna()]
wells["well.level"] = wells["well.level"] * -1


def timefunc(well_time):
    """Well Time is a row from the wells matrix with season and year for simplicity
    returns the translated time"""
    if well_time["season"] == "pomr":
        img_time = "11-15"
    elif well_time["season"] == "prm":
        img_time = "01-15"
    elif well_time["season"] == "mo":
        img_time = "05-15"
    else:
        img_time = "08-15"

    img_time = dt.datetime.strptime(
        str(well_time["year_obs"]) + "-" + img_time, "%Y-%m-%d"
    )
    return img_time


wells["time_long"] = wells.apply(timefunc, axis=1)

wells = wells[wells.time_long.isin(evi_dataset.date.values)]
wells_x = wells.loc[:, ["lat", "lon", "time_long"]]
wells_y = wells.loc[:, ["well.level", "time_long"]]

g = wells_x.groupby("time_long").cumcount()
L = (
    wells_x.set_index(["time_long", g])
    .unstack(fill_value=0)
    .stack()
    .groupby(level=0)
    .apply(lambda x: x.values.tolist())
    .tolist()
)

x_wells = [None] * len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    x_wells[i] = temp

g = wells_y.groupby("time_long").cumcount()
L = (
    wells_y.set_index(["time_long", g])
    .unstack(fill_value=0)
    .stack()
    .groupby(level=0)
    .apply(lambda x: x.values.tolist())
    .tolist()
)

y_wells = [None] * len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    y_wells[i] = temp

da_nans = evi_dataset.where(evi_dataset > 0)

da_stacked = da_nans.stack(dim=["y", "x", "date"])
nona_coords = da_stacked[da_stacked.notnull()].dim

nona_coords_pd = pd.DataFrame(nona_coords.values)
farm_coords = pd.DataFrame(nona_coords_pd.iloc[:, 0].tolist())
farm_coords.rename(columns={0: "y", 1: "x", 2: "date"}, inplace=True)

farm_coords["evi_value"] = da_stacked[da_stacked.notnull()]
farm_coords.head()

farm_x = farm_coords.loc[:, ["y", "x", "date"]]
farm_y = farm_coords.loc[:, ["evi_value", "date"]]

g = farm_x.groupby("date").cumcount()
L = (
    farm_x.set_index(["date", g])
    .unstack(fill_value=0)
    .stack()
    .groupby(level=0)
    .apply(lambda x: x.values.tolist())
    .tolist()
)

x_farm = [None] * len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    x_farm[i] = temp

g = farm_y.groupby("date").cumcount()
L = (
    farm_y.set_index(["date", g])
    .unstack(fill_value=0)
    .stack()
    .groupby(level=0)
    .apply(lambda x: x.values.tolist())
    .tolist()
)

y_farm = [None] * len(L)
for i in range(len(L)):
    temp = []
    for j in range(len(L[0])):
        if L[i][j][0] != 0:
            temp.append(L[i][j])

    y_farm[i] = temp

os.chdir("C://Users/Matttt/Documents/probprog-finalproject/preprocessing")

file = open("dataset.pkl", "wb")

pickle.dump(x_farm, file)
pickle.dump(y_farm, file)
pickle.dump(x_wells, file)
pickle.dump(y_wells, file)

file.close()


def date_func(date):
    if date.month == 11:
        season = 1
    if date.month == 1:
        season = 2
    if date.month == 8:
        season = 3
    return season


date_list_season = []
for i in range(len(date_list)):
    date_list_season.append(date_func(date_list[i]))

date_list_season
