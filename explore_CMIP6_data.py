import pandas as pd
import xarray as xr
import numpy as np
import os.path

import matplotlib.pyplot as plt
# Useful for plotting maps
import cartopy.crs as ccrs

# This can be useful for working with multiple processors - to be explored later on
# from dask.distributed import Client, LocalCluster

def get_MIP(experiment):
    """
    Utility function to get teh activity associated with a particular experiment
    """
    if experiment == 'ssp245-covid':
        return 'DAMIP'
    elif experiment == 'ssp370-lowNTCF':
        return 'AerChemMIP'
    elif experiment.startswith('ssp'):
        return 'ScenarioMIP'
    elif experiment.startswith('hist-'):
        return 'DAMIP'
    else:
        return 'CMIP'

def get_data(variable, experiment, member):
    """
    Read a particular CMIP6 (Amon) variable from NorESM2
    """
    import glob
    files = glob.glob(f"/glade/collections/cmip/CMIP6/{get_MIP(experiment)}/NCC/NorESM2-LM/{experiment}/{member}/Amon/{variable}/gn/v20190815/{variable}/*.nc")
    return xr.open_mfdataset(files)[variable]

def get_data_daily(variable, experiment, member):
    """
    Read a particular CMIP6 (Amon) variable from NorESM2
    """
    import glob
    files = glob.glob(f"/glade/collections/cmip/CMIP6/{get_MIP(experiment)}/NCC/NorESM2-LM/{experiment}/{member}/day/{variable}/gn/v20190815/{variable}/*.nc")
    return xr.open_mfdataset(files)[variable]

tas = get_data('tas', 'historical', 'r1i1p1f1')

# Plot a map of the average temperature between 1850-1900

tas.sel(time=slice('1850','1900')).mean('time').plot(
    transform=ccrs.PlateCarree(), # This is the projection the data is stored as
    subplot_kws={"projection": ccrs.PlateCarree()}, # This describes the projection to plot onto (which happens to be the projection the data is already in so no transformation is needed in this case)
)

# Feel free to explore other projections here: https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html

plt.title("Average Global Temp Between 1850-1900")
plt.gca().coastlines()

tas.sel(time=slice('2005','2015')).mean('time').plot(
    transform=ccrs.PlateCarree(), # This is the projection the data is stored as
    subplot_kws={"projection": ccrs.PlateCarree()}, # This describes the projection to plot onto (which happens to be the projection the data is already in so no transformation is needed in this case)
)
plt.title("Average Global Temp Between 2005-2015")
plt.gca().coastlines()

difference = tas.sel(time=slice('1850', '1900')).mean('time') - tas.sel(time=slice('2005', '2015')).mean('time')

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
c = difference.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal'}  # Make the color bar horizontal
)
plt.title("Difference in Average Global Temp From 1850-1900 to 2005-2015")
ax.coastlines()
plt.subplots_adjust(bottom=0.15)  # Adjust as needed
plt.show()

# When averaging gridded data on a sphere, we need to account for the fact that
    #the values near the poles have less area
weights = np.cos(np.deg2rad(tas.lat))
weights.name = "weights"

tas_timeseries = tas.weighted(weights).mean(['lat', 'lon'])

tas_timeseries.plot()

# Plot maps of precipitation

pr = get_data('pr', 'historical', 'r1i1p1f1')

pr.sel(time=slice('1850','1900')).mean('time').plot(
    transform=ccrs.PlateCarree(),
    subplot_kws={"projection": ccrs.PlateCarree()},
)
plt.title("Average Precipitation Between 1850-1900")
plt.gca().coastlines()
plt.show()

pr.sel(time=slice('2005','2015')).mean('time').plot(
    transform=ccrs.PlateCarree(),
    subplot_kws={"projection": ccrs.PlateCarree()},
)
plt.title("Average Precipitation Between 2005-2015")
plt.gca().coastlines()
plt.show()

difference = pr.sel(time=slice('1850', '1900')).mean('time') - pr.sel(time=slice('2005', '2015')).mean('time')

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
c = difference.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal'}  # Make the color bar horizontal
)
plt.title("Difference in Average Precipitation From 1850-1900 to 2005-2015")
ax.coastlines()
plt.subplots_adjust(bottom=0.15)  # Adjust as needed
plt.show()

weights = np.cos(np.deg2rad(pr.lat))
weights.name = "weights"

pr_timeseries = pr.weighted(weights).mean(['lat', 'lon'])

pr_timeseries.plot()
plt.title("Precipitation Timeseries")
plt.show()

# Plot maps of Daily Maximum Temperature

tasmax = get_data_daily('tasmax', 'historical', 'r1i1p1f1')

tasmax.sel(time=slice('1850','1900')).mean('time').plot(
    transform=ccrs.PlateCarree(),
    subplot_kws={"projection": ccrs.PlateCarree()},
)
plt.title("Average Daily Max Temp Between 1850-1900")
plt.gca().coastlines()
plt.show()

tasmax.sel(time=slice('2005','2015')).mean('time').plot(
    transform=ccrs.PlateCarree(),
    subplot_kws={"projection": ccrs.PlateCarree()},
)
plt.title("Average Daily Max Temp Between 2005-2015")
plt.gca().coastlines()
plt.show()

weights = np.cos(np.deg2rad(tasmax.lat))
weights.name = "weights"

tasmax_timeseries = tasmax.weighted(weights).mean(['lat', 'lon'])

tasmax_timeseries.plot()
plt.title("Daily Maximum Temperature Timeseries")
plt.show()

# Plot maps of Surface Downwelling Shortwave Radiation

rsds = get_data('rsds', 'historical', 'r1i1p1f1')

rsds_1850_1900 = rsds.sel(time=slice('1850', '1900')).mean('time')

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
c = rsds_1850_1900.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal'}  # Make the color bar horizontal
)
plt.title("Average Surface Downwelling Shortwave Radiation Between 1850-1900")
ax.coastlines()
plt.subplots_adjust(bottom=0.15)  # Adjust as needed
output_filename = "avg_rsds_1850_1900.png"
plt.savefig(f"{output_folder}/{output_filename}")
plt.show()

rsds_2005_2005 = rsds.sel(time=slice('2005', '2015')).mean('time')

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
c = rsds_2005_2005.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal'}  # Make the color bar horizontal
)
plt.title("Average Surface Downwelling Shortwave Radiation Between 2005-2015")
ax.coastlines()
plt.subplots_adjust(bottom=0.15)  # Adjust as needed
output_filename = "avg_rsds_2005_2015.png"
plt.savefig(f"{output_folder}/{output_filename}")
plt.show()

difference = rsds.sel(time=slice('1850', '1900')).mean('time') - rsds.sel(time=slice('2005', '2015')).mean('time')

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
c = difference.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal'}  # Make the color bar horizontal
)
plt.title("Difference in Average Surface Downwelling\nShortwave Radiation From 1850-1900 to 2005-2015")
ax.coastlines()
plt.subplots_adjust(bottom=0.15)  # Adjust as needed
output_filename = "diff_avg_rsds.png"
plt.savefig(f"{output_folder}/{output_filename}")
plt.show()

weights = np.cos(np.deg2rad(rsds.lat))
weights.name = "weights"

rsds_timeseries = rsds.weighted(weights).mean(['lat', 'lon'])

rsds_timeseries.plot()
plt.title("Surface Downwelling Shortwave Radiation Timeseries")
output_filename = "rsds_timeseries.png"
plt.savefig(f"{output_folder}/{output_filename}")
plt.show()