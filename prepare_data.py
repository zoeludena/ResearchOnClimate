from siphon import catalog
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
import os.path
overwrite = False

model = 'NorESM2-LM'
experiments = [
               '1pctCO2', 'abrupt-4xCO2', 'historical', 'piControl', # CMIP
               'hist-GHG', 'hist-aer', # DAMIP
               'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF', 'ssp585' #	ScenarioMIP
]
variables = [
             'tas', 'tasmin', 'tasmax', 'pr'
]

# Cache the full catalogue from NorESG
full_catalog = catalog.TDSCatalog('http://noresg.nird.sigma2.no/thredds/catalog/esgcet/catalog.xml')
print("Read full catalogue")

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

def get_data_daily(variable, experiment, member):
    """
    Read a particular CMIP6 (Amon) variable from NorESM2
    """
    import glob
    files = glob.glob(f"/glade/collections/cmip/CMIP6/{get_MIP(experiment)}/NCC/NorESM2-LM/{experiment}/{member}/day/{variable}/gn/v20190815/{variable}/*.nc")
    return xr.open_mfdataset(files)[variable]

def get_data(variable, experiment, member):
    """
    Read a particular CMIP6 (Amon) variable from NorESM2
    """
    import glob
    files = glob.glob(f"/glade/collections/cmip/CMIP6/{get_MIP(experiment)}/NCC/NorESM2-LM/{experiment}/{member}/Amon/{variable}/gn/v20190815/{variable}/*.nc")
    return xr.open_mfdataset(files)[variable]

#Loop over experiments and members creating one (annual mean) file with all variables in for each one
for experiment in experiments:
    for i in range(3):
        physics = 2 if experiment == 'ssp245-covid' else 1  # The COVID simulation uses a different physics setup
          
        # TODO - check the differences...
        member = f"r{i+1}i1p1f{physics}"
        print(f"Processing {member} of {experiment}...")
        outfile = f"{model}_{experiment}_{member}.nc"
        if (not overwrite) and os.path.isfile(outfile):
            print("File already exists, skipping.")
            continue
    
        tas_month = False
        tas_day = False
        pr_month = False
        pr_day = False
        tasmin_day = False
        tasmax_day = False
    
        try:
            tasmin = get_data_daily('tasmin', experiment, member)
            tasmin_day = True
        except OSError:
            print("Skipping this realisation as no data present tasmin")
        try:
            tasmax = get_data_daily('tasmax', experiment, member)
            tasmax_day = True
        except OSError:
            print("Skipping this realisation as no data present tasmax")
        try:
            tas = get_data('tas', experiment, member)
            tas_month = True
        except OSError:
            print("Skipping this realisation as no data present tas")
        try:
            pr = get_data('pr', experiment, member).persist()  # Since we need to process it twice
            pr_month = True
        except OSError:
            print("Skipping this realisation as no data present pr")
    
        if not tas_month:
            try:
                tas = get_data_daily('tas', experiment, member)
                tas_day = True
            except OSError:
                print("Skipping this realisation as no data present tas day")
        if not pr_month:
            try:
                pr = get_data_daily('pr', experiment, member).persist()  # Since we need to process it twice
                pr_day = True
            except OSError:
                print("Skipping this realisation as no data present pr day")
    
        if pr_month or pr_day:
            pr = pr.chunk(dict(time=-1))  # Rechunk time into a single chunk
    
        # If all of the elements are present
        if tasmin_day and tasmax_day:
            if pr_day or pr_month:
                if tas_day or tas_month:
                    # Derive additional vars
                    dtr = tasmax-tasmin
                    ds = xr.Dataset({'diurnal_temperature_range': dtr.groupby('time.year').mean('time'),
                             'tas': tas.groupby('time.year').mean('time'),
                             'pr': pr.groupby('time.year').mean('time'),
                             'pr90': pr.groupby('time.year').quantile(0.9, skipna=True)})
                    ds.to_netcdf(f"{model}_{experiment}_{member}.nc")
                    
        # If only pr, tasmin, and tasmax are present
        if not tas_day or not tas_month:
            if pr_day or pr_month:
                if tasmin_day and tasmax_day:
                    dtr = tasmax-tasmin
                    ds = xr.Dataset({'diurnal_temperature_range': dtr.groupby('time.year').mean('time'),
                             'pr': pr.groupby('time.year').mean('time'),
                             'pr90': pr.groupby('time.year').quantile(0.9, skipna=True)})
                    ds.to_netcdf(f"{model}_{experiment}_{member}_no_tas.nc")
                    
        #If only tas, tasmin, and tasmax are present
        if not pr_day or not pr_month:
            if tas_day or tas_month:
                if tasmin_day and tasmax_day:
                    dtr = tasmax-tasmin
                    ds = xr.Dataset({
                        'diurnal_temperature_range': dtr.groupby('time.year').mean('time'),
                        'tas': tas.groupby('time.year').mean('time')})
                    ds.to_netcdf(f"{model}_{experiment}_{member}_no_pr.nc")
    
        # If only pr and tas are present
        if pr_day or pr_month:
            if tas_day or tas_month:
                ds = xr.Dataset({
                     'tas': tas.groupby('time.year').mean('time'),
                     'pr': pr.groupby('time.year').mean('time'),
                     'pr90': pr.groupby('time.year').quantile(0.9, skipna=True)})
                ds.to_netcdf(f"{model}_{experiment}_{member}_no_tasmin_max.nc")