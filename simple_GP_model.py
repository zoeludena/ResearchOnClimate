import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
import esem

from esem import gp_model
from esem.data_processors import Whiten, Normalise
from matplotlib import colors

import warnings
warnings.filterwarnings('ignore')

# Utilities for normalizing the emissions data
min_co2 = 0.
max_co2 = 2400 #CAN CHANGE
def normalize_co2(data):
    return data / max_co2

def un_normalize_co2(data):
    return data * max_co2

min_ch4 = 0.
max_ch4 = 0.6
def normalize_ch4(data):
    return data / max_ch4

def un_normalize_ch4(data):
    return data * max_ch4

data_path = "train_val/" #Change this to where your data is stored!

# Get one combined historical + ssp585 timeseries for now
X = xr.open_mfdataset([data_path + 'inputs_historical.nc', data_path + 'inputs_ssp585.nc']).compute()

# Take the 2nd ensemble member for the historical (the first one has some missing DTR values for some reason...) and the 1st (only) one for SSP585
Y = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').sel(member=2), xr.open_dataset(data_path + 'outputs_ssp585.nc').sel(member=1)], dim='time').compute()

# Convert the precip values to mm/day
Y["pr"] *= 86400
Y["pr90"] *= 86400

# Get the test data (NOT visible to contestants)

#Should this be the test data? Like data in the test file?
test_Y = xr.open_dataset('train_val/outputs_ssp245.nc').compute() #originally changed because we do not have 245
test_X = xr.open_dataset('train_val/inputs_ssp245.nc').compute()

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
bc_solver = Eof(X['BC'])

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
bc_eofs = bc_solver.eofsAsCorrelation(neofs=5)
bc_pcs = bc_solver.pcs(npcs=5, pcscaling=1)

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
so2_solver = Eof(X['SO2'])

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
so2_eofs = so2_solver.eofsAsCorrelation(neofs=5)
so2_pcs = so2_solver.pcs(npcs=5, pcscaling=1)

# Convert the Principle Components of the aerosol emissions (calculated above) in to Pandas DataFrames
bc_df = bc_pcs.to_dataframe().unstack('mode')
bc_df.columns = [f"BC_{i}" for i in range(5)]

so2_df = so2_pcs.to_dataframe().unstack('mode')
so2_df.columns = [f"SO2_{i}" for i in range(5)]

# Bring the emissions data back together again and normalise
leading_historical_inputs = pd.DataFrame({
    "CO2": normalize_co2(X["CO2"].data),
    "CH4": normalize_ch4(X["CH4"].data)
}, index=X["CO2"].coords['time'].data)

# Combine with aerosol EOFs
leading_historical_inputs=pd.concat([leading_historical_inputs, bc_df, so2_df], axis=1)

tas_gp = gp_model(leading_historical_inputs, Y["tas"])
tas_gp.train()

pr_gp = gp_model(leading_historical_inputs, Y["pr"])
pr_gp.train()

dtr_gp = gp_model(leading_historical_inputs, Y["diurnal_temperature_range"])
dtr_gp.train()

pr90_gp = gp_model(leading_historical_inputs, Y["pr90"])
pr90_gp.train()

# Will be hidden from contestants
tas_truth = test_Y["tas"].mean('member')
pr_truth = test_Y["pr"].mean('member') * 86400
pr90_truth = test_Y["pr90"].mean('member') * 86400
dtr_truth = test_Y["diurnal_temperature_range"].mean('member')

test_inputs = pd.DataFrame({
    "CO2": normalize_co2(test_X["CO2"].data),
    "CH4": normalize_ch4(test_X["CH4"].data)
}, index=test_X["CO2"].coords['time'].data)

# Combine with aerosol EOFs
test_inputs=pd.concat([test_inputs, 
                       bc_solver.projectField(test_X["BC"], neofs=5, eofscaling=1).to_dataframe().unstack('mode').rename(columns={i:f"BC_{i}" for i in range(5)}),
                       so2_solver.projectField(test_X["SO2"], neofs=5, eofscaling=1).to_dataframe().unstack('mode').rename(columns={i:f"_{i}" for i in range(5)}),
                       ], axis=1)

#Evaluate predictions

m_tas, _ = tas_gp.predict(test_inputs)
m_pr, _ = pr_gp.predict(test_inputs)
m_pr90, _ = pr90_gp.predict(test_inputs)
m_dtr, _ = dtr_gp.predict(test_inputs)

# tas Plot (Truth)

divnorm=colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)

with plt.style.context("dark_background"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    tas_truth.sel(time=2050).plot(cmap="coolwarm", norm=divnorm,
                                  cbar_kwargs={"label":"Temperature change / K"})
    ax.set_title("tas True 2050")
    ax.coastlines()

plt.show()

# tas Plot (Emulated)

with plt.style.context("dark_background"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    m_tas.sel(sample=35).plot(cmap="coolwarm", norm=divnorm,
                             cbar_kwargs={"label":"Temperature change / K"})
    ax.set_title("tas Emulated 2050")
    ax.coastlines()

plt.show()

# pr plot (Truth)

from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)

with plt.style.context("dark_background"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    pr_truth.sel(time=2050).plot(cmap="coolwarm", norm=divnorm,
                                  cbar_kwargs={"label":"Precipitation change"})
    ax.set_title("pr True 2050")
    ax.coastlines()

plt.show()

# pr plot (Emulated)

with plt.style.context("dark_background"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    m_pr.sel(sample=35).plot(cmap="coolwarm", norm=divnorm,
                             cbar_kwargs={"label":"Precipitation change"})
    ax.set_title("pr Emulated 2050")
    ax.coastlines()

plt.show()

# dtr plot (Truth)

from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)

with plt.style.context("dark_background"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    dtr_truth.sel(time=2050).plot(cmap="coolwarm", norm=divnorm,
                                  cbar_kwargs={"label":"Diurnal Temperature Range change / K"})
    ax.set_title("dtr True 2050")
    ax.coastlines()

plt.show()

# dtr plot (Emulated)

with plt.style.context("dark_background"):
    ax = plt.axes(projection=ccrs.PlateCarree())
    m_dtr.sel(sample=35).plot(cmap="coolwarm", norm=divnorm,
                             cbar_kwargs={"label":"Diurnal Temperature Range change / K"})
    ax.set_title("dtr Emulated 2050")
    ax.coastlines()

plt.show()

# Specific Year RMSE tas between truth and prediction for original model (default kernel)

def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth-pred)**2).weighted(weights).mean(['lat', 'lon'])).data

orig_tas_2015 = get_rmse(tas_truth[0], m_tas[0])
orig_tas_2050 = get_rmse(tas_truth[35], m_tas[35])
orig_tas_2100 = get_rmse(tas_truth[85], m_tas[85])

# dtr

orig_dtr_2015 = get_rmse(dtr_truth[0], m_dtr[0])
orig_dtr_2050 = get_rmse(dtr_truth[35], m_dtr[35])
orig_dtr_2100 = get_rmse(dtr_truth[85], m_dtr[85])

# pr

orig_pr_2015 = get_rmse(pr_truth[0], m_pr[0])
orig_pr_2050 = get_rmse(pr_truth[35], m_pr[35])
orig_pr_2100 = get_rmse(pr_truth[85], m_pr[85])

orig_pr90_2015 = get_rmse(pr90_truth[0], m_pr90[0])
orig_pr90_2050 = get_rmse(pr90_truth[35], m_pr90[35])
orig_pr90_2100 = get_rmse(pr90_truth[85], m_pr90[85])

# This function allows you to compare RMSE for different models and the original one (default kernel)

def testing(kernels, op):
    """
    kernels: A list of strings that contain the kernels to be used in the model
    op: A string that says how the kernels should be added together

    This function prints the difference of the Original's RMSE - New RMSE
    """

    # Making all of the GP models
    tas_gp = gp_model(leading_historical_inputs, Y["tas"], kernel=kernels, kernel_op = op)
    tas_gp.train()
    pr_gp = gp_model(leading_historical_inputs, Y["pr"], kernel=kernels, kernel_op = op)
    pr_gp.train()
    dtr_gp = gp_model(leading_historical_inputs, Y["diurnal_temperature_range"], kernel=kernels, kernel_op = op)
    dtr_gp.train()
    pr90_gp = gp_model(leading_historical_inputs, Y["pr90"], kernel=kernels, kernel_op = op)
    pr90_gp.train()

    # Making predictions with the models
    m_tas, _ = tas_gp.predict(test_inputs)
    m_pr, _ = pr_gp.predict(test_inputs)
    m_pr90, _ = pr90_gp.predict(test_inputs)
    m_dtr, _ = dtr_gp.predict(test_inputs)

    #Comparing the models to the original (default)
    print("tas\n")
    print(f"RMSE at 2015 Diff: {np.round(orig_tas_2015 - get_rmse(tas_truth[0], m_tas[0]), 4)}")
    print(f"RMSE at 2050 Diff: {np.round(orig_tas_2050 - get_rmse(tas_truth[35], m_tas[35]), 4)}")
    print(f"RMSE at 2100 Diff: {np.round(orig_tas_2100 - get_rmse(tas_truth[85], m_tas[85]), 4)}")

    print("\ndtr\n")
    print(f"RMSE at 2015 Diff: {np.round(orig_dtr_2015 - get_rmse(dtr_truth[0], m_dtr[0]), 4)}")
    print(f"RMSE at 2050 Diff: {np.round(orig_dtr_2050 - get_rmse(dtr_truth[35], m_dtr[35]), 4)}")
    print(f"RMSE at 2100 Diff: {np.round(orig_dtr_2100 - get_rmse(dtr_truth[85], m_dtr[85]), 4)}")

    print("\npr\n")
    print(f"RMSE at 2015 Diff: {np.round(orig_pr_2015 - get_rmse(pr_truth[0], m_pr[0]), 4)}")
    print(f"RMSE at 2050 Diff: {np.round(orig_pr_2050 - get_rmse(pr_truth[35], m_pr[35]), 4)}")
    print(f"RMSE at 2100 Diff: {np.round(orig_pr_2100 - get_rmse(pr_truth[85], m_pr[85]), 4)}")

    print("\npr_90\n")
    print(f"RMSE at 2015 Diff: {np.round(orig_pr90_2015 - get_rmse(pr90_truth[0], m_pr90[0]), 4)}")
    print(f"RMSE at 2050 Diff: {np.round(orig_pr90_2050 - get_rmse(pr90_truth[35], m_pr90[35]), 4)}")
    print(f"RMSE at 2100 Diff: {np.round(orig_pr90_2100 - get_rmse(pr90_truth[85], m_pr90[85]), 4)}")

    # Plotting new model under the truth!

    divnorm=colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)

    with plt.style.context("dark_background"):
        ax = plt.axes(projection=ccrs.PlateCarree())
        tas_truth.sel(time=2050).plot(cmap="coolwarm", norm=divnorm,
                                      cbar_kwargs={"label":"Temperature change / K"})
        ax.set_title("tas True 2050")
        ax.coastlines()

    plt.show()

    with plt.style.context("dark_background"):
        ax = plt.axes(projection=ccrs.PlateCarree())
        m_tas.sel(sample=35).plot(cmap="coolwarm", norm=divnorm,
                                 cbar_kwargs={"label":"Temperature change / K"})
        ax.set_title("tas Emulated 2050")
        ax.coastlines()

    plt.show()

    with plt.style.context("dark_background"):
        ax = plt.axes(projection=ccrs.PlateCarree())
        pr_truth.sel(time=2050).plot(cmap="coolwarm", norm=divnorm,
                                      cbar_kwargs={"label":"Precipitation change"})
        ax.set_title("pr True 2050")
        ax.coastlines()

    plt.show()

    with plt.style.context("dark_background"):
        ax = plt.axes(projection=ccrs.PlateCarree())
        m_pr.sel(sample=35).plot(cmap="coolwarm", norm=divnorm,
                                 cbar_kwargs={"label":"Precipitation change"})
        ax.set_title("pr Emulated 2050")
        ax.coastlines()

    plt.show()

    with plt.style.context("dark_background"):
        ax = plt.axes(projection=ccrs.PlateCarree())
        dtr_truth.sel(time=2050).plot(cmap="coolwarm", norm=divnorm,
                                      cbar_kwargs={"label":"Diurnal Temperature Range change / K"})
        ax.set_title("dtr True 2050")
        ax.coastlines()

    plt.show()

    with plt.style.context("dark_background"):
        ax = plt.axes(projection=ccrs.PlateCarree())
        m_dtr.sel(sample=35).plot(cmap="coolwarm", norm=divnorm,
                                 cbar_kwargs={"label":"Diurnal Temperature Range change / K"})
        ax.set_title("dtr Emulated 2050")
        ax.coastlines()

    plt.show()
    return

# Examples of using the function:

testing(["Linear"], "add")

testing(["RBF"], "add")

testing(["Polynomial"], "add")

testing(["Linear", "RBF"], "add")

testing(["Linear", "Polynomial"], "add")

testing(["Polynomial", "RBF"], "add")

testing(["Linear", "RBF"], "mul")

testing(["Linear", "Polynomial"], "mul")

testing(["Polynomial", "RBF"], "mul")