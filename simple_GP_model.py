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

def view_model_rmse(kernels, op):
    """
    kernels: A list of strings that contain the kernels to be used in the model
    op: A string that says how the kernels should be added together

    This function prints the difference of the Original's RMSE - New RMSE
    """
    # Train the model
    tas_gp = gp_model(leading_historical_inputs, Y["tas"], kernel=kernels, kernel_op = op)
    tas_gp.train()
    pr_gp = gp_model(leading_historical_inputs, Y["pr"], kernel=kernels, kernel_op = op)
    pr_gp.train()
    dtr_gp = gp_model(leading_historical_inputs, Y["diurnal_temperature_range"], kernel=kernels, kernel_op = op)
    dtr_gp.train()
    pr90_gp = gp_model(leading_historical_inputs, Y["pr90"], kernel=kernels, kernel_op = op)
    pr90_gp.train()

    # Make predictions
    m_tas, _ = tas_gp.predict(test_inputs)
    m_pr, _ = pr_gp.predict(test_inputs)
    m_pr90, _ = pr90_gp.predict(test_inputs)
    m_dtr, _ = dtr_gp.predict(test_inputs)
    
    # Look at RMSE
    print(f"RMSE for {kernels} combined with {op}: {get_rmse(tas_truth, m_tas)}")
    print(f"RMSE for {kernels} combined with {op}: {get_rmse(dtr_truth, m_dtr)}")
    print(f"RMSE for {kernels} combined with {op}: {get_rmse(pr_truth, m_pr)}")
    print(f"RMSE for {kernels} combined with {op}: {get_rmse(pr90_truth, m_pr90)}")
    return

# Examples of using the function:

view_model_rmse(["Linear"], "add")

view_model_rmse(["RBF"], "add")

view_model_rmse(["Polynomial"], "add")

view_model_rmse(["Linear", "RBF"], "add")

view_model_rmse(["Linear", "Polynomial"], "add")

view_model_rmse(["Polynomial", "RBF"], "add")

view_model_rmse(["Linear", "RBF"], "mul")

view_model_rmse(["Linear", "Polynomial"], "mul")

view_model_rmse(["Polynomial", "RBF"], "mul")

# We found Linear + Polynomial worked best and was closest to default

kernels = ["Linear", "Polynomial"]
op = "add"

tas_gp = gp_model(leading_historical_inputs, Y["tas"], kernel=kernels, kernel_op = op)
tas_gp.train()
pr_gp = gp_model(leading_historical_inputs, Y["pr"], kernel=kernels, kernel_op = op)
pr_gp.train()
dtr_gp = gp_model(leading_historical_inputs, Y["diurnal_temperature_range"], kernel=kernels, kernel_op = op)
dtr_gp.train()
pr90_gp = gp_model(leading_historical_inputs, Y["pr90"], kernel=kernels, kernel_op = op)
pr90_gp.train()

m_tas, _ = tas_gp.predict(test_inputs)
m_pr, _ = pr_gp.predict(test_inputs)
m_pr90, _ = pr90_gp.predict(test_inputs)
m_dtr, _ = dtr_gp.predict(test_inputs)

# tas plot

# Create a figure with three side-by-side subplots
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(21, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
ax = axes[0]
tas_truth.sel(time=2050).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Temperature change / K"}
)
ax.set_title("True 2050")
ax.coastlines()

# Plot the emulated result
ax = axes[1]
m_tas.sel(sample=35).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Temperature change / K"}
)
ax.set_title("Emulated 2050")
ax.coastlines()

# Plot the difference (True - Emulated)
ax = axes[2]
difference = tas_truth.sel(time=2050) - m_tas.sel(sample=35)  # Calculate the difference
difference.plot(
    ax=ax, cmap="bwr",  # Use a diverging colormap to highlight positive/negative differences
    cbar_kwargs={"label": "Difference in Temperature change / K"}
)
ax.set_title("Difference (True - Emulated)")
ax.coastlines()

# Add a main title
fig.suptitle("tas (Near-Surface Air Temperature)", fontsize=18, x=0.5)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("gp_tas_comparison_with_difference.png", dpi=250, bbox_inches="tight")

plt.show()

# dtr plot

# Create a figure with three side-by-side subplots
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(21, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
ax = axes[0]
dtr_truth.sel(time=2050).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Temperature change / K"}
)
ax.set_title("True 2050")
ax.coastlines()

# Plot the emulated result
ax = axes[1]
m_dtr.sel(sample=35).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Temperature change / K"}
)
ax.set_title("Emulated 2050")
ax.coastlines()

# Plot the difference (True - Emulated)
ax = axes[2]
difference = dtr_truth.sel(time=2050) - m_dtr.sel(sample=35)  # Calculate the difference
difference.plot(
    ax=ax, cmap="bwr",  # Use a diverging colormap to highlight positive/negative differences
    cbar_kwargs={"label": "Difference in Temperature change / K"}
)
ax.set_title("Difference (True - Emulated)")
ax.coastlines()

# Add a main title
fig.suptitle("dtr (Diurnal Temperature Range)", fontsize=18, x=0.5)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("gp_dtr_comparison_with_difference.png", dpi=250, bbox_inches="tight")

plt.show()

# pr plot

# Create a figure with three side-by-side subplots
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(21, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
ax = axes[0]
pr_truth.sel(time=2050).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Precipitation change / mm day-1"}
)
ax.set_title("True 2050")
ax.coastlines()

# Plot the emulated result
ax = axes[1]
m_dtr.sel(sample=35).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Precipitation change / mm day-1"}
)
ax.set_title("Emulated 2050")
ax.coastlines()

# Plot the difference (True - Emulated)
ax = axes[2]
difference = dtr_truth.sel(time=2050) - m_dtr.sel(sample=35)  # Calculate the difference
difference.plot(
    ax=ax, cmap="bwr",  # Use a diverging colormap to highlight positive/negative differences
    cbar_kwargs={"label": "Difference in Precipitation change / mm day-1"}
)
ax.set_title("Difference (True - Emulated)")
ax.coastlines()

# Add a main title
fig.suptitle("pr (Precipitation)", fontsize=18, x=0.5)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("gp_pr_comparison_with_difference.png", dpi=250, bbox_inches="tight")

plt.show()

# pr90 plot

# Create a figure with three side-by-side subplots
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(21, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
ax = axes[0]
pr90_truth.sel(time=2050).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Precipitation change / mm day-1"}
)
ax.set_title("True 2050")
ax.coastlines()

# Plot the emulated result
ax = axes[1]
m_dtr.sel(sample=35).plot(
    ax=ax, cmap="coolwarm", norm=divnorm,
    cbar_kwargs={"label": "Precipitation change / mm day-1"}
)
ax.set_title("Emulated 2050")
ax.coastlines()

# Plot the difference (True - Emulated)
ax = axes[2]
difference = dtr_truth.sel(time=2050) - m_dtr.sel(sample=35)  # Calculate the difference
difference.plot(
    ax=ax, cmap="bwr",  # Use a diverging colormap to highlight positive/negative differences
    cbar_kwargs={"label": "Difference in Precipitation change / mm day-1"}
)
ax.set_title("Difference (True - Emulated)")
ax.coastlines()

# Add a main title
fig.suptitle("pr90 (90th percentile Precipitation)", fontsize=18, x=0.5)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("gp_pr90_comparison_with_difference.png", dpi=250, bbox_inches="tight")

plt.show()