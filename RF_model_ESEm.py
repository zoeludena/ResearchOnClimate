# %%
import os

os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"

import datetime as dt 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr

from sklearn import metrics

from eofs.xarray import Eof
from esem import rf_model

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors

from utils import *

# %%
# path to save the netcdf file
path_output ='outputs_ssp245_prediction_ESEm.nc'


# %%
train_files = [ "historical", "ssp585", "ssp126", "ssp370", ]
# Create training and testing arrays
X, solvers = create_predictor_data(train_files)
Y = create_predictdand_data(train_files)



# %%
xr.open_dataset("inputs_historical.nc")

# %%

# rf_tas = rf_model(X, Y['tas'], random_state=0, bootstrap=True, max_features='auto', **{'n_estimators': 250, 'min_samples_split': 5, 'min_samples_leaf': 7,  'max_depth': 5,})
# rf_pr = rf_model(X, Y['pr'], random_state=0, bootstrap=True, max_features='auto', **{'n_estimators': 150, 'min_samples_split': 15, 'min_samples_leaf': 8,'max_depth': 40,})
# rf_pr90 = rf_model(X, Y['pr90'], random_state=0, bootstrap=True, max_features='auto',**{'n_estimators': 250, 'min_samples_split': 15, 'min_samples_leaf': 12,'max_depth': 25,})
# rf_dtr = rf_model(X, Y["diurnal_temperature_range"], random_state=0, bootstrap=True, max_features='auto',**{'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 12, 'max_depth': 20,})

# Seems like max_features='auto' no longer works, replacing with 'sqrt'
rf_tas = rf_model(X, Y['tas'], random_state=0, bootstrap=True, max_features='sqrt', **{'n_estimators': 250, 'min_samples_split': 5, 'min_samples_leaf': 7,  'max_depth': 5,})
rf_pr = rf_model(X, Y['pr'], random_state=0, bootstrap=True, max_features='sqrt', **{'n_estimators': 150, 'min_samples_split': 15, 'min_samples_leaf': 8,'max_depth': 40,})
rf_pr90 = rf_model(X, Y['pr90'], random_state=0, bootstrap=True, max_features='sqrt',**{'n_estimators': 250, 'min_samples_split': 15, 'min_samples_leaf': 12,'max_depth': 25,})
rf_dtr = rf_model(X, Y["diurnal_temperature_range"], random_state=0, bootstrap=True, max_features='sqrt',**{'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 12, 'max_depth': 20,})

rf_tas.train()
rf_pr.train()
rf_pr90.train()
rf_dtr.train()


# %%
## Test on SSP245

X_test = get_test_data('ssp245', solvers)
Y_test = create_predictdand_data(['ssp245'])

tas_truth = Y_test["tas"]
pr_truth = Y_test["pr"]
pr90_truth = Y_test["pr90"]
dtr_truth = Y_test["diurnal_temperature_range"]


# %%
m_out_tas, _ = rf_tas.predict(X_test)
m_out_pr, _ = rf_pr.predict(X_test)
m_out_pr90, _ = rf_pr90.predict(X_test)
m_out_dtr, _ = rf_dtr.predict(X_test)


# %%

xr_output = xr.Dataset(dict(tas=m_out_tas, pr=m_out_pr, pr90=m_out_pr90, diurnal_temperature_range=m_out_dtr)).assign_coords(time=m_out_tas.sample + 2014)
#save output to netcdf 
xr_output.to_netcdf(path_output,'w')

# %%
print(f"RMSE: {get_rmse(tas_truth[35:], m_out_tas[35:]).mean()}")
print("\n")

print(f"RMSE: {get_rmse(dtr_truth[35:], m_out_dtr[35:]).mean()}")
print("\n")

print(f"RMSE: {get_rmse(pr_truth[35:], m_out_pr[35:]).mean()}")
print("\n")

print(f"RMSE: {get_rmse(pr90_truth[35:], m_out_pr90[35:]).mean()}")

# %%
# Define the color normalization
divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)

# %%
## Temperature
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(18, 3))
fig.suptitle('Near-Surface Air Temperature')

# Test
plt.subplot(131, projection=proj)
tas_truth.sel(time=2050).plot(cmap="coolwarm", vmax=3,
                              cbar_kwargs={"label":"Temperature change / K"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='True')

# Emulator
plt.subplot(132, projection=proj)
m_out_tas.sel(sample=35).plot(cmap="coolwarm", vmax=3,
                       cbar_kwargs={"label":"Temperature change / K"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Random Forest')

# Difference
difference = tas_truth.sel(time=2050) - m_out_tas.sel(sample=35)
plt.subplot(133, projection=proj)
difference.plot(cmap="bwr", vmax=1,
                cbar_kwargs={"label":"Temperature change / K"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Difference')
fig.tight_layout()

plt.savefig('rf_tas_comparison.png', dpi=250, bbox_inches="tight")

# %%
## dtr
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(18, 3))
fig.suptitle('Diurnal Temperature Range')

# Test
plt.subplot(131, projection=proj)
dtr_truth.sel(time=2050).plot(cmap="coolwarm", vmax=3,
                              cbar_kwargs={"label":"Temperature / K"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='True')

# Emulator
plt.subplot(132, projection=proj)
m_out_dtr.sel(sample=35).plot(cmap="coolwarm", vmax=3,
                       cbar_kwargs={"label":"Temperature / K"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Random Forest')

# Difference
difference = dtr_truth.sel(time=2050) - m_out_dtr.sel(sample=35)
plt.subplot(133, projection=proj)
difference.plot(cmap="bwr", vmax=1,
                cbar_kwargs={"label":"Temperature / K"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Difference')
fig.tight_layout()

plt.savefig('rf_dtr_comparison.png', dpi=250, bbox_inches="tight")

# %%
## pr
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(18, 3))
fig.suptitle('Precipitation')

# Test
plt.subplot(131, projection=proj)
pr_truth.sel(time=2050).plot(cmap="coolwarm", vmax=3,
                              cbar_kwargs={"label":"Precipitation (mm / day)"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='True')

# Emulator
plt.subplot(132, projection=proj)
m_out_pr.sel(sample=35).plot(cmap="coolwarm", vmax=3,
                       cbar_kwargs={"label":"Precipitation (mm / day)"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Random Forest')

# Difference
difference = pr_truth.sel(time=2050) - m_out_pr.sel(sample=35)
plt.subplot(133, projection=proj)
difference.plot(cmap="bwr", vmax=1,
                cbar_kwargs={"label":"Precipitation (mm / day)"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Difference')
fig.tight_layout()

plt.savefig('rf_pr_comparison.png', dpi=250, bbox_inches="tight")

# %%
## pr90
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(18, 3))
fig.suptitle('90th Percentile Precipitation')

# Test
plt.subplot(131, projection=proj)
pr90_truth.sel(time=2050).plot(cmap="coolwarm", vmax=3,
                              cbar_kwargs={"label":"Precipitation (mm / day)"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='True')

# Emulator
plt.subplot(132, projection=proj)
m_out_pr90.sel(sample=35).plot(cmap="coolwarm", vmax=3,
                       cbar_kwargs={"label":"Precipitation (mm / day)"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Random Forest')

# Difference
difference = pr90_truth.sel(time=2050) - m_out_pr90.sel(sample=35)
plt.subplot(133, projection=proj)
difference.plot(cmap="bwr", vmax=1,
                cbar_kwargs={"label":"Precipitation (mm / day)"})
plt.gca().coastlines()
plt.setp(plt.gca(), title='Difference')
fig.tight_layout()

plt.savefig('rf_pr90_comparison.png', dpi=250, bbox_inches="tight")

# %%
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, check_scoring, mean_squared_error

def get_rmse_array(truth, pred):
    print(truth.shape, pred.shape)
    weights = np.cos(np.deg2rad(truth.lat))

    return np.sqrt(((truth - pred.reshape(-1, 96,144))**2).weighted(weights).mean(['lat', 'lon'])).data.mean()

# %%
%%time
pr_result = permutation_importance(
    rf_pr.model.model, X_test[35:], pr_truth.sel(time=slice(2050,None)), n_repeats=10, random_state=42, n_jobs=1, scoring=make_scorer(get_rmse_array))


# %%
importances = rf_pr.model.model.feature_importances_

# %%
feature_names = list(X.columns)

# %%
# Feature names
feature_names = ["CO2", "CH4", "BC_0", "BC_1", "BC_2", "BC_3", "BC_4", "SO2_0", "SO2_1", "SO2_2", "SO2_3", "SO2_4"]

# Start a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define the models and titles
models = [rf_tas, rf_dtr, rf_pr, rf_pr90]
titles = ["tas", "dtr", "pr", "pr90"]

# Plot each feature importance in the grid
for i, (model, title) in enumerate(zip(models, titles)):
    row, col = divmod(i, 2)  # Determine the subplot position
    std = np.std([tree.feature_importances_ for tree in model.model.model.estimators_], axis=0)
    importances = model.model.model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    
    # Plot bar chart
    forest_importances.plot.bar(yerr=std, ax=axes[row, col])
    axes[row, col].set_title(title)
    axes[row, col].set_ylabel("Feature importances")

# Adjust layout and save
fig.tight_layout()
fig.savefig("rf_feature_importances_2x2.png")
plt.show()


# %%
# Feature names
feature_names = ["CO2", "CH4", "BC_0", "BC_1", "BC_2", "BC_3", "BC_4", "SO2_0", "SO2_1", "SO2_2", "SO2_3", "SO2_4"]

# Collect feature importances and standard deviations for all models
models = [rf_tas, rf_dtr, rf_pr, rf_pr90]
titles = ["tas", "dtr", "pr", "pr90"]

importances = []
stds = []

for model in models:
    std = np.std([tree.feature_importances_ for tree in model.model.model.estimators_], axis=0)
    importance = model.model.model.feature_importances_
    importances.append(importance)
    stds.append(std)

# Convert to DataFrame for plotting
df_importances = pd.DataFrame(importances, columns=feature_names, index=titles)
df_stds = pd.DataFrame(stds, columns=feature_names, index=titles)

# Create a grouped bar plot
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(feature_names))  # the label locations
width = 0.2  # the width of the bars

# Plot bars for each model
for i, title in enumerate(titles):
    ax.bar(
        x + i * width,
        df_importances.loc[title],
        width,
        yerr=df_stds.loc[title],
        label=title,
        capsize=4
    )

# Add labels, legend, and formatting
ax.set_xlabel("Features", size=18)
ax.set_ylabel("Feature Importances", size=18)
ax.set_title("Feature Importances by Model", size=20)
ax.set_xticks(x + width * (len(titles) - 1) / 2)
ax.set_xticklabels(feature_names, rotation=45, size=16)
ax.tick_params(axis='y', labelsize=14)
ax.legend(title="Models")
fig.tight_layout()

# Save and show the plot
fig.savefig("rf_feature_importances_grouped.png")
plt.show()



