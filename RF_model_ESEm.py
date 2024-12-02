#!/usr/bin/env python
# coding: utf-8

# In[11]:


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


# In[12]:


# path to save the netcdf file
path_output ='outputs_ssp245_prediction_ESEm.nc'


# In[13]:


train_files = [ "historical", "ssp585", "ssp126", "ssp370", ]
# Create training and testing arrays
X, solvers = create_predictor_data(train_files)
Y = create_predictdand_data(train_files)



# In[75]:


xr.open_dataset("inputs_historical.nc")


# In[15]:


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


# In[16]:


## Test on SSP245

X_test = get_test_data('ssp245', solvers)
Y_test = create_predictdand_data(['ssp245'])

tas_truth = Y_test["tas"]
pr_truth = Y_test["pr"]
pr90_truth = Y_test["pr90"]
dtr_truth = Y_test["diurnal_temperature_range"]


# In[17]:


m_out_tas, _ = rf_tas.predict(X_test)
m_out_pr, _ = rf_pr.predict(X_test)
m_out_pr90, _ = rf_pr90.predict(X_test)
m_out_dtr, _ = rf_dtr.predict(X_test)


# In[18]:


xr_output = xr.Dataset(dict(tas=m_out_tas, pr=m_out_pr, pr90=m_out_pr90, diurnal_temperature_range=m_out_dtr)).assign_coords(time=m_out_tas.sample + 2014)
#save output to netcdf 
xr_output.to_netcdf(path_output,'w')


# In[19]:


print(f"RMSE: {get_rmse(tas_truth[35:], m_out_tas[35:]).mean()}")
print("\n")

print(f"RMSE: {get_rmse(dtr_truth[35:], m_out_dtr[35:]).mean()}")
print("\n")

print(f"RMSE: {get_rmse(pr_truth[35:], m_out_pr[35:]).mean()}")
print("\n")

print(f"RMSE: {get_rmse(pr90_truth[35:], m_out_pr90[35:]).mean()}")


# In[36]:


# Define the color normalization
divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)


# In[62]:


# Create a side-by-side figure
fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(14, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
with plt.style.context("dark_background"):
    ax = axes[0]
    tas_truth.sel(time=2050).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("True 2050")
    ax.coastlines()

# Plot the emulated result
with plt.style.context("dark_background"):
    ax = axes[1]
    m_out_tas.sel(sample=35).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("Emulated 2050")
    ax.coastlines()

fig.suptitle("tas (Near-Surface Air Temperature)", fontsize=18, x=0.45)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("tas_comparison.png", dpi=80, bbox_inches="tight")

plt.show()


# In[66]:


# Create a side-by-side figure
fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(14, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
with plt.style.context("dark_background"):
    ax = axes[0]
    dtr_truth.sel(time=2050).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("True 2050")
    ax.coastlines()

# Plot the emulated result
with plt.style.context("dark_background"):
    ax = axes[1]
    m_out_dtr.sel(sample=35).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("Emulated 2050")
    ax.coastlines()

fig.suptitle("dtr (Diurnal Temperature Range)", fontsize=18, x=0.45)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("dtr_comparison.png", dpi=80, bbox_inches="tight")

plt.show()


# In[65]:


# Create a side-by-side figure
fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(14, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
with plt.style.context("dark_background"):
    ax = axes[0]
    pr_truth.sel(time=2050).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("True 2050")
    ax.coastlines()

# Plot the emulated result
with plt.style.context("dark_background"):
    ax = axes[1]
    m_out_pr.sel(sample=35).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("Emulated 2050")
    ax.coastlines()

fig.suptitle("pr (Precipitation)", fontsize=18, x=0.45)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("pr_comparison.png", dpi=80, bbox_inches="tight")

plt.show()


# In[68]:


# Create a side-by-side figure
fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(14, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot the true projection
with plt.style.context("dark_background"):
    ax = axes[0]
    pr90_truth.sel(time=2050).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("True 2050")
    ax.coastlines()

# Plot the emulated result
with plt.style.context("dark_background"):
    ax = axes[1]
    m_out_pr90.sel(sample=35).plot(
        ax=ax, cmap="coolwarm", norm=divnorm,
        cbar_kwargs={"label": "Temperature change / K"}
    )
    ax.set_title("Emulated 2050")
    ax.coastlines()

fig.suptitle("pr90 (90th Percentile Precipitation)", fontsize=18, x=0.45)

# Adjust layout for better appearance
fig.tight_layout()

# Save the figure
fig.savefig("pr90_comparison.png", dpi=80, bbox_inches="tight")

plt.show()


# In[29]:


from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, check_scoring, mean_squared_error

def get_rmse_array(truth, pred):
    print(truth.shape, pred.shape)
    weights = np.cos(np.deg2rad(truth.lat))

    return np.sqrt(((truth - pred.reshape(-1, 96,144))**2).weighted(weights).mean(['lat', 'lon'])).data.mean()


# In[30]:


get_ipython().run_cell_magic('time', '', 'pr_result = permutation_importance(\n    rf_pr.model.model, X_test[35:], pr_truth.sel(time=slice(2050,None)), n_repeats=10, random_state=42, n_jobs=1, scoring=make_scorer(get_rmse_array))\n')


# In[31]:


importances = rf_pr.model.model.feature_importances_


# In[32]:


feature_names = list(X.columns)


# In[88]:


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


# In[122]:


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

