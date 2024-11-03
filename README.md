# ResearchOnClimate

**Dependencies**

Users are encouraged to use NCAR Casper Login*.
Data is available from the sixth Coupled Model Intercomparison Product (CMIP6). It is publicly archived and available, but the data is readily available on Casper. Here is a link to the data on the Earth System Grid Federation Portal at Centre for Environmental Data Analysis and the cloud.
Run “pip install -r requirements.txt” to download any missing Python dependencies.

(*NCAR Casper Login Aside: You can create an account by following the directions on the Casper website.)

**Exploratory Data Analysis**

Run the explore_CMIP6_data.py file to generate a variety of graphs of the historical_r1i1p1f1 model output data. Can also use this file as an example of how to generate new spatial plots of any other variables.

**Prepare Data for Model Input**

Use the prepare_data.py file to generate training-ready data files for the emulators. The preprocessing includes variable selection, annual averaging, and feature derivations (ie. diurnal temperature range = tasmax - tasmin).

**Data storage structure**

`/glade/collections/cmip/CMIP6/{MIP}/NCC/NorESM2-LM/{experiment}/{member}/{general variable category}/{variable}/*/*/{variable}/*.nc`

- MIP (Model Intercomparison Projects) : 'CMIP', 'DAMIP', 'ScenarioMIP', 'AerChemMIP'
- experiment: 
  - CMIP: '1pctCO2', 'abrupt-4xCO2', 'historical', 'piControl'
  - DAMIP: 'hist-GHG', 'hist-aer'
  - ScenarioMIP: 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF', 'ssp585'
- member:
  - E.g., ‘r1i1p1f1’, ‘r2i1p1d1’,  ‘r2i1p1d1’
  - `r` for realization, `i` for initialization, `p` for physics, and `f` for forcing
- General variable category:

| Acronym | Spelled Out Version                   |
|---------|-------------------------------|
| Amon    | Atmospheric Month             |
| Omon    | Oceanic Month                 |
| day     | Daily                         |
| Oday    | Oceanic Daily                 |
| Eday    | Earth Daily                   |

- variable:

**List of Climate Variables**
| Acronym     | Spelled Out Version                              |
|-------------|-------------------------------------------|
| prc         | Convective Precipitation                  |
| rtmt        | Net Top-of-Model Radiation                |
| hfls        | Surface Latent Heat Flux                  |
| ch4global   | Global Methane Concentration              |
| hur         | Relative Humidity                         |
| sfcWind     | Surface Wind Speed                        |
| rlds        | Surface Downwelling Longwave Radiation    |
| prw         | Water Vapor Path                          |
| hus         | Specific Humidity                         |
| n2oglobal   | Global Nitrous Oxide Concentration        |
| clwvi       | Column-Integrated Liquid Water            |
| prsn        | Snowfall Flux                             |
| va          | Meridional Wind                           |
| cfc11global | Global CFC-11 Concentration               |
| tauu        | Zonal Wind Stress                         |
| rlutcs      | Clear-Sky Upwelling Longwave Radiation    |
| tas         | Near-Surface Air Temperature              |
| clt         | Total Cloud Fraction                      |
| tauv        | Meridional Wind Stress                    |
| cfc12global | Global CFC-12 Concentration               |
| ci          | Sea Ice Concentration                     |
| co2         | Carbon Dioxide Concentration              |
| clivi       | Column-Integrated Ice Water               |
| cl          | Cloud Fraction                            |
| ua          | Zonal Wind                                |
| hfss        | Surface Sensible Heat Flux                |
| ps          | Surface Pressure                          |
| o3          | Ozone Concentration                       |
| wap         | Vertical Velocity                         |
| rsds        | Surface Downwelling Shortwave Radiation   |
| rlus        | Surface Upwelling Longwave Radiation      |
| rsutcs      | Clear-Sky Upwelling Shortwave Radiation   |
| rsdt        | Top-of-Atmosphere Downwelling Shortwave Radiation |
| tasmin      | Minimum Near-Surface Air Temperature      |
| ta          | Air Temperature                           |
| rlut        | Top-of-Atmosphere Upwelling Longwave Radiation |
| hurs        | Near-Surface Relative Humidity            |
| co2mass     | Total Mass of Carbon Dioxide              |
| sbl         | Sea Ice Thickness                         |
| rldscs      | Clear-Sky Surface Downwelling Longwave Radiation |
| psl         | Sea-Level Pressure                        |
| pr          | Precipitation                             |
| tasmax      | Maximum Near-Surface Air Temperature      |
| clw         | Cloud Liquid Water Content                |
| huss        | Near-Surface Specific Humidity            |
| zg          | Geopotential Height                       |
| evspsbl     | Evaporation                               |
| rsuscs      | Clear-Sky Surface Upwelling Shortwave Radiation |
| rsus        | Surface Upwelling Shortwave Radiation     |
| cli         | Cloud Ice Water Content                   |
| rsut        | Top-of-Atmosphere Upwelling Shortwave Radiation |
| ts          | Surface Temperature                       |
| rsdscs      | Clear-Sky Surface Downwelling Shortwave Radiation |
