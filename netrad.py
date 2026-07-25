import xarray as xr
import matplotlib.pyplot as plt

# File paths for input NetCDF files
file_lw_tot_toa = 'netrad_lw_tot_ipsl_toa.nc'
file_sw_tot_toa = 'netrad_sw_tot_ipsl_toa.nc'

file_lw_tot_sfc = 'netrad_lw_tot_ipsl_sfc.nc'
file_sw_tot_sfc = 'netrad_sw_tot_ipsl_sfc.nc'

# Load the NetCDF files
data_lw_tot_toa = xr.open_dataset(file_lw_tot_toa)
data_sw_tot_toa = xr.open_dataset(file_sw_tot_toa)

data_lw_tot_sfc = xr.open_dataset(file_lw_tot_sfc)
data_sw_tot_sfc = xr.open_dataset(file_sw_tot_sfc)

# Calculate IRF values
netrad_lw = data_lw_tot_toa['netrad_lw_tot'] - data_lw_tot_sfc['netrad_lw_tot']
netrad_sw = data_sw_tot_toa['netrad_sw_tot'] - data_sw_tot_sfc['netrad_sw_tot']
netrad_total = netrad_lw + netrad_sw

#-----------------------------------------------------------

# Save IRF values into a new NetCDF file
output_file = 'netrad_ipsl.nc'
netrad_dataset = xr.Dataset({
    'netrad_Total': netrad_total,
    'netrad_lw': netrad_lw,
    'netrad_sw': netrad_sw
})
netrad_dataset.to_netcdf(output_file)




