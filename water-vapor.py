import xarray as xr
import matplotlib.pyplot as plt

# 
file_lw_tot_toa_trop = 'fluxanom_lw_q_trop_tot_mpi_toa.nc'
file_sw_tot_toa_trop = 'fluxanom_sw_q_trop_tot_mpi_toa.nc'

file_lw_tot_sfc_trop = 'fluxanom_lw_q_trop_tot_mpi_sfc.nc'
file_sw_tot_sfc_trop = 'fluxanom_sw_q_trop_tot_mpi_sfc.nc'

# 
data_lw_tot_toa_trop = xr.open_dataset(file_lw_tot_toa_trop)
data_sw_tot_toa_trop = xr.open_dataset(file_sw_tot_toa_trop)

data_lw_tot_sfc_trop = xr.open_dataset(file_lw_tot_sfc_trop)
data_sw_tot_sfc_trop = xr.open_dataset(file_sw_tot_sfc_trop)

# Calculate water vapor 
fluxanom_lw_trop = data_lw_tot_toa_trop['fluxanom_lw_q_trop_tot'] - data_lw_tot_sfc_trop['fluxanom_lw_q_trop_tot']
fluxanom_sw_trop = data_sw_tot_toa_trop['fluxanom_sw_q_trop_tot'] - data_sw_tot_sfc_trop['fluxanom_sw_q_trop_tot']
fluxanom_total_trop = fluxanom_lw_trop + fluxanom_sw_trop

#-----------------------------------------------------------

# 
file_lw_tot_toa_strat = 'fluxanom_lw_q_strat_tot_mpi_toa.nc'
file_sw_tot_toa_strat = 'fluxanom_sw_q_strat_tot_mpi_toa.nc'

file_lw_tot_sfc_strat = 'fluxanom_lw_q_strat_tot_mpi_sfc.nc'
file_sw_tot_sfc_strat = 'fluxanom_sw_q_strat_tot_mpi_sfc.nc'

# 
data_lw_tot_toa_strat = xr.open_dataset(file_lw_tot_toa_strat)
data_sw_tot_toa_strat = xr.open_dataset(file_sw_tot_toa_strat)

data_lw_tot_sfc_strat = xr.open_dataset(file_lw_tot_sfc_strat)
data_sw_tot_sfc_strat = xr.open_dataset(file_sw_tot_sfc_strat)

# Calculate values
fluxanom_lw_strat = data_lw_tot_toa_strat['fluxanom_lw_q_strat_tot'] - data_lw_tot_sfc_strat['fluxanom_lw_q_strat_tot']
fluxanom_sw_strat = data_sw_tot_toa_strat['fluxanom_sw_q_strat_tot'] - data_sw_tot_sfc_strat['fluxanom_sw_q_strat_tot']
fluxanom_total_strat = fluxanom_lw_strat + fluxanom_sw_strat


#------------------------------------------------------------------------------
hus = fluxanom_total_trop +  fluxanom_total_strat

# Save into a new NetCDF file
output_file = 'hus_mpi.nc'
fluxanom_dataset = xr.Dataset({
    'hus': hus,
    'hus_Total_trop': fluxanom_total_trop,
    'hus_Total_strat': fluxanom_total_strat    
})
fluxanom_dataset.to_netcdf(output_file)
