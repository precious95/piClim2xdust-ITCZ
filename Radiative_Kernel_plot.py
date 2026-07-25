import os
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.colors import TwoSlopeNorm


DATA_DIR = Path("/Users/precious/Downloads/moisture-budget")


PR_ITCZ_DIR = Path("/Users/precious/Downloads/moisture-budget/pr")

os.chdir(DATA_DIR)

# ============================================================
# CONFIG
# ============================================================
models = [
    ("CNRM-ESM2-1", "cnrm"),
    ("GFDL-ESM4", "gfdl"),
    ("GISS-E2-1-G", "giss"),
    ("IPSL-CM6A-LR-INCA", "ipsl"),
    ("MIROC6", "miroc6"),
    ("MPI-ESM-1-2-HAM", "mpi"),
    ("UKESM1-0-LL", "ukesm"),
]

MODEL_COLORS = {
    "MME":    "black",
    "cnrm":   "#ff5a36",
    "gfdl":   "#d9b382",
    "giss":   "#c4833d",
    "ipsl":   "#4b8a57",
    "miroc6": "#8c908a",
    "mpi":    "#d9cf4a",
    "ukesm":  "#b07b72",
}

FIELD_INFO = {
    "netrad":   {"title": r"$dR_{atm}$"},
    "irf":      {"title": "IRF"},
    "fluxanom": {"title": r"$A_c$"},
}

field_order = ["netrad", "irf", "fluxanom"]

VAR_CANDIDATES = {
    "netrad":   ["netrad_Total", "netrad"],
    "irf":      ["IRF_Total", "irf_Total", "irf", "IRF"],
    "fluxanom": ["fluxanom_Total", "fluxanom"],
    "pr":       ["pr"],
}

MME_TOP_FILES = {
    "netrad": "MME_netrad.nc",
    "fluxanom": "MME_fluxanom.nc",
}


lon0, lon1, lat0, lat1 = -60, 40, -20, 40


lon_min_latmon, lon_max_latmon = -40, 5
lat_min_data, lat_max_data = -12, 25
lat_min_plot, lat_max_plot = -10, 25

OCEAN_BOX = (-40, 5, 0, 10)
SAHEL_BOX = (-40, 5, -10, 0)

FV = 1e20

plt.rcParams["font.size"] = 12
plt.rcParams["hatch.color"] = "gray"
plt.rcParams["hatch.linewidth"] = 0.5


# ----------Functions---------
def coord_names(ds_or_da):
    lon_candidates = ["lon", "longitude", "x", "rlon", "nav_lon"]
    lat_candidates = ["lat", "latitude", "y", "rlat", "nav_lat"]

    lon = next((c for c in lon_candidates if c in ds_or_da.coords or c in ds_or_da.dims), None)
    lat = next((c for c in lat_candidates if c in ds_or_da.coords or c in ds_or_da.dims), None)

    if lon is None or lat is None:
        raise KeyError(
            f"Could not find lon/lat names in coords={list(ds_or_da.coords)} dims={list(ds_or_da.dims)}"
        )

    return lon, lat


def open_dataset_robust(fname, decode_times=True):
    errors = []

    for engine in ["h5netcdf", "scipy", "netcdf4", None]:
        try:
            if engine is None:
                ds = xr.open_dataset(fname, decode_times=decode_times)
            else:
                ds = xr.open_dataset(fname, decode_times=decode_times, engine=engine)

            print(f"Opened {fname} with {engine if engine else 'default'} (decode_times={decode_times})")
            return ds

        except Exception as e:
            errors.append(f"{engine if engine else 'default'}: {repr(e)}")

    raise RuntimeError(f"Could not open {fname}\n" + "\n".join(errors))


def find_var(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]

    lower_map = {k.lower(): k for k in ds.data_vars}

    for name in candidates:
        if name.lower() in lower_map:
            return ds[lower_map[name.lower()]]

    for name in candidates:
        for dv in ds.data_vars:
            if name.lower() in dv.lower():
                return ds[dv]

    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]

    raise ValueError(f"None of {candidates} found. Variables are: {list(ds.data_vars)}")


def coord_is_bad(da, coord_name):
    """
    True when a lon/lat coordinate exists but is unusable.

    This fixes the IRF issue where the file has a lon coordinate,
    but all longitude values are 0.0. In that case xarray thinks lon
    exists, so the old code did not borrow the correct lon coordinate
    from the matching precipitation file.
    """
    if coord_name not in da.coords:
        return True

    vals = np.asarray(da[coord_name].values)

    try:
        vals = vals.astype(float)
    except Exception:
        return True

    vals = vals[np.isfinite(vals)]

    if vals.size < 2:
        return True

    if np.nanmax(vals) - np.nanmin(vals) == 0:
        return True

    return False


def borrow_missing_lonlat(da_target, da_ref):
    """
    Borrow lon/lat from the reference precipitation file if the target
    coordinate is missing OR bad.

    Needed for IRF files where lon exists but is constant/all zeros.
    """
    target_lon, target_lat = coord_names(da_target)
    ref_lon, ref_lat = coord_names(da_ref)

    if coord_is_bad(da_target, target_lon):
        if target_lon in da_target.dims and da_target.sizes[target_lon] == da_ref.sizes[ref_lon]:
            da_target = da_target.assign_coords({target_lon: da_ref[ref_lon].values})
            print(f"Borrowed longitude from pr reference for {da_target.name}")

    if coord_is_bad(da_target, target_lat):
        if target_lat in da_target.dims and da_target.sizes[target_lat] == da_ref.sizes[ref_lat]:
            da_target = da_target.assign_coords({target_lat: da_ref[ref_lat].values})
            print(f"Borrowed latitude from pr reference for {da_target.name}")

    return da_target


def standardize_lon(da):
    lon_name, _ = coord_names(da)

    if lon_name in da.coords and np.issubdtype(da[lon_name].dtype, np.number):
        lon = da[lon_name]
        lon180 = ((lon + 180) % 360) - 180
        da = da.assign_coords({lon_name: lon180}).sortby(lon_name)

    return da


def to_0360(da):
    lon_name, _ = coord_names(da)

    if lon_name in da.coords and np.issubdtype(da[lon_name].dtype, np.number):
        lon360 = da[lon_name] % 360
        da = da.assign_coords({lon_name: lon360}).sortby(lon_name)

    return da


def rename_lonlat_to_standard(da):
    lon_name, lat_name = coord_names(da)

    rename_dict = {}

    if lon_name != "lon":
        rename_dict[lon_name] = "lon"

    if lat_name != "lat":
        rename_dict[lat_name] = "lat"

    if rename_dict:
        da = da.rename(rename_dict)

    return da


def sort_lat_if_needed(da):
    if "lat" in da.coords and da["lat"].size > 1:
        if np.any(np.diff(da["lat"].values) < 0):
            da = da.sortby("lat")

    return da


def clean_da(da):
    da = da.where(np.isfinite(da))
    da = da.where(np.abs(da) < FV)
    da = standardize_lon(da)

    return da


def safe_sel_box(da, lon_bounds, lat_bounds):
    lon_name, lat_name = coord_names(da)

    if lon_name in da.coords and da[lon_name].size > 0:
        lon_vals = da[lon_name].values
        x0, x1 = lon_bounds

        if np.all(np.diff(lon_vals) >= 0):
            da = da.sel({lon_name: slice(min(x0, x1), max(x0, x1))})
        else:
            da = da.sel({lon_name: slice(max(x0, x1), min(x0, x1))})

    if lat_name in da.coords and da[lat_name].size > 0:
        lat_vals = da[lat_name].values
        y0, y1 = lat_bounds

        if np.all(np.diff(lat_vals) >= 0):
            da = da.sel({lat_name: slice(min(y0, y1), max(y0, y1))})
        else:
            da = da.sel({lat_name: slice(max(y0, y1), min(y0, y1))})

    return da


def subset_lon_wrap_0360(da, lon_left=-60, lon_right=40, lat_bottom=-20, lat_top=40):
    """
    Convert to 0..360 and make the domain continuous.
    -60..40 becomes 300..400 by shifting 0..40 to 360..400.
    """
    lon_name, lat_name = coord_names(da)

    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)
    da = to_0360(da)

    west = da.sel({lon_name: slice(300, 360)})
    east = da.sel({lon_name: slice(0, 40)})

    if east[lon_name].size > 0:
        east = east.assign_coords({lon_name: east[lon_name] + 360})

    if west[lon_name].size == 0 and east[lon_name].size == 0:
        return da.isel({lon_name: slice(0, 0)})

    if west[lon_name].size == 0:
        da_wrap = east.sortby(lon_name)
    elif east[lon_name].size == 0:
        da_wrap = west.sortby(lon_name)
    else:
        da_wrap = xr.concat([west, east], dim=lon_name).sortby(lon_name)

    lat_vals = da_wrap[lat_name].values

    if np.all(np.diff(lat_vals) >= 0):
        da_wrap = da_wrap.sel({lat_name: slice(min(lat_bottom, lat_top), max(lat_bottom, lat_top))})
    else:
        da_wrap = da_wrap.sel({lat_name: slice(max(lat_bottom, lat_top), min(lat_bottom, lat_top))})

    return da_wrap


def back_to_180_for_plot(da):
    lon_name, _ = coord_names(da)
    lon180 = ((da[lon_name] + 180) % 360) - 180
    da = da.assign_coords({lon_name: lon180}).sortby(lon_name)

    return da


def fill_allnan_lon_columns(da):
    """
    
    """
    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)

    if "lon" not in da.dims or da.sizes.get("lon", 0) < 3:
        return da

    other_dims = [d for d in da.dims if d != "lon"]
    all_nan_by_lon = da.isnull().all(dim=other_dims)
    bad_lons = da["lon"].where(all_nan_by_lon, drop=True).values

    if len(bad_lons) == 0:
        return da

    out = da.copy(deep=True)
    lon_vals = out["lon"].values

    for bad_lon in bad_lons:
        j = int(np.argmin(np.abs(lon_vals - bad_lon)))

        if j == 0 or j == len(lon_vals) - 1:
            print(f"Skipping edge all-NaN lon column at lon={float(bad_lon):.3f}")
            continue

        left = out.isel(lon=j - 1)
        right = out.isel(lon=j + 1)
        fill_col = xr.concat([left, right], dim="_fill_side").mean("_fill_side", skipna=True)

        out.loc[{"lon": out["lon"].isel(lon=j)}] = fill_col
        print(f"Filled all-NaN lon column at lon={float(bad_lon):.3f}")

    return out


def fix_internal_lon_gap(da, target_lon=0.0, max_bad_frac=0.5):
    """
    If the longitude column nearest target_lon is mostly NaN,
    replace it by the mean of the immediate left/right columns.
    """
    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)

    lon = da["lon"].values
    Z = da.transpose("lat", "lon").values.copy()

    j = int(np.argmin(np.abs(lon - target_lon)))

    if j == 0 or j == len(lon) - 1:
        return da

    bad_mask = ~np.isfinite(Z[:, j])
    bad_frac = bad_mask.mean()

    print(f"Nearest lon to {target_lon} is {lon[j]:.3f}, bad fraction = {bad_frac:.3f}")

    if bad_frac >= max_bad_frac:
        left = Z[:, j - 1]
        right = Z[:, j + 1]
        Z[:, j] = np.nanmean(np.column_stack([left, right]), axis=1)

        da_fixed = xr.DataArray(
            Z,
            dims=("lat", "lon"),
            coords={"lat": da["lat"].values, "lon": lon},
            attrs=da.attrs,
            name=da.name,
        )

        return da_fixed

    return da


def prep_component_map(da):
    da = clean_da(da)
    da = safe_sel_box(da, lon_bounds=(lon0, lon1), lat_bounds=(lat0, lat1))

    return da


def prep_component_latmon(da):
    da = clean_da(da)
    da = safe_sel_box(
        da,
        lon_bounds=(lon_min_latmon, lon_max_latmon),
        lat_bounds=(lat_min_data, lat_max_data),
    )

    return da


def xyZ(da):
    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)
    da2 = da.transpose("lat", "lon")

    lon = da2["lon"].values
    lat = da2["lat"].values
    Z = da2.values

    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180

    idx = np.argsort(lon)
    lon = lon[idx]
    Z = Z[:, idx]

    # remove exact duplicate lon columns if present after wrap-back
    _, unique_idx = np.unique(np.round(lon, 8), return_index=True)
    unique_idx = np.sort(unique_idx)
    lon = lon[unique_idx]
    Z = Z[:, unique_idx]

    return lon, lat, Z


def get_month_values(time_coord):
    vals = time_coord.values
    months = []

    for t in vals:
        try:
            ts = pd.Timestamp(t)

            if ts.year < 1901 or ts.year == 1970:
                raise ValueError

            months.append(int(ts.month))
            continue

        except Exception:
            pass

        try:
            months.append(int(t.month))
            continue

        except Exception:
            pass

    if len(months) != len(vals):
        ntime = len(vals)
        months = ((np.arange(ntime) % 12) + 1).astype(int)

    return np.array(months, dtype=int)


def get_year_values(time_coord):
    vals = time_coord.values
    years = []

    for t in vals:
        try:
            ts = pd.Timestamp(t)
            years.append(int(ts.year))
            continue

        except Exception:
            pass

        try:
            years.append(int(t.year))
            continue

        except Exception:
            pass

    if len(years) != len(vals):
        ntime = len(vals)
        years = np.repeat(np.arange(ntime // 12 + 1), 12)[:ntime]

    return np.array(years, dtype=int)


def get_days_in_month_values(time_coord):
    vals = time_coord.values
    days = []

    for t in vals:
        try:
            ts = pd.Timestamp(t)

            if ts.year < 1901 or ts.year == 1970:
                raise ValueError

            days.append(int(ts.days_in_month))
            continue

        except Exception:
            pass

        try:
            days.append(int(t.daysinmonth))
            continue

        except Exception:
            pass

    if len(days) != len(vals):
        ntime = len(vals)
        days = np.full(ntime, 30, dtype=int)

    return np.array(days, dtype=int)


def add_month_coord_if_needed(da):
    return da.assign_coords(month=("time", get_month_values(da["time"])))


def annual_mean_by_year(da):
    years = get_year_values(da["time"])
    da2 = da.assign_coords(year=("time", years))

    days = xr.DataArray(
        get_days_in_month_values(da["time"]),
        dims=["time"],
        coords={"time": da["time"]},
    ).assign_coords(year=("time", years))

    num = (da2 * days).groupby("year").sum("time", skipna=True)
    den = days.groupby("year").sum("time", skipna=True)

    return num / den


def area_mean_box(da, box):
    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)

    x0, x1, y0, y1 = box

    lon_vals = da["lon"].values
    lat_vals = da["lat"].values

    if np.all(np.diff(lon_vals) >= 0):
        lon_slice = slice(min(x0, x1), max(x0, x1))
    else:
        lon_slice = slice(max(x0, x1), min(x0, x1))

    if np.all(np.diff(lat_vals) >= 0):
        lat_slice = slice(min(y0, y1), max(y0, y1))
    else:
        lat_slice = slice(max(y0, y1), min(y0, y1))

    sub = da.sel({"lon": lon_slice, "lat": lat_slice})
    weights = np.cos(np.deg2rad(sub["lat"]))

    return sub.weighted(weights).mean(dim=["lat", "lon"], skipna=True)


def mean_sem_from_series(da):
    arr = np.asarray(da.values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan, np.nan

    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.nan

    return mean, sem


def build_common_grid_irf_wrap():
    dlon = (lon1 - lon0) / 240.0
    dlat = (lat1 - lat0) / 180.0

    common_lon_vals = np.arange(300 + 0.5 * dlon, 400, dlon)
    common_lat_vals = np.arange(lat0, lat1 + 0.5 * dlat, dlat)

    common_lon = xr.DataArray(
        common_lon_vals,
        dims=["lon"],
        coords={"lon": common_lon_vals},
    )

    common_lat = xr.DataArray(
        common_lat_vals,
        dims=["lat"],
        coords={"lat": common_lat_vals},
    )

    return common_lon, common_lat




def add_shaded_boundary(ax, lw=1.8, shade_width=3.0,
                        shade_color="black", line_color="darkgreen"):
    ax.add_patch(
        PathPatch(
            box_path,
            transform=ccrs.PlateCarree(),
            fill=False,
            linewidth=lw + shade_width,
            edgecolor=shade_color,
            zorder=8,
        )
    )

    ax.add_patch(
        PathPatch(
            box_path,
            transform=ccrs.PlateCarree(),
            fill=False,
            linewidth=lw,
            edgecolor=line_color,
            zorder=9,
        )
    )


def draw_boxes(ax, lw=1.8):
    x0, x1, y0, y1 = OCEAN_BOX

    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            transform=ccrs.PlateCarree(),
            fill=False,
            linewidth=lw,
            edgecolor="limegreen",
            zorder=11,
        )
    )

    s0, s1, t0, t1 = SAHEL_BOX

    ax.add_patch(
        Rectangle(
            (s0, t0),
            s1 - s0,
            t1 - t0,
            transform=ccrs.PlateCarree(),
            fill=False,
            linewidth=lw,
            edgecolor="red",
            zorder=11,
        )
    )


# ============================================================
#
def load_reference_pr(model):
    fname = DATA_DIR / f"pr_{model}.nc"
    ds = open_dataset_robust(fname, decode_times=True)
    da = find_var(ds, VAR_CANDIDATES["pr"])
    ds.close()

    return da


def load_component(varname, model):
    fname = DATA_DIR / f"{varname}_{model}.nc"
    ds = open_dataset_robust(fname, decode_times=True)
    da = find_var(ds, VAR_CANDIDATES[varname])

    if varname == "irf":
        ref = load_reference_pr(model)
        da = borrow_missing_lonlat(da, ref)

        # Repair IRF coordinates before any subsetting or wrapping.
        # This is the main fix for the white vertical line near 0° longitude.
        da = standardize_lon(da)
        da = rename_lonlat_to_standard(da)
        da = sort_lat_if_needed(da)
        da = fill_allnan_lon_columns(da)

    ds.close()

    return da


def load_top_mme_file(field_key):
    fname = DATA_DIR / MME_TOP_FILES[field_key]
    ds = open_dataset_robust(fname, decode_times=False)
    da = find_var(ds, VAR_CANDIDATES[field_key])
    ds.close()

    return da


def load_model_fields(model):
    print(f"\nLoading {model}")

    out = {}

    for field_key in field_order:
        da = load_component(field_key, model)

        if field_key == "irf":
            map_da = clean_da(da)   # keep full cleaned field for wrapped processing
        else:
            map_da = prep_component_map(da)

        out[field_key] = {
            "map": map_da,
            "latmon": prep_component_latmon(da),
        }

    return out



# DIAGNOSTICS

def annual_map_from_precomputed_mme(field_key):
    da = load_top_mme_file(field_key)
    da = prep_component_map(da)

    lon_name, lat_name = coord_names(da)

    if da[lon_name].size == 0 or da[lat_name].size == 0:
        raise ValueError(
            f"{field_key}: empty domain after subsetting. "
            f"lon size={da[lon_name].size}, lat size={da[lat_name].size}"
        )

    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)

    return da.mean("time", skipna=True) if "time" in da.dims else da


def annual_map_irf_from_models(field_model_data):
    common_lon, common_lat = build_common_grid_irf_wrap()

    arrays = []

    for _, m in models:
        da = field_model_data[m]["map"]

        annual_field = annual_mean_by_year(da).mean("year", skipna=True)
        annual_field = rename_lonlat_to_standard(annual_field)
        annual_field = sort_lat_if_needed(annual_field)

        annual_field = subset_lon_wrap_0360(
            annual_field,
            lon_left=lon0,
            lon_right=lon1,
            lat_bottom=lat0,
            lat_top=lat1,
        )

        #
        annual_field = fill_allnan_lon_columns(annual_field)

        if annual_field["lon"].size == 0 or annual_field["lat"].size == 0:
            print(f"Skipping {m}: empty after wrapped lon subset")
            continue

        annual_interp = annual_field.interp(
            lon=common_lon,
            lat=common_lat,
            method="nearest",
        )

        valid_n = int(np.isfinite(annual_interp.values).sum())
        print(f"IRF model {m}: wrapped-grid valid cells = {valid_n}")

        arrays.append(annual_interp.expand_dims(model=[m]))

    if len(arrays) == 0:
        raise ValueError("IRF: no valid model fields available for annual MME spatial map.")

    stack = xr.concat(arrays, dim="model", join="outer", coords="minimal", compat="override")
    mme = stack.mean("model", skipna=True)

    mme = back_to_180_for_plot(mme)

    # 
    mme = fill_allnan_lon_columns(mme)
    mme = fix_internal_lon_gap(mme, target_lon=0.0, max_bad_frac=0.5)

    lon = mme["lon"].values
    Z = mme.transpose("lat", "lon").values
    jj = np.argmin(np.abs(lon - 0.0))
    lo = max(0, jj - 2)
    hi = min(len(lon), jj + 3)

    print("MME lon near 0:", lon[lo:hi])
    print("MME finite counts near 0:", np.sum(np.isfinite(Z[:, lo:hi]), axis=0))

    valid_n_mme = int(np.isfinite(mme.values).sum())
    print(f"IRF annual MME valid cells = {valid_n_mme}")

    return mme


def monthly_latmon_mme(field_model_data):
    common_lat = np.linspace(lat_min_data, lat_max_data, 100)

    common_lat_da = xr.DataArray(
        common_lat,
        dims=["lat"],
        coords={"lat": common_lat},
    )

    common_month = xr.DataArray(
        np.arange(1, 13),
        dims=["month"],
        coords={"month": np.arange(1, 13)},
    )

    latmon_list = []

    for _, m in models:
        da = field_model_data[m]["latmon"]
        da = rename_lonlat_to_standard(da)
        da = sort_lat_if_needed(da)

        mon = add_month_coord_if_needed(da).groupby("month").mean("time", skipna=True)
        latmon = mon.mean("lon", skipna=True)
        latmon = latmon.interp(lat=common_lat_da, month=common_month)

        latmon_list.append(latmon.expand_dims(model=[m]))

    return xr.concat(latmon_list, dim="model").mean("model", skipna=True)



# ITCZ lines for latitude-month 

ITCZ_LAT_MIN = -20
ITCZ_LAT_MAX = 20
#ITCZ_SMOOTH_MONTH = 3


def open_pr_itcz_file(kind):
    """
    
    """
    if kind == "ctl":
        candidates = [
            "MME_ctl.nc",
            "pr_MME_ctl.nc",
            "pr_ctl.nc",
            "ctl_pr.nc",
            "control_pr.nc",
        ]

    elif kind == "dust":
        candidates = [
            "MME_dust.nc",
            "pr_MME_dust.nc",
            "pr_dust.nc",
            "dust_pr.nc",
            "2xdust_pr.nc",
            "piClim-2xDust_pr.nc",
        ]

    else:
        raise ValueError("kind must be 'ctl' or 'dust'")

    search_dirs = [PR_ITCZ_DIR, DATA_DIR]

    for folder in search_dirs:
        for fname in candidates:
            path = folder / fname

            if path.exists():
                ds = open_dataset_robust(path, decode_times=True)
                da = find_var(ds, VAR_CANDIDATES["pr"])
                da = da.load()
                ds.close()

                print(f"Loaded {kind} ITCZ precipitation from {path}")
                return da

    print(f"WARNING: could not find {kind} precipitation file for ITCZ lines.")

    return None


def monthly_pr_latmon_for_itcz(pr_da):
    """
   
    """
    if pr_da is None:
        return None

    da = clean_da(pr_da)
    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)

    da = safe_sel_box(
        da,
        lon_bounds=(lon_min_latmon, lon_max_latmon),
        lat_bounds=(ITCZ_LAT_MIN, ITCZ_LAT_MAX),
    )

   
    units = str(da.attrs.get("units", "")).lower()

    if "s-1" in units or "s^-1" in units or "/s" in units:
        da = da * 86400.0

    mon = add_month_coord_if_needed(da).groupby("month").mean("time", skipna=True)

    # Zonal mean precipitation over the Atlantic-West Africa sector.
    latmon = mon.mean("lon", skipna=True)

    return latmon


def itcz_lat_month_centroid(
    pr_latmon,
    lat_name="lat",
    lat_band=(ITCZ_LAT_MIN, ITCZ_LAT_MAX),
    smooth_month=ITCZ_SMOOTH_MONTH,
):
    """
    Monthly ITCZ latitude from the precipitation-centroid method.

    Formula:
        phi_cent =
            sum(phi * cos(phi) * P(phi)) / sum(cos(phi) * P(phi))

    
    """
    if pr_latmon is None:
        return None

    if lat_name not in pr_latmon.coords and lat_name not in pr_latmon.dims:
        lat_name = "lat"

    pr_sub = pr_latmon.sel({
        lat_name: slice(lat_band[0], lat_band[1])
    })

    if pr_sub[lat_name].size == 0:
        return None

    if np.any(np.diff(pr_sub[lat_name].values) < 0):
        pr_sub = pr_sub.sortby(lat_name)

    lat_deg = pr_sub[lat_name]
    weights = np.cos(np.deg2rad(lat_deg))

    numerator = (pr_sub * lat_deg * weights).sum(
        dim=lat_name,
        skipna=True,
    )

    denominator = (pr_sub * weights).sum(
        dim=lat_name,
        skipna=True,
    )

    itcz = numerator / denominator

    if smooth_month and smooth_month > 1 and "month" in itcz.dims:
        itcz = itcz.rolling(
            month=smooth_month,
            center=True,
            min_periods=1,
        ).mean()

    return itcz


def load_itcz_lines_for_panels():
    """
    Return smoothed monthly Control and 2xDust ITCZ curves
    using the precipitation-centroid method.

    If files are missing, the script continues without ITCZ lines.
    """
    pr_ctl = open_pr_itcz_file("ctl")
    pr_dust = open_pr_itcz_file("dust")

    ctl_latmon = monthly_pr_latmon_for_itcz(pr_ctl)
    dust_latmon = monthly_pr_latmon_for_itcz(pr_dust)

    ctl_itcz = None
    dust_itcz = None

    if ctl_latmon is not None:
        ctl_itcz = itcz_lat_month_centroid(
            ctl_latmon,
            lat_name="lat",
            lat_band=(ITCZ_LAT_MIN, ITCZ_LAT_MAX),
            smooth_month=ITCZ_SMOOTH_MONTH,
        )

    if dust_latmon is not None:
        dust_itcz = itcz_lat_month_centroid(
            dust_latmon,
            lat_name="lat",
            lat_band=(ITCZ_LAT_MIN, ITCZ_LAT_MAX),
            smooth_month=ITCZ_SMOOTH_MONTH,
        )

    return ctl_itcz, dust_itcz


# ============================================================
# BOX STATS
# ============================================================
def annual_box_series(da, box):
    # for IRF maps stored full-domain, subset happens here
    da = rename_lonlat_to_standard(da)
    da = sort_lat_if_needed(da)

    if da["lon"].max() > 180:
        da = back_to_180_for_plot(da)

    da = safe_sel_box(
        da,
        lon_bounds=(lon0, lon1),
        lat_bounds=(lat0, lat1),
    )

    return area_mean_box(annual_mean_by_year(da), box)


def mme_box_stats(field_model_data, box):
    series_list = []

    for _, m in models:
        s = annual_box_series(field_model_data[m]["map"], box)
        series_list.append(s.expand_dims(model=[m]))

    stack = xr.concat(series_list, dim="model", join="outer", coords="minimal", compat="override")

    return mean_sem_from_series(stack.mean("model", skipna=True))


def model_box_stats(field_model_data, box):
    means, sems = [], []

    for _, m in models:
        s = annual_box_series(field_model_data[m]["map"], box)
        mm, ss = mean_sem_from_series(s)

        means.append(mm)
        sems.append(ss)

    return means, sems


# 

def plot_map_panel(ax, da_delta, title=None, panel_label=None, levels=None, norm=None):
    lon, lat, Z = xyZ(da_delta)

    ax.set_extent([lon0, lon1, lat0, lat1], ccrs.PlateCarree())
    ax.set_boundary(box_path, transform=ccrs.PlateCarree())
    add_shaded_boundary(ax)

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.92", zorder=0)
    ax.coastlines(resolution="50m", lw=0.6, zorder=7)

    mesh = ax.contourf(
        lon,
        lat,
        Z,
        levels=levels,
        cmap="RdBu_r",
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    draw_boxes(ax)

    if title is not None:
        ax.set_title(title, fontsize=16, pad=6)

    if panel_label is not None:
        ax.text(
            0.03,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

    gl = ax.gridlines(draw_labels=False, linewidth=0.30, color="gray", alpha=0.35)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    return mesh


def plot_latmonth_panel(
    ax,
    da_latmon,
    panel_label=None,
    levels=None,
    norm=None,
    ctl_itcz=None,
    dust_itcz=None,
    add_itcz_legend=False,
):
    Z = da_latmon.transpose("lat", "month").values
    lats = da_latmon["lat"].values
    months = da_latmon["month"].values
    M, L = np.meshgrid(months, lats)

    cf = ax.contourf(
        M,
        L,
        Z,
        levels=levels,
        cmap="RdBu_r",
        norm=norm,
        extend="both",
    )

    ax.contour(
        M,
        L,
        Z,
        levels=levels[::2],
        colors="k",
        linewidths=0.35,
        alpha=0.35,
    )

    o0, o1 = sorted([OCEAN_BOX[2], OCEAN_BOX[3]])
    s0, s1 = sorted([SAHEL_BOX[2], SAHEL_BOX[3]])

    # Latitude bounds of the green and red boxes
    ax.hlines(
        [o0, o1],
        xmin=1,
        xmax=12,
        colors="limegreen",
        linewidth=1.4,
        alpha=0.95,
        zorder=6,
    )

    ax.hlines(
        [s0, s1],
        xmin=1,
        xmax=12,
        colors="red",
        linewidth=1.4,
        alpha=0.95,
        zorder=6,
    )

    # ITCZ lines from precipitation centroid:
    # black = Control, red dashed = 2xDust
    if ctl_itcz is not None:
        ax.plot(
            ctl_itcz["month"].values,
            ctl_itcz.values,
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="Control ITCZ",
            zorder=9,
        )

    if dust_itcz is not None:
        ax.plot(
            dust_itcz["month"].values,
            dust_itcz.values,
            color="firebrick",
            linewidth=2.0,
            linestyle="--",
            label="2xDust ITCZ",
            zorder=10,
        )

    if add_itcz_legend and (ctl_itcz is not None or dust_itcz is not None):
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    if panel_label is not None:
        ax.text(
            0.03,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

    ax.set_xlim(1, 12)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(calendar.month_abbr[1:13])
    ax.set_ylim(lat_min_plot, lat_max_plot)

    yticks = np.arange(lat_min_plot, lat_max_plot + 1, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [
            f"{abs(int(y))}°S" if y < 0 else ("0°" if y == 0 else f"{int(y)}°N")
            for y in yticks
        ]
    )

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)

    return cf


def plot_box_bar_panel(
    ax,
    north_means,
    north_sems,
    south_means,
    south_sems,
    panel_label=None,
    title=None,
    ylabel=None,
    ylim=None,
    add_legend=False,
    show_yaxis=True,
):
    """
    Bottom-row bar panels using fixed box colors:
    North box = green, South box = red.
    """
    entries = [("MME", "MME")] + models
    x = np.arange(len(entries))
    w = 0.36

    north_color = "green"
    south_color = "red"

    for i, (label, key) in enumerate(entries):
        # North box: green with hatch
        ax.bar(
            x[i] - w / 2,
            north_means[i],
            width=w,
            color=north_color,
            edgecolor="black",
            hatch="\\\\",
            linewidth=0.6,
            alpha=0.95,
            zorder=2,
        )

        ax.errorbar(
            x[i] - w / 2,
            north_means[i],
            yerr=north_sems[i],
            fmt="o",
            mfc="white",
            mec="black",
            mew=1.0,
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            ms=4.5,
            zorder=4,
        )

        # South box: red with hatch
        ax.bar(
            x[i] + w / 2,
            south_means[i],
            width=w,
            color=south_color,
            edgecolor="black",
            hatch="//",
            linewidth=0.6,
            alpha=0.70,
            zorder=1,
        )

        ax.errorbar(
            x[i] + w / 2,
            south_means[i],
            yerr=south_sems[i],
            fmt="o",
            mfc="white",
            mec="black",
            mew=0.9,
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            ms=4.5,
            zorder=4,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in entries], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.35)

    if title is not None:
        ax.set_title(title, fontsize=16, pad=4)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if not show_yaxis:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    if panel_label is not None:
        ax.text(
            0.03,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

    if add_legend:
        handles = [
            Patch(facecolor=north_color, edgecolor="black", hatch="\\\\", label="North TA"),
            Patch(facecolor=south_color, edgecolor="black", hatch="//", label="South TA"),
        ]

        ax.legend(handles=handles, frameon=True, fontsize=9, loc="upper right")


# -------------
model_fields = {}

for _, m in models:
    model_fields[m] = load_model_fields(m)


# 
annual_map = {}
latmon = {}
bar_stats = {}
top_map_vals = []

for field_key in field_order:
    field_model_data = {
        m: model_fields[m][field_key]
        for _, m in models
    }

    if field_key == "irf":
        delta_mme = annual_map_irf_from_models(field_model_data)
    else:
        delta_mme = annual_map_from_precomputed_mme(field_key)

    annual_map[field_key] = {
        "delta": delta_mme
    }

    vals = np.asarray(delta_mme.values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size > 0:
        top_map_vals.append(vals)

    latmon[field_key] = monthly_latmon_mme(field_model_data)

    north_means, north_sems, south_means, south_sems = [], [], [], []

    mm, ss = mme_box_stats(field_model_data, OCEAN_BOX)
    north_means.append(mm)
    north_sems.append(ss)

    mm, ss = mme_box_stats(field_model_data, SAHEL_BOX)
    south_means.append(mm)
    south_sems.append(ss)

    mns, sms = model_box_stats(field_model_data, OCEAN_BOX)
    north_means.extend(mns)
    north_sems.extend(sms)

    mns, sms = model_box_stats(field_model_data, SAHEL_BOX)
    south_means.extend(mns)
    south_sems.extend(sms)

    bar_stats[field_key] = {
        "north_means": north_means,
        "north_sems": north_sems,
        "south_means": south_means,
        "south_sems": south_sems,
    }



# COLOR SCALES

if len(top_map_vals) > 0:
    top_map_vals = np.concatenate(top_map_vals)
    vmax_top = float(np.nanquantile(np.abs(top_map_vals), 0.95))
else:
    vmax_top = 1.0

if not np.isfinite(vmax_top) or vmax_top == 0:
    vmax_top = 1.0

norm_top = TwoSlopeNorm(vcenter=0.0, vmin=-vmax_top, vmax=vmax_top)
top_levels = np.linspace(-vmax_top, vmax_top, 17)

latmon_vals = []

for field_key in field_order:
    vals = np.asarray(latmon[field_key].values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]

    if vals.size > 0:
        latmon_vals.append(vals)

latmon_vals = np.concatenate(latmon_vals) if len(latmon_vals) > 0 else np.array([1.0])
vmax_latmon = float(np.nanquantile(np.abs(latmon_vals), 0.99))

if not np.isfinite(vmax_latmon) or vmax_latmon == 0:
    vmax_latmon = 1.0

norm_latmon = TwoSlopeNorm(vcenter=0.0, vmin=-vmax_latmon, vmax=vmax_latmon)
latmon_levels = np.linspace(-vmax_latmon, vmax_latmon, 21)

bar_vals = []

for field_key in field_order:
    d = bar_stats[field_key]

    vals = np.array(
        d["north_means"] +
        d["south_means"] +
        list(np.array(d["north_means"]) + np.nan_to_num(d["north_sems"], nan=0.0)) +
        list(np.array(d["south_means"]) + np.nan_to_num(d["south_sems"], nan=0.0)) +
        list(np.array(d["north_means"]) - np.nan_to_num(d["north_sems"], nan=0.0)) +
        list(np.array(d["south_means"]) - np.nan_to_num(d["south_sems"], nan=0.0))
    )

    vals = vals[np.isfinite(vals)]

    if vals.size > 0:
        bar_vals.append(vals)

if len(bar_vals) > 0:
    bar_vals = np.concatenate(bar_vals)
    ymin = min(0.0, bar_vals.min())
    ymax = max(0.0, bar_vals.max())
    pad = 0.10 * (ymax - ymin if ymax > ymin else 1.0)
    common_bar_ylim = (ymin - pad, ymax + pad)
else:
    common_bar_ylim = (-1, 1)


# Load Control and 2xDust ITCZ curves for panels (d), (e), and (f)
ctl_itcz_line, dust_itcz_line = load_itcz_lines_for_panels()



# FIGURE

proj = ccrs.LambertConformal(
    central_longitude=-10,
    central_latitude=10,
    standard_parallels=(10, 20),
)

fig = plt.figure(figsize=(18, 15))

gs = fig.add_gridspec(
    3,
    3,
    left=0.06,
    right=0.90,
    top=0.96,
    bottom=0.08,
    wspace=0.08,
    hspace=0.24,
    height_ratios=[1.0, 0.95, 0.90],
)

axes_top = [
    fig.add_subplot(gs[0, i], projection=proj)
    for i in range(3)
]

axes_mid = [
    fig.add_subplot(gs[1, i])
    for i in range(3)
]

axb0 = fig.add_subplot(gs[2, 0])

axes_bot = [
    axb0,
    fig.add_subplot(gs[2, 1], sharey=axb0),
    fig.add_subplot(gs[2, 2], sharey=axb0),
]

letters = [
    f"({chr(97 + i)})"
    for i in range(9)
]



# TOP ROW

top_images = []

for i, field_key in enumerate(field_order):
    im_map = plot_map_panel(
        ax=axes_top[i],
        da_delta=annual_map[field_key]["delta"],
        title=FIELD_INFO[field_key]["title"],
        panel_label=letters[i],
        levels=top_levels,
        norm=norm_top,
    )

    top_images.append(im_map)


# MIDDLE ROW

for i, field_key in enumerate(field_order):
    plot_latmonth_panel(
        ax=axes_mid[i],
        da_latmon=latmon[field_key],
        panel_label=letters[3 + i],
        levels=latmon_levels,
        norm=norm_latmon,
        ctl_itcz=ctl_itcz_line,
        dust_itcz=dust_itcz_line,
        add_itcz_legend=(i == 2),
    )

    axes_mid[i].set_xlabel("Month", fontsize=13)

    if i == 0:
        axes_mid[i].set_ylabel("Latitude", fontsize=13)
    else:
        axes_mid[i].set_yticklabels([])



# BOTTOM ROW

for i, field_key in enumerate(field_order):
    plot_box_bar_panel(
        ax=axes_bot[i],
        north_means=bar_stats[field_key]["north_means"],
        north_sems=bar_stats[field_key]["north_sems"],
        south_means=bar_stats[field_key]["south_means"],
        south_sems=bar_stats[field_key]["south_sems"],
        panel_label=letters[6 + i],
        title=FIELD_INFO[field_key]["title"],
        ylabel=r"(W m$^{-2}$)" if i == 0 else None,
        ylim=common_bar_ylim,
        add_legend=(i == 0),
        show_yaxis=(i == 0),
    )



# COLORBAR

cax = fig.add_axes([0.92, 0.33, 0.018, 0.42])

cbar = fig.colorbar(
    top_images[0],
    cax=cax,
    orientation="vertical",
    extend="both",
)

cbar.set_label(r"(W m$^{-2}$)", fontsize=13)
cbar.ax.tick_params(labelsize=10)



plt.savefig(
    "figure4_updated_40-5.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
