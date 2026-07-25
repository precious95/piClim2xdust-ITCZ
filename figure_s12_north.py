#!/usr/bin/env python3
import os
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# SETTINGS YOU EDIT
# ============================================================
ROOT   = "/home/precious/Downloads/aerchem2/ForcingFeedbacksResults_CMIP6_2xdust/net/aerchem/p3/p4/moisture-budget_models"
REGION = "north"   # "north" (0–10N) or "equatorial" (-10–0)
ADD_MME = True

SEASONS = ["DJF", "MAM", "JJA", "SON"]
MODELS  = ["cnrm", "gfdl", "giss", "ipsl", "miroc6", "mpi", "ukesm"]  # MME added below

MODEL_TITLES = {
    "cnrm": "CNRM-ESM2-1",
    "gfdl": "GFDL-ESM4",
    "giss": "GISS-E2-1-G",
    "ipsl": "IPSL-CM6A-LR",
    "miroc6": "MIROC6",
    "mpi": "MPI-ESM1-2",
    "ukesm": "UKESM1-0-LL",
    "mme": "MME",
}

# Wide enough domain for derivatives, consistent with your map script
LON0, LON1, LAT0, LAT1 = -60, 40, -20, 40

# Vertical integral bounds
P_BOT_PA = 100000.0  # 1000 hPa
P_TOP_PA = 10000.0   # 100 hPa

# Analysis boxes
BOX_LON0, BOX_LON1 = -40, 5

# North tropical Atlantic box
NORTH_LAT0, NORTH_LAT1 = 0, 10

# Equatorial / southern tropical Atlantic box
EQ_LAT0, EQ_LAT1 = -10, 0

# Output
OUTPNG = f"updated_figure_S12_north1.png"

# Per-season y-limits
SEASON_YLIM = {
    "DJF": 0.30,
    "MAM": 0.50,
    "JJA": 0.50,
    "SON": 0.30,
}

# ============================================================
# CONSTANTS
# ============================================================
R_EARTH = 6_371_000.0
DEG2RAD = np.pi / 180.0
G = 9.80665
SEC_PER_DAY = 86400.0
MIN_VALID_SEGMENTS = 3

# ============================================================
# HELPERS
# ============================================================
def ensure_lat_lon_names(da):
    latn = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
    lonn = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)

    if latn is None or lonn is None:
        raise ValueError("Could not find lat/lon coordinates (lat/lon or latitude/longitude).")

    if latn != "lat":
        da = da.rename({latn: "lat"})

    if lonn != "lon":
        da = da.rename({lonn: "lon"})

    return da


def standardize_lon_180(da):
    lon = da["lon"]
    lon180 = ((lon + 180) % 360) - 180
    return da.assign_coords(lon=lon180).sortby("lon")


def find_pressure_name(da):
    for name in ["plev", "lev", "level", "pressure", "p"]:
        if (name in da.coords) or (name in da.dims):
            return name
    return None


def to_pa_and_select_range(da, ptop_pa=P_TOP_PA, pbot_pa=P_BOT_PA):
    p_name = find_pressure_name(da)

    if p_name is None:
        raise ValueError("No pressure coordinate found (plev/lev/level/pressure/p).")

    if p_name != "plev":
        da = da.rename({p_name: "plev"})

    p = da["plev"]
    p_pa = xr.where(p.max() > 2000, p, p * 100.0)  # if hPa -> Pa
    da = da.assign_coords(plev=p_pa).sortby("plev")

    ptop_sel = float(da["plev"].sel(plev=ptop_pa, method="nearest").values)
    pbot_sel = float(da["plev"].sel(plev=pbot_pa, method="nearest").values)

    pmin, pmax = sorted([ptop_sel, pbot_sel])

    return da.sel(plev=slice(pmin, pmax))


def clean_extremes(da, max_abs=None, min_val=None, max_val=None):
    out = da.where(np.isfinite(da))
    out = out.where(np.abs(out) < 1e19)

    if max_abs is not None:
        out = out.where(np.abs(out) <= max_abs)

    if min_val is not None:
        out = out.where(out >= min_val)

    if max_val is not None:
        out = out.where(out <= max_val)

    return out


def open_var(path, candidates):
    ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True)

    for v in candidates:
        if v in ds.data_vars:
            return ds[v]

    raise KeyError(
        f"None of {candidates} found in {path}. "
        f"Available: {list(ds.data_vars)}"
    )


def to_mmday(da):
    u = (da.attrs.get("units") or "").lower()

    if ("kg" in u and "m-2" in u and "s-1" in u) or ("kg m-2 s-1" in u) or ("kg/m2/s" in u):
        return da * SEC_PER_DAY

    if ("m s-1" in u) or ("m/s" in u):
        return da * 1000.0 * SEC_PER_DAY

    if ("mm" in u) and (("day" in u) or ("d-1" in u)):
        return da

    mx = float(np.nanmax(np.abs(da.values)))

    if mx < 0.05:
        return da * SEC_PER_DAY

    return da


def prep_common_3d(da):
    da = ensure_lat_lon_names(da)
    da = standardize_lon_180(da)
    da = da.sortby("lat")
    da = to_pa_and_select_range(da, P_TOP_PA, P_BOT_PA)
    da = da.sel(lon=slice(LON0, LON1), lat=slice(LAT0, LAT1))

    return da


def prep_common_2d(da):
    da = ensure_lat_lon_names(da)
    da = standardize_lon_180(da)
    da = da.sortby("lat")
    da = da.sel(lon=slice(LON0, LON1), lat=slice(LAT0, LAT1))

    return da


def seasonal_yearly_mean(da, season):
    da_s = da.where(da["time"].dt.season == season, drop=True)

    if da_s.sizes.get("time", 0) == 0:
        raise ValueError(f"{season} selection returned zero time steps.")

    t = da_s["time"]

    if season == "DJF":
        season_year = xr.where(t.dt.month == 12, t.dt.year + 1, t.dt.year)
    else:
        season_year = t.dt.year

    da_s = da_s.assign_coords(season_year=("time", season_year.data))

    out = da_s.groupby("season_year").mean("time", skipna=True)

    return out.rename({"season_year": "year"})


def area_mean(da, lon0, lon1, lat0, lat1):
    sub = da.sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1))

    w = xr.DataArray(
        np.cos(sub["lat"] * DEG2RAD),
        coords={"lat": sub["lat"]},
        dims=("lat",)
    )

    return sub.weighted(w).mean(("lat", "lon"))


def dq_dx_dq_dy(q):
    q = q.sortby("lon").sortby("lat")

    dq_dlon_deg = q.differentiate("lon")
    dq_dlat_deg = q.differentiate("lat")

    coslat = np.cos(q["lat"] * DEG2RAD)

    dx_per_deg = R_EARTH * coslat * DEG2RAD
    dy_per_deg = R_EARTH * DEG2RAD

    dq_dx = dq_dlon_deg / dx_per_deg
    dq_dy = dq_dlat_deg / dy_per_deg

    return dq_dx, dq_dy


def vint_nantrapz(integrand, min_segments=MIN_VALID_SEGMENTS):
    integrand = integrand.sortby("plev").transpose("plev", "year", "lat", "lon")

    p = integrand["plev"].values.astype(float)
    y = integrand.values

    y0 = y[:-1, ...]
    y1 = y[1:, ...]

    dp = np.diff(p)

    valid = np.isfinite(y0) & np.isfinite(y1)
    nseg = valid.sum(axis=0)

    dp_b = dp.reshape((dp.size,) + (1,) * (y0.ndim - 1))

    contrib = 0.5 * (
        np.where(valid, y0, 0.0) +
        np.where(valid, y1, 0.0)
    ) * dp_b

    integ = contrib.sum(axis=0)
    integ = np.where(nseg >= min_segments, integ, np.nan)

    out = xr.DataArray(
        integ,
        coords={
            "year": integrand["year"],
            "lat": integrand["lat"],
            "lon": integrand["lon"],
        },
        dims=("year", "lat", "lon"),
    )

    return out / G


def compute_yearly_hadv_vadv(hus_path, ua_path, va_path, omega_path, season):
    q = prep_common_3d(open_var(hus_path,   ["hus", "q"]))
    u = prep_common_3d(open_var(ua_path,    ["ua", "u"]))
    v = prep_common_3d(open_var(va_path,    ["va", "v"]))
    w = prep_common_3d(open_var(omega_path, ["wap", "omega", "w"]))

    q = clean_extremes(q, min_val=0.0, max_val=0.2)
    u = clean_extremes(u, max_abs=200.0)
    v = clean_extremes(v, max_abs=200.0)
    w = clean_extremes(w, max_abs=50.0)

    qy = seasonal_yearly_mean(q, season)
    uy = seasonal_yearly_mean(u, season)
    vy = seasonal_yearly_mean(v, season)
    wy = seasonal_yearly_mean(w, season)

    qy, uy, vy, wy = xr.align(qy, uy, vy, wy, join="inner")

    dq_dx, dq_dy = dq_dx_dq_dy(qy)
    dq_dp = qy.sortby("plev").differentiate("plev")

    hadv_int = uy * dq_dx + vy * dq_dy
    vadv_int = wy * dq_dp

    hadv = -vint_nantrapz(hadv_int) * SEC_PER_DAY
    vadv = -vint_nantrapz(vadv_int) * SEC_PER_DAY

    return clean_extremes(hadv, max_abs=1e3), clean_extremes(vadv, max_abs=1e3)


def compute_yearly_vadv_decomp(hus_dust, omega_dust, hus_ctl, omega_ctl, season):
    qd = prep_common_3d(open_var(hus_dust,   ["hus", "q"]))
    wd = prep_common_3d(open_var(omega_dust, ["wap", "omega", "w"]))
    qc = prep_common_3d(open_var(hus_ctl,    ["hus", "q"]))
    wc = prep_common_3d(open_var(omega_ctl,  ["wap", "omega", "w"]))

    qd = clean_extremes(qd, min_val=0.0, max_val=0.2)
    qc = clean_extremes(qc, min_val=0.0, max_val=0.2)
    wd = clean_extremes(wd, max_abs=50.0)
    wc = clean_extremes(wc, max_abs=50.0)

    qdy = seasonal_yearly_mean(qd, season)
    wdy = seasonal_yearly_mean(wd, season)
    qcy = seasonal_yearly_mean(qc, season)
    wcy = seasonal_yearly_mean(wc, season)

    qdy, wdy, qcy, wcy = xr.align(qdy, wdy, qcy, wcy, join="inner")

    qbar = qcy.mean("year", skipna=True)
    wbar = wcy.mean("year", skipna=True)

    qprime = qdy - qcy
    wprime = wdy - wcy

    dqbar_dp   = qbar.sortby("plev").differentiate("plev")
    dqprime_dp = qprime.sortby("plev").differentiate("plev")

    dyn    = -vint_nantrapz(wprime * dqbar_dp)   * SEC_PER_DAY
    thermo = -vint_nantrapz(wbar   * dqprime_dp) * SEC_PER_DAY
    nonlin = -vint_nantrapz(wprime * dqprime_dp) * SEC_PER_DAY

    return dyn, thermo, nonlin


def region_bounds(region):
    if region == "north":
        return BOX_LON0, BOX_LON1, NORTH_LAT0, NORTH_LAT1

    if region == "equatorial":
        return BOX_LON0, BOX_LON1, EQ_LAT0, EQ_LAT1

    raise ValueError("REGION must be 'north' or 'equatorial'.")


def compute_model_vector(model, season, region):
    lon0, lon1, lat0, lat1 = region_bounds(region)

    pr_d = prep_common_2d(open_var(f"pr_{model}_dust.nc", ["pr"]))
    pr_c = prep_common_2d(open_var(f"pr_{model}_ctl.nc",  ["pr"]))

    pr_d = clean_extremes(to_mmday(pr_d), max_abs=1000)
    pr_c = clean_extremes(to_mmday(pr_c), max_abs=1000)

    ev_d = prep_common_2d(open_var(
        f"evaporation_{model}_dust.nc",
        ["evspsbl", "evaporation", "evap", "e"]
    ))

    ev_c = prep_common_2d(open_var(
        f"evaporation_{model}_ctl.nc",
        ["evspsbl", "evaporation", "evap", "e"]
    ))

    ev_d = clean_extremes(to_mmday(ev_d), max_abs=1000)
    ev_c = clean_extremes(to_mmday(ev_c), max_abs=1000)

    pry_d = seasonal_yearly_mean(pr_d, season)
    pry_c = seasonal_yearly_mean(pr_c, season)
    evy_d = seasonal_yearly_mean(ev_d, season)
    evy_c = seasonal_yearly_mean(ev_c, season)

    pry_d, pry_c = xr.align(pry_d, pry_c, join="inner")
    evy_d, evy_c = xr.align(evy_d, evy_c, join="inner")

    Pprime_y = area_mean(pry_d - pry_c, lon0, lon1, lat0, lat1)
    Eprime_y = area_mean(evy_d - evy_c, lon0, lon1, lat0, lat1)

    hadv_d, vadv_d = compute_yearly_hadv_vadv(
        f"hus_{model}_dust.nc",
        f"ua_{model}_dust.nc",
        f"va_{model}_dust.nc",
        f"omega_{model}_dust.nc",
        season
    )

    hadv_c, vadv_c = compute_yearly_hadv_vadv(
        f"hus_{model}_ctl.nc",
        f"ua_{model}_ctl.nc",
        f"va_{model}_ctl.nc",
        f"omega_{model}_ctl.nc",
        season
    )

    hadv_d, hadv_c = xr.align(hadv_d, hadv_c, join="inner")
    vadv_d, vadv_c = xr.align(vadv_d, vadv_c, join="inner")

    HADVprime_y = area_mean(hadv_d - hadv_c, lon0, lon1, lat0, lat1)
    VADVprime_y = area_mean(vadv_d - vadv_c, lon0, lon1, lat0, lat1)

    Res_y = Pprime_y - (Eprime_y + HADVprime_y + VADVprime_y)

    dyn_y, thermo_y, nonlin_y = compute_yearly_vadv_decomp(
        f"hus_{model}_dust.nc",
        f"omega_{model}_dust.nc",
        f"hus_{model}_ctl.nc",
        f"omega_{model}_ctl.nc",
        season
    )

    Dyn = area_mean(dyn_y,    lon0, lon1, lat0, lat1).mean("year")
    Thm = area_mean(thermo_y, lon0, lon1, lat0, lat1).mean("year")
    Nln = area_mean(nonlin_y, lon0, lon1, lat0, lat1).mean("year")

    Pm = Pprime_y.mean("year")
    Em = Eprime_y.mean("year")
    Hm = HADVprime_y.mean("year")
    Vm = VADVprime_y.mean("year")
    Rm = Res_y.mean("year")

    return np.array(
        [
            float(Pm),
            float(Em),
            float(Hm),
            float(Vm),
            float(Rm),
            float(Dyn),
            float(Thm),
            float(Nln),
        ],
        dtype=float
    )


# ============================================================
# 4x8 FIGURE
# ============================================================
def make_4x8_allseasons(region=REGION, add_mme=True):
    model_cols = MODELS.copy()

    if add_mme:
        model_cols = model_cols + ["mme"]

    nrows = len(SEASONS)
    ncols = len(model_cols)

    # compute all season/model vectors
    vals = {s: {} for s in SEASONS}

    for s in SEASONS:
        ok = []

        for m in MODELS:
            try:
                vals[s][m] = compute_model_vector(m, s, region)
                ok.append(m)

            except Exception as e:
                print(f"[WARN] skipping {m} ({s}): {e}")
                vals[s][m] = np.full(8, np.nan)

        if add_mme:
            good = [
                vals[s][m]
                for m in ok
                if np.all(np.isfinite(vals[s][m]))
            ]

            vals[s]["mme"] = (
                np.nanmean(np.vstack(good), axis=0)
                if len(good)
                else np.full(8, np.nan)
            )

    xtxt = [
        r"$\Delta P$",
        r"$\Delta E$",
        r"$-\Delta\langle\mathbf{V}_h\cdot\nabla q\rangle$",
        r"$-\Delta\langle\omega\,\partial_p q\rangle$",
        r"$\Delta\mathrm{Res}$",
        r"$-\langle\Delta\omega\,\partial_p\bar{q}\rangle$",
        r"$-\langle\bar{\omega}\,\partial_p\Delta q\rangle$",
        r"$-\langle\Delta\omega\,\partial_p\Delta q\rangle$",
    ]
    
 
    colors = [
        "red",
        "green",
        "#8ecae6",
        "#f4b6c2",
        "magenta",
        "yellow",
        "black",
        "orange",
    ]

    legend_labels = [
        r"$\Delta P$",
        r"$\Delta E$",
        r"$-\Delta\langle\mathbf{V}_h\cdot\nabla q\rangle$",
        r"$-\Delta\langle\omega\,\partial_p q\rangle$",
        r"$\Delta\mathrm{Res}$",
        r"$-\langle\Delta\omega\,\partial_p\bar{q}\rangle$",
        r"$-\langle\bar{\omega}\,\partial_p\Delta q\rangle$",
        r"$-\langle\Delta\omega\,\partial_p\Delta q\rangle$",
    ]
    
    

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.35 * ncols + 2.0, 2.2 * nrows + 1.5),
        sharex=False,
        sharey=False,
    )

    axes = np.atleast_2d(axes)
    x = np.arange(8)

    for ri, season in enumerate(SEASONS):
        yabs = SEASON_YLIM.get(season, None)

        if yabs is None:
            allv = np.vstack([vals[season][m] for m in model_cols])
            yabs = float(np.nanpercentile(np.abs(allv), 98))
            yabs = yabs if np.isfinite(yabs) and yabs > 0 else 0.5

        for ci, model in enumerate(model_cols):
            ax = axes[ri, ci]
            v = vals[season][model]

            ax.bar(
                x,
                v,
                color=colors,
                edgecolor="k",
                linewidth=0.3,
            )

            ax.axhline(0, color="k", lw=0.7)
            ax.axvline(4.5, color="k", lw=0.7, ls="--")
            ax.set_ylim(-yabs, yabs)

            # column titles only at top
            if ri == 0:
                ax.set_title(
                    MODEL_TITLES.get(model, model.upper()),
                    fontsize=15
                )

            # season label on left side of row
            if ci == 0:
                ax.text(
                    -0.30,
                    0.5,
                    season,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                )
                ax.set_ylabel("")

            # x-ticks only on bottom row
            if ri == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(
                    xtxt,
                    rotation=65,
                    ha="right",
                    fontsize=11
                )
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=8)

    handles = [
        Patch(color=colors[i], label=legend_labels[i])
        for i in range(8)
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=24,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])

    plt.savefig(
        OUTPNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Saved:", OUTPNG)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    os.chdir(ROOT)
    make_4x8_allseasons(region=REGION, add_mme=ADD_MME)
