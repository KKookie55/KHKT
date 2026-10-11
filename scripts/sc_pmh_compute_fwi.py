#!/usr/bin/env python3
"""
SC-PMH Step 2: Compute Canadian Fire Weather Index (CFFWIS) for all 273 cells.
Adds 6 columns: fwi_dc, fwi_dmc, fwi_ffmc, fwi_isi, fwi_bui, fwi
Saves to /home/z/my-project/download/sc_pmh/panel_with_fwi.parquet

Algorithm: xclim.indices.fire.cffwis_indices (Wang et al. 2015 revision)
Reference: Van Wagner (1987), Canadian Forest Service standard
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from xclim.indices.fire import cffwis_indices

DL = Path('/home/z/my-project/download')
SC = DL / 'sc_pmh'
SC.mkdir(exist_ok=True)
PANEL_P = DL / 'v871' / 'panel_v871.parquet'
OUT_P = SC / 'panel_with_fwi.parquet'

print("[1/4] loading panel...")
panel = pd.read_parquet(PANEL_P)
panel['date'] = pd.to_datetime(panel['date'])
print(f"  shape={panel.shape}, dates={panel.date.min().date()}..{panel.date.max().date()}")
cells = panel[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).reset_index(drop=True)
print(f"  n_cells={len(cells)}")

# Convert to xarray DataArrays indexed by (cell, date)
# Need to add 'cell_id' as a coordinate
print("[2/4] pivoting to xarray (cell, date)...")
panel2 = panel.merge(cells.reset_index().rename(columns={'index': 'cell_id'}), on=['lat', 'lon'], how='left')
panel2 = panel2.set_index(['cell_id', 'date']).sort_index()

# xclim expects DataArray with (time, spatial) dims; here (date, cell)
# Stack: tas(time, cell)
def to_da(col, name, units):
    """Pivot to (date, cell) DataArray"""
    df = panel2[col].unstack('cell_id')  # rows=date, cols=cell_id
    da = xr.DataArray(
        df.values,
        coords={'time': df.index.values, 'cell': np.arange(len(df.columns))},
        dims=('time', 'cell'),
        name=name,
    )
    da.attrs.update({'units': units, 'standard_name': name})
    return da

tas = to_da('t2m', 'tas', 'degC')        # temperature (we have t2m in °C already, since NASA POWER provides °C)
pr = to_da('precip', 'pr', 'mm d-1')       # precipitation mm/day
sfcWind = to_da('wind', 'sfcWind', 'm s-1')  # wind — note: xclim expects km/h, will convert
hurs = to_da('rh', 'hurs', '%')
lat = xr.DataArray(
    cells.lat.values,
    coords={'cell': np.arange(len(cells))},
    dims=('cell',),
    name='lat',
)
lat.attrs.update({'units': 'degrees_north', 'standard_name': 'latitude'})

# Convert wind m/s -> km/h (FWI standard unit)
sfcWind_kmh = sfcWind * 3.6
sfcWind_kmh.attrs.update({'units': 'km h-1', 'standard_name': 'wind_speed'})
sfcWind_kmh.name = 'sfcWind'

print(f"  tas shape={tas.shape}, pr shape={pr.shape}")

# xclim expects tas in K or °C? Default attrs say 'degC' is fine but check
# Actually xclim expects tas in K by default, with units attribute. Let me check.
print(f"  tas.attrs: {tas.attrs}")
print(f"  tas min/max: {float(tas.min()):.2f} / {float(tas.max()):.2f}")  # if these are around 0-50 they're degC

# Convert to Kelvin for xclim (it does unit conversion via pint)
tas_k = tas + 273.15
tas_k.attrs.update({'units': 'K', 'standard_name': 'air_temperature'})
tas_k.name = 'tas'

print("[3/4] computing CFFWIS indices (this is the slow step)...")
t0 = time.time()
# Disable fire season computation (season_method=None) for tropical Vietnam — fires happen year-round
# Initial values: default (start from typical values)
DC, DMC, FFMC, ISI, BUI, FWI = cffwis_indices(
    tas=tas_k,
    pr=pr,
    sfcWind=sfcWind_kmh,
    hurs=hurs,
    lat=lat,
    season_method=None,  # no season mask — Vietnam has no real winter shutdown
    initial_start_up=True,
)
print(f"  done in {time.time()-t0:.1f}s")
print(f"  FWI mean={float(FWI.mean()):.2f}, max={float(FWI.max()):.2f}")
print(f"  FFMC mean={float(FFMC.mean()):.2f}")
print(f"  DMC mean={float(DMC.mean()):.2f}")
print(f"  DC mean={float(DC.mean()):.2f}")
print(f"  ISI mean={float(ISI.mean()):.2f}")
print(f"  BUI mean={float(BUI.mean()):.2f}")

# Convert back to long-format DataFrame
print("[4/4] joining back to panel...")
def to_long(da, col_name):
    """Convert DataArray to (cell_id, date) -> value series"""
    # xclim may return (cell, time) or (time, cell) — normalize
    if da.dims[0] == 'cell':
        df = pd.DataFrame(da.values.T, index=da.time.values, columns=da.cell.values)
    else:
        df = pd.DataFrame(da.values, index=da.time.values, columns=da.cell.values)
    df.index.name = 'date'
    df = df.stack().reset_index()
    df.columns = ['date', 'cell_id', col_name]
    return df.set_index(['cell_id', 'date'])[col_name]

fwi_long = to_long(FWI, 'fwi')
ffmc_long = to_long(FFMC, 'fwi_ffmc')
dmc_long = to_long(DMC, 'fwi_dmc')
dc_long = to_long(DC, 'fwi_dc')
isi_long = to_long(ISI, 'fwi_isi')
bui_long = to_long(BUI, 'fwi_bui')

# merge back
panel_fwi = panel2.copy()
panel_fwi['fwi'] = fwi_long
panel_fwi['fwi_ffmc'] = ffmc_long
panel_fwi['fwi_dmc'] = dmc_long
panel_fwi['fwi_dc'] = dc_long
panel_fwi['fwi_isi'] = isi_long
panel_fwi['fwi_bui'] = bui_long
panel_fwi = panel_fwi.reset_index()
panel_fwi = panel_fwi.drop(columns=['cell_id'])

# Save
panel_fwi.to_parquet(OUT_P, index=False)
print(f"\n[done] saved {OUT_P} shape={panel_fwi.shape}")
print(f"  new columns: fwi, fwi_ffmc, fwi_dmc, fwi_dc, fwi_isi, fwi_bui")
print(f"\nFWI stats:")
print(panel_fwi[['fwi', 'fwi_ffmc', 'fwi_dmc', 'fwi_dc', 'fwi_isi', 'fwi_bui']].describe().round(2).to_string())
