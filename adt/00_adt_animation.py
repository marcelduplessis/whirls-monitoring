#!/usr/bin/env python3
"""
Create an ADT/SLA regional animation with daily eddy contours and glider tracks.

Framework adapted from the GOFLOW regional animation script so map styling,
figure sizing, time-stepped animation, and platform overlays are consistent.
"""

from pathlib import Path
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import requests
import xarray as xr


DEFAULT_ADT_BASE = (
	"/home/mduplessis/share/www/data/adt/SEALEVEL_GLO_PHY_L4_NRT_008_046/"
	"cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D_202506/2026"
)
DEFAULT_EDDY_BASE = "https://thredds-x.ipsl.fr/thredds/fileServer/WHIRLS/PRODUCTS/TOEDDIES"
DEFAULT_SG_CSV = "/home/mduplessis/share/www/data/sg267_WHIRLS_Mission3_2026/sg267_mission3_track.csv"
DEFAULT_WG_NC = "/home/mduplessis/share/gliders/waveglider/wg1169/wg1169_WHIRLS_Mission3_L1.nc"


def parse_geojson_polygons(gj):
	polygons = []
	for feat in gj.get("features", []):
		geom = feat.get("geometry", {})
		gtype = geom.get("type")
		coords = geom.get("coordinates", [])

		if gtype == "Polygon":
			for ring in coords:
				ring = np.asarray(ring)
				if ring.size > 0:
					polygons.append((ring[:, 0], ring[:, 1]))
		elif gtype == "MultiPolygon":
			for poly in coords:
				for ring in poly:
					ring = np.asarray(ring)
					if ring.size > 0:
						polygons.append((ring[:, 0], ring[:, 1]))
	return polygons


def load_eddy_data(dates, base_url=DEFAULT_EDDY_BASE):
	session = requests.Session()
	session.headers.update({"User-Agent": "python-requests"})

	eddy_data = {}
	for d in pd.to_datetime(dates):
		date_str = d.strftime("%Y-%m-%d")
		cyclone_url = f"{base_url}/{date_str}_cyclones_outcontour.geojson"
		anticyclone_url = f"{base_url}/{date_str}_anticyclones_outcontour.geojson"

		day_data = {"cyclones": [], "anticyclones": []}

		try:
			r = session.get(cyclone_url, timeout=60)
			r.raise_for_status()
			day_data["cyclones"] = parse_geojson_polygons(r.json())
		except requests.HTTPError:
			print(f"Missing cyclone file for {date_str}")
		except Exception as exc:
			print(f"Error reading cyclone file for {date_str}: {exc}")

		try:
			r = session.get(anticyclone_url, timeout=60)
			r.raise_for_status()
			day_data["anticyclones"] = parse_geojson_polygons(r.json())
		except requests.HTTPError:
			print(f"Missing anticyclone file for {date_str}")
		except Exception as exc:
			print(f"Error reading anticyclone file for {date_str}: {exc}")

		eddy_data[date_str] = day_data

	return eddy_data


def load_glider_track(csv_path, lon_bounds=None, lat_bounds=None):
	csv_path = Path(csv_path)
	if not csv_path.exists():
		raise FileNotFoundError(f"Glider track CSV not found: {csv_path}")

	track = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
	track = np.atleast_1d(track)
	if track.size == 0:
		raise ValueError(f"Glider track CSV is empty: {csv_path}")

	times = np.array([np.datetime64(t, "ns") for t in track["time"]], dtype="datetime64[ns]")
	lons = np.asarray(track["longitude"], dtype=float)
	lats = np.asarray(track["latitude"], dtype=float)

	valid = np.isfinite(lons) & np.isfinite(lats)
	if lon_bounds is not None:
		lon_lo, lon_hi = sorted(lon_bounds)
		valid &= (lons >= lon_lo) & (lons <= lon_hi)
	if lat_bounds is not None:
		lat_lo, lat_hi = sorted(lat_bounds)
		valid &= (lats >= lat_lo) & (lats <= lat_hi)

	times = times[valid]
	lons = lons[valid]
	lats = lats[valid]
	if times.size == 0:
		raise ValueError("No valid seaglider points remain after filtering.")

	order = np.argsort(times)
	return {"time": times[order], "lon": lons[order], "lat": lats[order]}


def _extract_datetime64_from_dataarray(time_da):
	values = np.asarray(time_da.values)
	if np.issubdtype(values.dtype, np.datetime64):
		return values.astype("datetime64[ns]")

	units = str(time_da.attrs.get("units", "")).lower()
	match = re.match(r"^(seconds|minutes|hours|days) since (.+)$", units)
	if not match:
		raise ValueError(
			"Unsupported NetCDF time format. Expected datetime64 values or CF units like 'seconds since ...'."
		)

	unit_key = match.group(1)
	base_time = np.datetime64(match.group(2).strip(), "ns")
	values = np.asarray(values, dtype=float)

	if unit_key == "seconds":
		delta = (values * 1e9).astype("timedelta64[ns]")
	elif unit_key == "minutes":
		delta = (values * 60 * 1e9).astype("timedelta64[ns]")
	elif unit_key == "hours":
		delta = (values * 3600 * 1e9).astype("timedelta64[ns]")
	else:
		delta = (values * 86400 * 1e9).astype("timedelta64[ns]")

	return base_time + delta


def load_waveglider_track(nc_path, lon_bounds=None, lat_bounds=None):
	nc_path = Path(nc_path)
	if not nc_path.exists():
		raise FileNotFoundError(f"Wave glider NetCDF not found: {nc_path}")

	ds = xr.open_dataset(nc_path)
	time_name = "time" if "time" in ds.variables else None
	lon_name = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
	lat_name = "latitude" if "latitude" in ds.variables else ("lat" if "lat" in ds.variables else None)

	if time_name is None or lon_name is None or lat_name is None:
		raise KeyError(
			"Could not find required variables in wave glider NetCDF. "
			"Expected time + (longitude or lon) + (latitude or lat)."
		)

	times = _extract_datetime64_from_dataarray(ds[time_name])
	lons = np.asarray(ds[lon_name].values, dtype=float)
	lats = np.asarray(ds[lat_name].values, dtype=float)

	if lons.ndim > 1:
		lons = np.ravel(lons)
	if lats.ndim > 1:
		lats = np.ravel(lats)
	if times.ndim > 1:
		times = np.ravel(times)

	valid = np.isfinite(lons) & np.isfinite(lats)
	if lon_bounds is not None:
		lon_lo, lon_hi = sorted(lon_bounds)
		valid &= (lons >= lon_lo) & (lons <= lon_hi)
	if lat_bounds is not None:
		lat_lo, lat_hi = sorted(lat_bounds)
		valid &= (lats >= lat_lo) & (lats <= lat_hi)

	times = times[valid]
	lons = lons[valid]
	lats = lats[valid]
	if times.size == 0:
		raise ValueError("No valid wave glider points remain after filtering.")

	order = np.argsort(times)
	return {"time": times[order], "lon": lons[order], "lat": lats[order]}


def load_adt_dataset(base_dir, months, start_date, end_date, lon_bounds, lat_bounds):
	base = Path(base_dir)
	nc_files = []
	for mm in months:
		month_dir = base / f"{int(mm):02d}"
		if not month_dir.exists():
			continue
		nc_files.extend(
			f for f in month_dir.glob("nrt_global_allsat_phy_l4_*.nc") if "(" not in f.name
		)

	nc_files = sorted(nc_files)
	if not nc_files:
		raise FileNotFoundError(f"No ADT files found in {base} for months {months}")

	ds = xr.open_mfdataset([str(p) for p in nc_files], combine="by_coords").sortby("time")
	ds = ds.sel(time=slice(start_date, end_date))

	lat_name = "latitude" if "latitude" in ds.coords else "lat"
	lon_name = "longitude" if "longitude" in ds.coords else "lon"

	lat_lo, lat_hi = sorted(lat_bounds)
	lon_lo, lon_hi = sorted(lon_bounds)
	ds = ds.sel({lat_name: slice(lat_lo, lat_hi), lon_name: slice(lon_lo, lon_hi)})

	if ds.sizes.get("time", 0) == 0:
		raise ValueError("ADT subset contains no timesteps after date/bounds filtering.")

	return ds, lon_name, lat_name


def make_animation(
	ds,
	lon_name,
	lat_name,
	eddy_data,
	output_path,
	lon_bounds,
	lat_bounds,
	sla_var="sla",
	adt_var="adt",
	adt_levels=(-0.7, 0.7),
	cmap="RdBu_r",
	fps=10,
	interval=120,
	dpi=200,
	glider_track=None,
	waveglider_track=None,
):
	if sla_var not in ds:
		raise KeyError(f"SLA variable '{sla_var}' not found in dataset")
	if adt_var not in ds:
		raise KeyError(f"ADT variable '{adt_var}' not found in dataset")

	sla = ds[sla_var]
	adt = ds[adt_var]
	vmax = float(np.nanpercentile(np.abs(sla.values), 98))

	proj = ccrs.PlateCarree()
	fig = plt.figure(figsize=(14, 8), constrained_layout=True)
	ax = plt.axes(projection=proj)
	ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())

	fig.patch.set_facecolor("#f8f9fa")
	ax.set_facecolor("#f8f9fa")
	ax.coastlines(resolution="10m", linewidth=0.8, zorder=10)
	ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
	ax.add_feature(cfeature.RIVERS, edgecolor="white", linewidth=0.6, zorder=11)

	ax.plot(
		[0.88], [0.96], marker="o", markersize=5, markerfacecolor="gold",
		markeredgecolor="black", markeredgewidth=0.8, linestyle="None",
		transform=ax.transAxes, zorder=30,
	)
	ax.text(0.90, 0.96, "SG Koeksister", transform=ax.transAxes, va="center", ha="left", fontsize=11, zorder=30)
	ax.plot(
		[0.88], [0.92], marker="s", markersize=5, markerfacecolor="peru",
		markeredgecolor="black", markeredgewidth=0.8, linestyle="None",
		transform=ax.transAxes, zorder=30,
	)
	ax.text(0.90, 0.92, "WG Melktert", transform=ax.transAxes, va="center", ha="left", fontsize=11, zorder=30)

	gl = ax.gridlines(draw_labels=True, linewidth=0.75, color="gray", alpha=1, linestyle="--", zorder=15)
	gl.top_labels = False
	gl.right_labels = False

	pcm = ax.pcolormesh(
		ds[lon_name],
		ds[lat_name],
		sla.isel(time=0),
		transform=ccrs.PlateCarree(),
		cmap=cmap,
		vmin=-vmax,
		vmax=vmax,
		shading="auto",
		zorder=1,
	)

	contours = ax.contour(
		ds[lon_name],
		ds[lat_name],
		adt.isel(time=0),
		levels=list(adt_levels),
		colors="k",
		linewidths=1.0,
		transform=ccrs.PlateCarree(),
		zorder=12,
	)

	cbar = plt.colorbar(pcm, ax=ax, pad=0.02, aspect=30)
	cbar.set_label("SLA")
	title = ax.set_title(
		f"ADT/SLA {np.datetime_as_string(ds.time.values[0], unit='D')}",
		bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
	)

	def draw_eddy_lines(frame_date):
		lines = []
		cyclones = eddy_data.get(frame_date, {}).get("cyclones", [])
		anticyclones = eddy_data.get(frame_date, {}).get("anticyclones", [])

		for lon_poly, lat_poly in cyclones:
			line, = ax.plot(
				lon_poly,
				lat_poly,
				color="tab:blue",
				linewidth=1.0,
				transform=ccrs.PlateCarree(),
				zorder=13,
			)
			lines.append(line)

		for lon_poly, lat_poly in anticyclones:
			line, = ax.plot(
				lon_poly,
				lat_poly,
				color="tab:red",
				linewidth=1.0,
				transform=ccrs.PlateCarree(),
				zorder=13,
			)
			lines.append(line)
		return lines

	eddy_lines = draw_eddy_lines(np.datetime_as_string(ds.time.values[0], unit="D"))

	sg_line = sg_latest = sg_time = sg_lon = sg_lat = None
	if glider_track is not None:
		sg_time = np.asarray(glider_track["time"], dtype="datetime64[ns]")
		sg_lon = np.asarray(glider_track["lon"], dtype=float)
		sg_lat = np.asarray(glider_track["lat"], dtype=float)
		sg_line, = ax.plot([], [], color="gold", linewidth=2.0, transform=ccrs.PlateCarree(), zorder=20)
		sg_latest, = ax.plot(
			[], [], marker="o", markersize=7, markerfacecolor="gold", markeredgecolor="black",
			markeredgewidth=1.0, linestyle="None", transform=ccrs.PlateCarree(), zorder=21,
		)

	wg_line = wg_latest = wg_time = wg_lon = wg_lat = None
	if waveglider_track is not None:
		wg_time = np.asarray(waveglider_track["time"], dtype="datetime64[ns]")
		wg_lon = np.asarray(waveglider_track["lon"], dtype=float)
		wg_lat = np.asarray(waveglider_track["lat"], dtype=float)
		wg_line, = ax.plot([], [], color="#B87333", linewidth=2.0, transform=ccrs.PlateCarree(), zorder=22)
		wg_latest, = ax.plot(
			[], [], marker="s", markersize=7, markerfacecolor="#B87333", markeredgecolor="black",
			markeredgewidth=1.0, linestyle="None", transform=ccrs.PlateCarree(), zorder=23,
		)

	def update(frame):
		nonlocal pcm, contours, eddy_lines

		pcm.remove()
		pcm = ax.pcolormesh(
			ds[lon_name],
			ds[lat_name],
			sla.isel(time=frame),
			transform=ccrs.PlateCarree(),
			cmap=cmap,
			vmin=-vmax,
			vmax=vmax,
			shading="auto",
			zorder=1,
		)

		contours.remove()
		contours = ax.contour(
			ds[lon_name],
			ds[lat_name],
			adt.isel(time=frame),
			levels=list(adt_levels),
			colors="k",
			linewidths=1.0,
			transform=ccrs.PlateCarree(),
			zorder=12,
		)

		for line in eddy_lines:
			line.remove()
		frame_date = np.datetime_as_string(ds.time.values[frame], unit="D")
		eddy_lines = draw_eddy_lines(frame_date)

		frame_time = np.datetime64(ds.time.values[frame], "ns")

		if sg_time is not None:
			n_track = np.searchsorted(sg_time, frame_time, side="right")
			if n_track > 0:
				sg_line.set_data(sg_lon[:n_track], sg_lat[:n_track])
				sg_latest.set_data([sg_lon[n_track - 1]], [sg_lat[n_track - 1]])
			else:
				sg_line.set_data([], [])
				sg_latest.set_data([], [])

		if wg_time is not None:
			n_track = np.searchsorted(wg_time, frame_time, side="right")
			if n_track > 0:
				wg_line.set_data(wg_lon[:n_track], wg_lat[:n_track])
				wg_latest.set_data([wg_lon[n_track - 1]], [wg_lat[n_track - 1]])
			else:
				wg_line.set_data([], [])
				wg_latest.set_data([], [])

		title.set_text(f"ADT/SLA {frame_date}")

		artists = [pcm, title, contours]
		artists.extend(eddy_lines)
		if sg_line is not None and sg_latest is not None:
			artists.extend([sg_line, sg_latest])
		if wg_line is not None and wg_latest is not None:
			artists.extend([wg_line, wg_latest])
		return tuple(artists)

	ani = FuncAnimation(
		fig,
		update,
		frames=ds.sizes["time"],
		interval=interval,
		blit=False,
		repeat=True,
	)

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	ani.save(
		output_path,
		writer="pillow",
		fps=fps,
		dpi=dpi,
		savefig_kwargs={"transparent": False, "facecolor": "#f8f9fa", "edgecolor": "none"},
	)
	plt.close(fig)
	print(f"Saved animation to {output_path}")


def parse_args():
	parser = argparse.ArgumentParser(description="Create ADT animation with eddies and glider overlays.")
	parser.add_argument("--adt-base-dir", default=DEFAULT_ADT_BASE, help="Base directory containing month subfolders.")
	parser.add_argument("--months", nargs="+", type=int, default=[5, 6], help="Month folders to scan under adt-base-dir.")
	parser.add_argument("--start-date", default="2026-05-15", help="Start date (YYYY-MM-DD).")
	parser.add_argument("--end-date", default="2026-12-31", help="End date (YYYY-MM-DD).")
	parser.add_argument("--lon-min", type=float, default=5)
	parser.add_argument("--lon-max", type=float, default=30)
	parser.add_argument("--lat-min", type=float, default=-42)
	parser.add_argument("--lat-max", type=float, default=-30)
	parser.add_argument("--sla-var", default="sla", help="SLA variable name in dataset.")
	parser.add_argument("--adt-var", default="adt", help="ADT variable name in dataset.")
	parser.add_argument("--adt-levels", nargs=2, type=float, default=[-0.7, 0.7], help="ADT contour levels.")
	parser.add_argument("--cmap", default="RdBu_r", help="Colormap for SLA panel.")
	parser.add_argument("--fps", type=int, default=10)
	parser.add_argument("--interval", type=int, default=120, help="Animation interval in ms.")
	parser.add_argument("--dpi", type=int, default=200)
	parser.add_argument("--eddy-base-url", default=DEFAULT_EDDY_BASE, help="Base URL for daily eddy geojsons.")
	parser.add_argument("--output", default="/home/mduplessis/share/www/html/img/sla_animation_latest.gif", help="Output GIF path.")
	parser.add_argument("--glider-track-csv", default=DEFAULT_SG_CSV, help="CSV with time, longitude, latitude.")
	parser.add_argument("--no-glider-track", action="store_true", help="Disable seaglider overlay.")
	parser.add_argument("--waveglider-track-nc", default=DEFAULT_WG_NC, help="Wave glider track NetCDF.")
	parser.add_argument("--no-waveglider-track", action="store_true", help="Disable wave glider overlay.")
	return parser.parse_args()


def main():
	args = parse_args()
	lon_bounds = (args.lon_min, args.lon_max)
	lat_bounds = (args.lat_min, args.lat_max)

	ds, lon_name, lat_name = load_adt_dataset(
		base_dir=args.adt_base_dir,
		months=args.months,
		start_date=args.start_date,
		end_date=args.end_date,
		lon_bounds=lon_bounds,
		lat_bounds=lat_bounds,
	)

	frame_dates = pd.to_datetime(ds.time.values).normalize().unique()
	eddy_data = load_eddy_data(frame_dates, base_url=args.eddy_base_url)

	glider_track = None
	waveglider_track = None

	if not args.no_glider_track:
		glider_track = load_glider_track(
			args.glider_track_csv,
			lon_bounds=lon_bounds,
			lat_bounds=lat_bounds,
		)
	if not args.no_waveglider_track:
		waveglider_track = load_waveglider_track(
			args.waveglider_track_nc,
			lon_bounds=lon_bounds,
			lat_bounds=lat_bounds,
		)

	make_animation(
		ds=ds,
		lon_name=lon_name,
		lat_name=lat_name,
		eddy_data=eddy_data,
		output_path=args.output,
		lon_bounds=lon_bounds,
		lat_bounds=lat_bounds,
		sla_var=args.sla_var,
		adt_var=args.adt_var,
		adt_levels=args.adt_levels,
		cmap=args.cmap,
		fps=args.fps,
		interval=args.interval,
		dpi=args.dpi,
		glider_track=glider_track,
		waveglider_track=waveglider_track,
	)


if __name__ == "__main__":
	main()
