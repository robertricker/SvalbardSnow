import pandas as pd
import geopandas as gpd
import glob
import os
import numpy as np
from collect_data.icesat2_heights import Icesat2Heights
from collect_data.dem import ArcticDem
from io_tools import create_polygon
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d


def get_glacier_inv(file, out_epsg):
    data = gpd.read_file(file)
    return data[['geometry']].to_crs(out_epsg)


def get_coastline(file, out_epsg):
    data = gpd.read_file(file)
    return data[['geometry']].to_crs(out_epsg)


def collect_data(config):
    t0 = config['proc_step_options']['collect']['t0']
    t1 = config['proc_step_options']['collect']['t1']
    spatial_extent = config['proc_step_options']['collect']['spatial_extent']
    atl03_dir = config['dir']['icesat2']['atl03']
    dem_dir = config['dir']['dem']['arctic_dem']
    out_epsg = config['options']['out_epsg']

    aoi_gdf = gpd.GeoDataFrame({'geometry': [create_polygon(spatial_extent)]}, crs='epsg:4326').to_crs(out_epsg)

    atl03 = Icesat2Heights(t0=t0, t1=t1, spatial_extent=spatial_extent, local_dir=atl03_dir, out_epsg=out_epsg)
    #atl03.get_atl03()
    atl03.get_file_list()
    atl03.get_file_dates()

    dem = ArcticDem(host_dir=dem_dir['host_dir'], local_dir=dem_dir['local_dir'], tile_idx_file=dem_dir['tile_idx'])
    dem.collect_in_aoi(aoi_gdf)
    dem.get_data()

    glacier_inv = get_glacier_inv(config['dir']['auxiliary']['glacier_inv'], out_epsg)
    coastline = get_coastline(config['dir']['auxiliary']['coastline'], out_epsg)

    running_date = datetime.strptime(t0, '%Y-%m-%d')
    dt = timedelta(days=1)
    while running_date <= datetime.strptime(t1, '%Y-%m-%d'):
        atl03.get_target_files(running_date, running_date+dt)
        if atl03.target_files:
            gdf_list = list()
            print(running_date)
            print(atl03.target_files)
            for tile_file in dem.target_files:
                print(tile_file)
                f_rgi = dem.data[os.path.basename(tile_file)]['f_rgi']
                dem_extent = dem.data[os.path.basename(tile_file)]['extent']
                tmp = atl03.atl03_to_gdf(aoi=dem_extent, coast=coastline, glaciers=glacier_inv)
                if len(tmp) != 0:
                    tmp['dem_h'] = f_rgi((np.array(tmp.geometry.x), np.array(tmp.geometry.y)))
                    gdf_list.append(tmp)
            if len(gdf_list) != 0:
                gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame)
                gdf_final.crs = gdf_list[0].crs

                gdf_final = gdf_final.sort_values(by='time').reset_index(drop=True)
                gdf_final['distance'] = gdf_final.groupby(['orbit_number', 'beam'])['geometry'].shift().distance(
                    gdf_final['geometry']).fillna(0)
                h_diff = gdf_final.groupby(['orbit_number', 'beam'])['dem_h'].diff()
                gdf_final['h_grad'] = gaussian_filter1d(
                    h_diff.values, sigma=10) / gaussian_filter1d(gdf_final['distance'].values, sigma=10)
                gdf_final['distance'] = gdf_final.groupby(['orbit_number', 'beam'])['distance'].cumsum()
                gdf_final = gdf_final.dropna().reset_index(drop=True)

                if len(gdf_final) != 0:
                    gdf_final.to_file(
                        config['dir']['product']['geojson'] +
                        'test_' + running_date.strftime('%Y%m%d') +
                        '.geojson', driver="GeoJSON")
        running_date += dt
