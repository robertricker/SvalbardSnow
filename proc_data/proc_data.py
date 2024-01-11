import pandas as pd
import geopandas as gpd
import shapely as shp
import glob
import os
import numpy as np
from proc_data.icesat2_heights import Icesat2Heights
from proc_data.dem import ArcticDem
from io_tools import create_polygon
from io_tools import calculate_min_distance
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import Rbf


def define_grid(bounds, n_cells, epsg):
    xmin, xmax = bounds[0], bounds[2]
    ymin, ymax = bounds[1], bounds[3]
    cell_size = (xmax - xmin) / n_cells
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):
            # bounds
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append(shp.geometry.box(x0, y0, x1, y1))
    return gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=epsg), cell_size


def grid_data(gdf, grid, var, var_str, hist_n_bins=None, hist_range=None, fill_nan=False, agg_mode=None):

    if agg_mode is None:
        agg_mode = ['mean', 'std']
    tmp_grid = grid.copy()
    merged = gpd.sjoin(gdf[var + ['geometry']].copy(), grid, how='left', predicate='within')
    if 'mean' in agg_mode:
        dissolve_mean = merged.dissolve(by='index_right', aggfunc=np.mean)
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_mean.index, var_str[i]] = dissolve_mean[var[i]].values
    if 'std' in agg_mode:
        dissolve_std = merged.dissolve(by='index_right', aggfunc=np.std)
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_std.index, var_str[i] + '_std'] = dissolve_std[var[i]].values
    if 'sum' in agg_mode:
        dissolve_sum = merged.dissolve(by='index_right', aggfunc=np.sum)
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_sum.index, var_str[i] + '_sum'] = dissolve_sum[var[i]].values
    if 'cnt' in agg_mode:
        dissolve_cnt = merged.dissolve(by='index_right', aggfunc='count')
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_cnt.index, var_str[i] + '_cnt'] = dissolve_cnt[var[i]].values
    if 'hist' in agg_mode:
        dissolve_hist = merged.dissolve(by='index_right', aggfunc=np.mean)
        dissolve_hist.reset_index(inplace=True)
        dissolve_hist[var[0]] = ''
        for i in range(0, len(var)):
            for j in dissolve_hist.index_right.unique():
                tmp = merged[merged.index_right == j].drop(['index_right', 'geometry'], axis=1).reset_index(drop=True)
                tmp_hist = np.histogram(np.array(tmp[var[i]]),
                                        bins=hist_n_bins,
                                        range=(hist_range[0], hist_range[1]))[0]
                dissolve_hist[var[i]][dissolve_hist.index[dissolve_hist.index_right == j][0]] = ' '.join(
                    map(str, tmp_hist))

            tmp_grid.loc[dissolve_hist.index_right, var_str[i] + '_hist'] = dissolve_hist[var[i]].values

    if not fill_nan:
        tmp_grid = tmp_grid.dropna()

    centroidseries = tmp_grid['geometry'].centroid
    tmp_grid['x'], tmp_grid['y'] = centroidseries.x, centroidseries.y
    tmp_grid = tmp_grid.set_index(['x', 'y'])
    return tmp_grid


def get_glacier_inv(file, out_epsg):
    data = gpd.read_file(file)
    return data[['geometry']].to_crs(out_epsg)


def get_coastline(file, out_epsg):
    data = gpd.read_file(file)
    return data[['geometry']].to_crs(out_epsg)


def is2proc(config):
    t0 = config['proc_step_options']['collect']['t0']
    t1 = config['proc_step_options']['collect']['t1']
    spatial_extent = config['proc_step_options']['collect']['spatial_extent']
    atl03_dir = config['dir']['icesat2']['atl03']
    dem_dir = config['dir']['dem']['arctic_dem']
    out_epsg = config['options']['out_epsg']

    aoi_gdf = gpd.GeoDataFrame({'geometry': [create_polygon(spatial_extent)]}, crs='epsg:4326').to_crs(out_epsg)

    atl03 = Icesat2Heights(t0=t0, t1=t1, spatial_extent=spatial_extent, local_dir=atl03_dir, out_epsg=out_epsg)
    # atl03.get_atl03()
    atl03.get_file_list()
    atl03.get_file_dates()

    dem = ArcticDem(host_dir=dem_dir['host_dir'], local_dir=dem_dir['local_dir'], tile_idx_file=dem_dir['tile_idx'])
    dem.collect_in_aoi(aoi_gdf)
    dem.get_data()

    glacier_inv = get_glacier_inv(config['dir']['auxiliary']['glacier_inv'], out_epsg)
    buffered_glaciers = glacier_inv.copy()
    buffered_glaciers['geometry'] = glacier_inv.buffer(500)
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
                f_rgi = dem.data[os.path.basename(tile_file)]['f_rgi']
                dem_extent = dem.data[os.path.basename(tile_file)]['extent']
                tmp = atl03.atl03_to_gdf(aoi=dem_extent, coast=coastline, glaciers=buffered_glaciers)
                if len(tmp) != 0:
                    tmp['dem_h'] = f_rgi((np.array(tmp.geometry.x), np.array(tmp.geometry.y)))
                    gdf_list.append(tmp)
            if len(gdf_list) != 0:
                gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame)
                gdf_final.crs = gdf_list[0].crs
                gdf_final['distance'] = gdf_final.groupby(['orbit_number', 'beam'])['geometry'].shift().distance(
                    gdf_final['geometry']).fillna(0)
                h_diff = gdf_final.groupby(['orbit_number', 'beam'])['dem_h'].diff().values
                gdf_final['h_grad'] = gaussian_filter1d(h_diff/gdf_final['distance'].values, sigma=20)
                # gdf_final['h_grad'] = gaussian_filter1d(
                #     h_diff.values, sigma=10) / gaussian_filter1d(gdf_final['distance'].values, sigma=10)
                gdf_final['distance'] = gdf_final.groupby(['orbit_number', 'beam'])['distance'].cumsum()
                gdf_final = gdf_final.dropna().reset_index(drop=True)
                gdf_final['time'] = gdf_final['time'].astype(int) / float(10 ** 9)
                if len(gdf_final) != 0:
                    gdf_final.to_file(
                        config['dir']['product']['elevation'] +
                        'icesat2_atl03_processed_' + running_date.strftime('%Y%m%d') +
                        '.geojson', driver="GeoJSON")
        running_date += dt


def snow_off(config):
    elevation_dir = config['dir']['product']['elevation']

    file_list = [file_path for file_path in glob.iglob(os.path.join(elevation_dir, "*.geojson"))]
    filenames = [os.path.basename(path) for path in file_list]
    path = os.path.dirname(file_list[0])

    target_months = [7, 8, 9]
    # Filter files based on the target months
    target_files = [
        path + '/' + filename
        for filename in filenames
        if datetime.strptime(filename.split('_')[3].split('.')[0], '%Y%m%d').month in target_months
    ]

    gdf_list = []
    for file in target_files:
        print(file)
        gdf = gpd.read_file(file)
        gdf_list.append(gdf)

    gdf = pd.concat([gdf for gdf in gdf_list]).pipe(gpd.GeoDataFrame)
    gdf = gdf.reset_index(drop=True)

    gdf = gdf[gdf['beam_type'] == 'strong']
    gdf = gdf[gdf['n_ph'] < 10]
    gdf = gdf[abs(gdf['h_grad']) < 0.05]
    gdf['delta_h_snowoff'] = gdf['h_ph_p50'] - gdf['dem_h']
    gdf = gdf[abs(gdf['delta_h_snowoff']) < 2]
    gdf = gdf[gdf['dem_h'] < 500]
    gdf = gdf.reset_index(drop=True)

    snow_off_dir = os.path.join(elevation_dir, 'snow_off')
    if not os.path.exists(snow_off_dir):
        os.makedirs(snow_off_dir)

    gdf.to_file(snow_off_dir + '/' + 'snow_off.geojson', driver="GeoJSON")


def snow_depth_proc(config):
    elevation_dir = config['dir']['product']['elevation']
    snow_depth_dir = config['dir']['product']['snow_depth']

    spatial_extent = config['proc_step_options']['collect']['spatial_extent']
    out_epsg = config['options']['out_epsg']
    aoi_gdf = gpd.GeoDataFrame({'geometry': [create_polygon(spatial_extent)]}, crs='epsg:4326').to_crs(out_epsg)
    xmin = aoi_gdf['geometry'].bounds['minx'][0]
    xmax = aoi_gdf['geometry'].bounds['maxx'][0]
    ymin = aoi_gdf['geometry'].bounds['miny'][0]
    ymax = aoi_gdf['geometry'].bounds['maxy'][0]
    bounds = [xmin, ymin, xmax, ymax]
    grid, cell_size = define_grid(bounds, np.round((xmax - xmin) / 50.0), out_epsg)

    files = [file_path for file_path in glob.iglob(os.path.join(elevation_dir, "*.geojson"))]

    delta_h = gpd.read_file(elevation_dir + 'snow_off/snow_off.geojson')
    var, var_str = ['delta_h_snowoff'], ['delta_h_snowoff']
    snow_off_g = grid_data(delta_h, grid, var, var_str, agg_mode=['mean', 'cnt'])
    snow_off_g = snow_off_g[snow_off_g['delta_h_snowoff_cnt'] > 10]

    rbf = Rbf(snow_off_g.geometry.centroid.x.values,
              snow_off_g.geometry.centroid.y.values,
              snow_off_g.delta_h_snowoff.values, function='linear')

    centroids_df = gpd.GeoDataFrame()
    centroids_df['geometry'] = snow_off_g['geometry'].centroid
    for file in files:
        print(file)
        data = gpd.read_file(file)
        data = data[abs(data['h_grad']) < 0.05].reset_index(drop=True)

        min_distances = []
        for index, row in data.iterrows():
            min_distance = calculate_min_distance(row.geometry, centroids_df)
            min_distances.append(min_distance)
        data['min_distance_to_snow_off'] = min_distances
        data = data[data['min_distance_to_snow_off'] < 500.0].reset_index(drop=True)

        data['delta_h_snowoff'] = rbf(data.geometry.x.values, data.geometry.y.values)
        data['snow_depth'] = data['h_ph_p50'] - data['dem_h'] - data['delta_h_snowoff']
        data = data[(data['h_ph_p99'] - data['h_ph_p50']) < 0.3]
        data = data[data['snow_depth'] > -1]
        data = data[data['snow_depth'] < 6]
        data = data[data['dem_h'] < 500]
        data = data[abs(data['delta_h_snowoff']) < 2.0]
        data = data.reset_index(drop=True)
        if not data.empty:
            data.to_file(snow_depth_dir + 'snow_depth-' + os.path.basename(file), driver="GeoJSON")
