import numpy as np
import os
import ftplib
import tarfile
import geopandas as gpd
import pandas as pd
import rioxarray
from io_tools import create_polygon
from scipy.interpolate import RegularGridInterpolator
from loguru import logger


class ArcticDem:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.host = 'ftp.data.pgc.umn.edu'
        self.tile_list = []
        self.target_files = []
        self.data = dict()

    def collect_in_aoi(self, aoi):
        tile_ind_list = gpd.read_file(self.tile_idx_file)
        intersecting_polygons = gpd.sjoin(tile_ind_list, aoi, predicate='intersects')
        intersecting_polygons.reset_index(drop=True, inplace=True)

        for idx in intersecting_polygons.index:
            super_tile = intersecting_polygons.iloc[idx]['supertile']
            tile = intersecting_polygons.iloc[idx]['tile']
            self.tile_list.append(tile)
            if any(tile in file for file in os.listdir(self.local_dir)):
                print(str(tile) + ' already in ' + self.local_dir)
            else:
                self.get_from_remote(super_tile, tile)
        self.get_target_files()

    def get_from_remote(self, super_tile, tile):
        ftp = ftplib.FTP(self.host)
        ftp.login()
        ftp.cwd(os.path.join(self.host_dir, super_tile))
        file_match = [file for file in ftp.nlst() if tile in file]
        out_file = os.path.join(self.local_dir, file_match[0])
        print('write to file: ' + out_file)
        with open(out_file, 'wb') as file:
            ftp.retrbinary('RETR %s' % file_match[0], file.write)
        self.extract_tar_gz(out_file)
        os.remove(out_file)

    def get_target_files(self):
        for tile_dir in os.listdir(self.local_dir):
            if any(tile in tile_dir for tile in self.tile_list):
                self.target_files.append(os.path.join(self.local_dir, tile_dir, tile_dir+'_dem.tif'))

    def get_data(self):
        for tile_file in self.target_files:
            print(tile_file)
            tile = rioxarray.open_rasterio(tile_file)
            # idx = gpd.read_file(os.path.join(os.path.dirname(tile_file), 'index'))
            xc, yc, dem_h = tile.x.values, tile.y.values, np.array(tile[0])
            np.ma.getdata(dem_h)[np.ma.getdata(dem_h) == -9999.0] = np.nan
            if xc[0] > xc[-1]:
                xc = xc[::-1]
                dem_h = dem_h[:, ::-1]
            if yc[0] > yc[-1]:
                yc = yc[::-1]
                dem_h = dem_h[::-1, :]
            spatial_extent = [np.min(xc), np.min(yc), np.max(xc), np.max(yc)]
            extent = gpd.GeoDataFrame({'geometry': [create_polygon(spatial_extent)]}, crs='epsg:3413')
            f_rgi = RegularGridInterpolator((xc, yc), dem_h.T, method='linear')
            self.data[os.path.basename(tile_file)] = {'extent': extent, 'f_rgi': f_rgi}

    @staticmethod
    def extract_tar_gz(file_name):
        with tarfile.open(file_name, 'r:gz') as tar:
            tar.extractall(path=file_name.replace(".tar.gz", ""))
