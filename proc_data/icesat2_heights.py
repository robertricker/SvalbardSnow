import icepyx as ipx
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import os
import h5py
import datetime
from astropy.time import Time
from datetime import datetime, timedelta
from shapely.geometry import MultiPoint


class Icesat2Heights:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_list = None
        self.target_files = None
        self.file_dates = None

    def get_atl03(self):
        date_range = [self.t0, self.t1]
        short_name = 'ATL03'
        region_a = ipx.Query(short_name, self.spatial_extent, date_range, start_time='00:00:00', end_time='23:59:59')
        region_a.avail_granules()
        atl_re = re.compile(r'ATL.._(?P<year>\d\d\d\d)(?P<month>\d\d)(?P<day>\d\d)\d+_(?P<track>\d\d\d\d)')
        dates_rgt_remote = []
        for count, item in enumerate(region_a.granules.avail):
            granule_info = atl_re.search(item['producer_granule_id']).groupdict()
            dates_rgt_remote += [
                (''.join([granule_info[key] for key in ['year', 'month', 'day']]), granule_info['track'])]

        dates_rgt_local = []
        for file in glob.glob(os.path.join(self.local_dir, '*.h5')):
            dates_rgt_local += [(file.split('_')[2][0:8], file.split('_')[3][0:4])]

        missing = [tuple1 for tuple1 in dates_rgt_remote if tuple1 not in dates_rgt_local]
        if missing:
            rgt = [item[1] for item in missing]
            region_a = ipx.Query(short_name, self.spatial_extent, date_range, tracks=rgt, start_time='00:00:00',
                                 end_time='23:59:59')
            region_a.order_vars.append(keyword_list=['heights', 'orbit_info'])
            region_a.subsetparams(Coverage=region_a.order_vars.wanted)
            region_a.avail_granules()

        region_a.order_granules()
        region_a.download_granules(self.local_dir)

    @staticmethod
    def read_atl03(filename, attributes=False):
        with h5py.File(filename, 'r') as fileid:
            atl03_data = {}
            atl03_attrs = {}
            atl03_beams = []
            for gtx in [k for k in fileid.keys() if bool(re.match(r'gt\d[lr]', k))]:
                try:
                    fileid[gtx]['heights']['delta_time']
                except KeyError:
                    pass
                else:
                    atl03_beams.append(gtx)

            atl03_data['orbit_info'] = {}
            for key, val in fileid['orbit_info'].items():
                if isinstance(val, h5py.Dataset):
                    atl03_data['orbit_info'][key] = val[:]

            for gtx in atl03_beams:
                atl03_data[gtx] = {}
                atl03_data[gtx]['heights'] = {}

                for key, val in fileid[gtx]['heights'].items():
                    if isinstance(val, h5py.Dataset):
                        atl03_data[gtx]['heights'][key] = val[:]

                if attributes:
                    atl03_attrs[gtx] = {}
                    atl03_attrs[gtx]['heights'] = {}
                    # global group attributes for atl03 beams
                    for att_name, att_val in fileid[gtx].attrs.items():
                        atl03_attrs[gtx][att_name] = att_val
                    for key, val in fileid[gtx]['heights'].items():
                        atl03_attrs[gtx]['heights'][key] = {}
                        for att_name, att_val in val.attrs.items():
                            atl03_attrs[gtx]['heights'][key][att_name] = att_val

            if attributes:
                for att_name, att_val in fileid.attrs.items():
                    atl03_attrs[att_name] = att_val

        return atl03_data, atl03_attrs, atl03_beams

    def atl03_to_gdf(self, aoi=None, coast=None, glaciers=None):
        atlas_sdp_gps_epoch = 1198800018.0

        def convert_gps_to_datetime(gps_secs):
            gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
            return gps_epoch + timedelta(seconds=gps_secs)

        gdf_list = list()
        for file in self.target_files:
            atl03_data, atl03_attrs, atl03_beams = self.read_atl03(file, attributes=True)
            beam_list = list()
            for beam in atl03_beams:
                tmp = pd.DataFrame()
                tmp['h_ph'] = atl03_data[beam]['heights']['h_ph']
                tmp['quality_ph'] = atl03_data[beam]['heights']['quality_ph']
                tmp['ph_id_pulse'] = atl03_data[beam]['heights']['ph_id_pulse']
                tmp['signal_conf_ph'] = np.array(
                    [sub_array[0] for sub_array in atl03_data[beam]['heights']['signal_conf_ph']])
                tmp['latitude'] = atl03_data[beam]['heights']['lat_ph']
                tmp['longitude'] = atl03_data[beam]['heights']['lon_ph']
                # tmp['time'] = Time(atl03_data[beam]['heights']['delta_time'] +
                #                    atlas_sdp_gps_epoch, format='gps').to_datetime()
                tmp['time'] = atl03_data[beam]['heights']['delta_time'] + atlas_sdp_gps_epoch
                tmp['time'] = tmp['time'].apply(lambda x: convert_gps_to_datetime(x))
                tmp['beam'] = beam
                tmp['beam_type'] = atl03_attrs[beam]['atlas_beam_type'].decode('utf8')
                tmp['orbit_number'] = atl03_data['orbit_info']['orbit_number'][0]
                tmp = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp.longitude, tmp.latitude), crs=4326)
                tmp = tmp[tmp['signal_conf_ph'] > 2].reset_index(drop=True)
                tmp = tmp.to_crs(self.out_epsg)

                tmp = gpd.sjoin(tmp, aoi, how="inner", predicate="within").drop(columns='index_right')
                tmp = gpd.sjoin(tmp, coast, how="inner", predicate="within").drop(columns='index_right')
                tmp = gpd.sjoin(tmp, glaciers, how='left', predicate="within")
                tmp = tmp[tmp['index_right'].isnull()]
                tmp.drop(columns='index_right', inplace=True)
                tmp.reset_index(drop=True, inplace=True)

                if len(tmp) != 0:
                    grouped = tmp.groupby('time').agg({'h_ph': list,
                                                       'geometry': list,
                                                       'longitude': list,
                                                       'latitude': list}).reset_index()

                    grouped['ph_count_shot'] = grouped['h_ph'].apply(len)
                    columns_to_filter = ['h_ph', 'longitude', 'latitude', 'geometry']
                    grouped_filtered = grouped.apply(lambda row:
                                                     self.filter_percentile(row, 'h_ph', columns_to_filter), axis=1)
                    grouped_filtered['time'] = grouped['time'].copy()
                    grouped_filtered['beam'] = tmp['beam'][0]
                    grouped_filtered['beam_type'] = tmp['beam_type'][0]
                    grouped_filtered['orbit_number'] = tmp['orbit_number'][0]
                    grouped_filtered['ph_count_shot'] = grouped['ph_count_shot'].copy()
                    grouped_filtered = grouped_filtered[grouped_filtered['h_ph'].apply(lambda x: len(x) > 0)]
                    grouped_filtered.reset_index(drop=True, inplace=True)
                    window_size = 5
                    p50 = []
                    p85 = []
                    p99 = []
                    n_ph = []
                    if len(grouped_filtered) > window_size*2:
                        for i in range(len(grouped_filtered) - window_size + 1):
                            subset = grouped_filtered['h_ph'][i:i + window_size].reset_index(drop=True)
                            subset = np.concatenate(subset)
                            subset_t = grouped_filtered['time'][i:i + window_size].reset_index(drop=True)
                            p50.append(self.aggregate_function(subset, subset_t, window_size, 50))
                            p85.append(self.aggregate_function(subset, subset_t, window_size, 85))
                            p99.append(self.aggregate_function(subset, subset_t, window_size, 99))
                            n_ph.append(len(subset))

                        padding = int((window_size - 1) / 2) * [np.nan]
                        grouped_filtered['h_ph_p50'] = padding + p50 + padding
                        grouped_filtered['h_ph_p85'] = padding + p85 + padding
                        grouped_filtered['h_ph_p99'] = padding + p99 + padding
                        grouped_filtered['n_ph'] = padding + n_ph + padding
                        grouped_filtered['geometry'] = grouped_filtered['geometry'].apply(self.calculate_mean_position)
                        if len(grouped_filtered) != 0:
                            beam_list.append(grouped_filtered)
            if len(beam_list) != 0:
                df = pd.concat([df for df in beam_list]).reset_index(drop=True)
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=self.out_epsg)
                gdf_list.append(gdf)

        if len(gdf_list) != 0:
            gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame)
            gdf_final.drop(columns=['latitude', 'longitude', 'h_ph'], inplace=True)
            gdf_final.crs = gdf_list[0].crs
            gdf_final = gdf_final.reset_index(drop=True)
        else:
            gdf_final = pd.DataFrame()
        return gdf_final

    @staticmethod
    def filter_percentile(row, ref_column, columns_to_filter):
        lower = np.percentile(row[ref_column], 15)
        upper = np.percentile(row[ref_column], 85)
        filtered_data = {}
        for column in columns_to_filter:
            filtered_data[column] = [val for i, val in enumerate(row[column]) if lower <= row[ref_column][i] <= upper]
        return pd.Series(filtered_data)

    @staticmethod
    def aggregate_function(subset, subset_t, window_size, threshold):
        delta_t = np.max(subset_t) - np.min(subset_t)
        if len(subset) < (window_size * 1) or delta_t.total_seconds() > 2e-3:
            return np.nan
        else:
            return np.percentile(subset, threshold)

    @staticmethod
    def calculate_mean_position(point_list):
        multi_point = MultiPoint(point_list)
        mean_point = multi_point.centroid
        return mean_point

    def get_file_list(self):
        file_list = [file_path for file_path in glob.iglob(os.path.join(self.local_dir, "**", "*"), recursive=True)]
        self.file_list = file_list

    def get_file_dates(self):
        date_str = '{14}'
        date_pt = '%Y%m%d%H%M%S'
        dates = [datetime.strptime(re.search(r'\d' + date_str, file).group(), date_pt) for file in self.file_list]
        self.file_dates = dates

    def get_target_files(self, t0, t1):
        dates = self.file_dates
        file_list = self.file_list
        self.target_files = [file for date, file in zip(dates, file_list) if t0 <= date < t1]
