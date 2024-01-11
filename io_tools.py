from shapely.geometry import Polygon
from shapely.ops import nearest_points
import geopandas as gpd


def calculate_min_distance(point_geom, points_gdf):
    nearest = nearest_points(point_geom, points_gdf.unary_union)
    return nearest[0].distance(nearest[1])


def create_polygon(bounds):
    x_min, y_min = bounds[0], bounds[1]
    x_max, y_max = bounds[2],  bounds[3]
    points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    polygon = Polygon(points)
    return polygon
