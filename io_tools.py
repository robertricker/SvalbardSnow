from shapely.geometry import Polygon


def create_polygon(bounds):
    x_min, y_min = bounds[0], bounds[1]
    x_max, y_max = bounds[2],  bounds[3]
    points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    polygon = Polygon(points)
    return polygon
