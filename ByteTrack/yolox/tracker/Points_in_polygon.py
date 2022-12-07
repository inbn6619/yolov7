import numpy as np
from shapely.geometry import Point, Polygon


def limited_area(bbox, area_poly):
    makecenter = lambda x1, y1, x2, y2 : (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
    bbox_center = [makecenter(box[0], box[1], box[2], box[3]) for box in bbox]
    checkpoint = lambda x : x.within(area_poly)
    check = np.array([checkpoint(Point(dot)) for dot in bbox_center])

    return check



area = [(36, 768), (936, 816), (940, 419), (1556, 430), (1384, 279), (491, 275)]

poly = Polygon(area)

mealarea = [(61, 816), (22, 690), (908, 739), (913, 897)]

waterarea = [(472, 347), (1394, 347), (1394, 214), (472, 214)]

mealarea_poly = Polygon(mealarea)

waterarea_poly = Polygon(waterarea)
