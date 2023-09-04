import json
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from matplotlib.patches import Ellipse
import matplotlib.image as mpimg
from math import sin, cos

# Opening JSON file
f = open('via_region_data.json')

# returns JSON object as
# a dictionary
data = json.load(f)


# Iterating through the json
# list
def discretizeEllipse(cx, cy, rx, ry, seg,alpha):
    TWO_PI = 44 / 7.0
    angle_shift = TWO_PI / seg
    phi = 0
    vertices = []
    for i in range(0, seg):
        phi += angle_shift
        x1 = rx * cos(phi)
        y1 = ry * sin(phi)
        x2 = x1 * cos(alpha) - y1 * sin(alpha)
        y2 = x1 * sin(alpha) + y1 * cos(alpha)
        vertices.append((cx + x2, cy + y2))
    return vertices

#ells = []
#polygon=[]
for i in data:
    #if data[i]['filename'] != '06.jpg':
    #    continue
    #img = mpimg.imread(data[i]['filename'])
    for r in (data[i]['regions']):
        if r['shape_attributes']['name'] == "ellipse":
            tmpdata = {}
            tmpdata['name'] = "polygon"
            allxs = []
            allys = []
            tmp =discretizeEllipse(r['shape_attributes']['cx'], r['shape_attributes']['cy']
                                    , r['shape_attributes']['rx'], r['shape_attributes']['ry']
                                    , 100,r['shape_attributes']['theta']
                                    )
            for j in tmp:
                allxs.append(j[0])
                allys.append(j[1])
            tmpdata['all_points_x'] = allxs
            tmpdata['all_points_y'] = allys
            tmpdata['region_attributes'] = {}
            r['shape_attributes'] = tmpdata
        if r['shape_attributes']['name'] == "circle":
            tmpdata = {}
            tmpdata['name'] = "polygon"
            allxs = []
            allys = []
            tmp = discretizeEllipse(r['shape_attributes']['cx'], r['shape_attributes']['cy']
                                    , r['shape_attributes']['r'], r['shape_attributes']['r']
                                    , 100, 0
                                    )
            for j in tmp:
                allxs.append(j[0])
                allys.append(j[1])
            tmpdata['all_points_x'] = allxs
            tmpdata['all_points_y'] = allys
            tmpdata['region_attributes'] = {}
            r['shape_attributes'] = tmpdata
        if r['shape_attributes']['name'] == "point":
            tmpdata = {}
            tmpdata['name'] = "polygon"
            allxs = []
            allys = []
            tmp = discretizeEllipse(r['shape_attributes']['cx'], r['shape_attributes']['cy']
                                    , 3, 3
                                    , 100, 0
                                    )
            for j in tmp:
                allxs.append(j[0])
                allys.append(j[1])
            tmpdata['all_points_x'] = allxs
            tmpdata['all_points_y'] = allys
            tmpdata['region_attributes'] = {}
            r['shape_attributes'] = tmpdata

a_file = open("sample_file.json", "w")
json.dump(data, a_file)
a_file.close()
# Closing file
f.close()
''''
imgplot = plt.imshow(img)

ax = plt.gca()
for e in ells:
    ax.add_patch(e)
for a in polygon:
    polygon1 = Polygon(a)
    x, y = polygon1.exterior.xy
    plt.plot(x, y, c="red")

plt.show()
'''