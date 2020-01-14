# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

test simple kriging cython class
"""

import os
import numpy as np
import fiona
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import shapely.geometry as shg

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# Bestimme Interpolationspunkte
from matplotlib import path

from pathlib import Path


plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})


# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'


bw_area_shp = (r"X:\hiwi\ElHachem\GitHub\extremes"
               r"\Landesgrenze_ETRS89\Landesgrenze_10000_ETRS89.shp")


#==============================================================================
# NEEDED FUNCTIONS
#==============================================================================


nrows = 250
ncols = 250
cellsize = 50000


# informationen aus shapefiles holen
ishape = fiona.open(bw_area_shp)
shp_objects_all = list(fiona.open(bw_area_shp))
# first = ishape.next()
first = next(iter(ishape))

# shp_geom1 = shape(first['geometry'])
# polygon = np.array(first['geometry']['coordinates'][0])

polygons = shape(shp_objects_all[0]['geometry'])
bounding_box = shape(polygons).bounds

(lon_sw, lat_sw,
    lon_ne, lat_ne) = (bounding_box[0], bounding_box[1],
                       bounding_box[2], bounding_box[3])


x_grid = np.linspace(lon_sw - 100, lon_ne + 100, nrows)
y_grid = np.linspace(lat_sw - 100, lat_ne + 100, ncols)

mesh_xx, mesh_yy = np.meshgrid(x_grid, y_grid)


try:
    bawue = shp_objects_all[0]['geometry']['coordinates'][0][0]
except IndexError:
    bawue = shp_objects_all[0]['geometry'][:, :]


poly_bw = Polygon(bawue)

p = path.Path(bawue)

coords_x = [i[0] for i in bawue]
coords_y = [i[1] for i in bawue]
#
# coords_xy = [i for i in bawue]
#
# coords_to_keep = [(x, y) for x, y in zip(mesh_xx.flatten(), mesh_yy.flatten())
#                   if poly_bw.contains(Point(x, y))]

coords_xk = []
coords_yk = []


for i, (x, y) in enumerate(zip(mesh_xx.flatten(), mesh_yy.flatten())):
    print(i, ' / ', mesh_xx.flatten().size)
    if poly_bw.contains(Point(x, y)):
        coords_xk.append(x)
        coords_yk.append(y)

# coords_xk = [i[0] for i in coords_to_keep]
# coords_yk = [i[1] for i in coords_to_keep]

coords_interpolate = pd.DataFrame()
coords_interpolate['X'] = coords_xk
coords_interpolate['Y'] = coords_yk

coords_interpolate.to_csv(
    r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\oridinary_kriging_compare_DWD_Netatmo\coords_interpolate_midle.csv',
    sep=';')
