'''
Created on May 27, 2019

@author: Faizan-Uni
'''

import ogr
import numpy as np

from ..misc import print_sl, print_el


class PolyAndCrdsItsctIdxs:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._poly_geoms = None
        self._poly_labels = None

        self._ras_type_labs = ('nc', 'gtiff')

        self._x_crds_orig = None
        self._y_crds_orig = None
        self._ras_type_lab = None
        self._crds_ndims = None

        self._min_itsct_area_pct_thresh = 1
        self._max_itct_cells_thresh = 5000

        self._itsct_idxs_dict = None

        self._set_itsct_idxs_polys_flag = False
        self._set_itsct_idxs_crds_flag = False
        self._set_itsct_idxs_vrfd_flag = False
        self._itsct_idxs_cmptd_flag = False
        return

    def set_polygons(self, polygon_geometries, labels):

        assert isinstance(polygon_geometries, dict)
        assert isinstance(labels, tuple)

        n_polys = len(polygon_geometries)

        assert n_polys

        assert n_polys == len(labels)

        for label in labels:
            assert label in polygon_geometries

            geom = polygon_geometries[label]

            assert isinstance(geom, ogr.Geometry)

            assert geom.GetGeometryType() == 3

            assert len(geom.GetGeometryRef(0).GetPoints()) > 2

        self._poly_geoms = polygon_geometries
        self._poly_labels = labels

        if self._vb:
            print_sl()

            print(
                f'INFO: Set {n_polys} polygons for '
                f'intersection with coordinates')

            print_el()

        self._set_itsct_idxs_polys_flag = True
        return

    def set_coordinates(self, x_crds, y_crds, raster_type_label):

        assert isinstance(x_crds, np.ndarray)
        assert 1 <= x_crds.ndim <= 2
        assert np.all(np.isfinite(x_crds))
        assert np.all(x_crds.shape)

        assert isinstance(y_crds, np.ndarray)
        assert 1 <= y_crds.ndim <= 2
        assert np.all(np.isfinite(y_crds))
        assert np.all(y_crds.shape)

        assert x_crds.ndim == y_crds.ndim

        assert isinstance(raster_type_label, str)
        assert raster_type_label in self._ras_type_labs

        self._x_crds_orig = x_crds
        self._y_crds_orig = y_crds

        self._ras_type_lab = raster_type_label
        self._crds_ndims = x_crds.ndim

        if (self._ras_type_lab == 'nc') and (self._crds_ndims == 1):
            assert x_crds.shape[0] > 1
            assert y_crds.shape[0] > 1

            x_left = (0.5 * (x_crds[+1] - x_crds[+0])) - x_crds[+0]
            x_rght = (0.5 * (x_crds[-1] - x_crds[-1])) + x_crds[-1]

            y_up = (0.5 * (y_crds[+1] - y_crds[+0])) - y_crds[+0]
            y_dn = (0.5 * (y_crds[-1] - y_crds[-1])) + y_crds[-1]

            self._x_crds_orig = np.concatenate(
                (x_left, self._x_crds_orig, x_rght))

            self._y_crds_orig = np.concatenate(
                (y_up, self._y_crds_orig, y_dn))

            if self._vb:
                print_sl()

                print('Padded X and Y coordinates!')

                print_el()

        if self._crds_ndims == 1:
            assert (
                np.all(np.ediff1d(self._x_crds_orig) > 0) or
                np.all(np.ediff1d(self._x_crds_orig[::-1]) > 0))

            assert (
                np.all(np.ediff1d(self._y_crds_orig) > 0) or
                np.all(np.ediff1d(self._y_crds_orig[::-1]) > 0))

        else:
            raise NotImplementedError

        self._x_crds_orig.flags.writeable = False
        self._y_crds_orig.flags.writeable = False

        if self._vb:
            print_sl()

            print(f'INFO: Set the following raster coordinate properties:')
            print(f'Shape of X coordinates: {self._x_crds_orig.shape}')

            print(
                f'X coordinates\' min and max: '
                f'{self._x_crds_orig.min():0.3f}, '
                f'{self._x_crds_orig.max():0.3f}')

            print(f'Shape of Y coordinates: {self._y_crds_orig.shape}')

            print(
                f'Y coordinates\' min and max: '
                f'{self._y_crds_orig.min():0.3f}, '
                f'{self._y_crds_orig.max():0.3f}')

            print_el()

        self._set_itsct_idxs_crds_flag = True
        return

    def set_intersect_misc_settings(
            self,
            minimum_cell_area_intersection_percentage,
            maximum_cells_threshold_per_polygon):

        assert isinstance(
            minimum_cell_area_intersection_percentage, (int, float))

        assert 0 < minimum_cell_area_intersection_percentage <= 100

        assert isinstance(maximum_cells_threshold_per_polygon, int)

        assert 0 < maximum_cells_threshold_per_polygon < np.inf

        self._min_itsct_area_pct_thresh = (
            minimum_cell_area_intersection_percentage)

        self._max_itct_cells_thresh = maximum_cells_threshold_per_polygon

        if self._vb:
            print_sl()

            print(f'INFO: Set the following misc. parameters:')

            print(
                f'Minimum cell area intersection percentage: '
                f'{minimum_cell_area_intersection_percentage}')

            print(
                f'Maximum cells threshold per polygon: '
                f'{maximum_cells_threshold_per_polygon}')

            print_el()
        return

    def verify(self):

        assert self._set_itsct_idxs_polys_flag
        assert self._set_itsct_idxs_crds_flag

        x_min = self._x_crds_orig.min()
        x_max = self._x_crds_orig.max()

        y_min = self._y_crds_orig.min()
        y_max = self._y_crds_orig.max()

        for label in self._poly_labels:
            geom = self._poly_geoms[label]

            gx_min, gx_max, gy_min, gy_max = geom.GetEnvelope()

            assert gx_min >= x_min
            assert gx_max <= x_max

            assert gy_min >= y_min
            assert gy_max <= y_max

        if self._vb:
            print_sl()

            print(
                f'INFO: All inputs for polygons\' and coordinates\' '
                f'intersection verified to be correct')

            print_el()

        self._set_itsct_idxs_vrfd_flag = True
        return

    def _cmpt_1d_idxs(self, geom):

        assert self._crds_ndims == 1

        gx_min, gx_max, gy_min, gy_max = geom.GetEnvelope()

        x_crds = self._x_crds_orig
        y_crds = self._y_crds_orig

        tot_x_idxs = (x_crds >= gx_min) & (x_crds <= gx_max)
        tot_y_idxs = (y_crds >= gy_min) & (y_crds <= gy_max)

        assert tot_x_idxs.sum() > 1
        assert tot_y_idxs.sum() > 1

        assert np.all(x_crds.shape)
        assert np.all(y_crds.shape)

        n_cells_acptd = 0
        x_crds_acptd_idxs = []
        y_crds_acptd_idxs = []
        itsct_rel_areas = []

        geom_area = geom.Area()
        assert geom_area > 0

        for x_idx in range(x_crds.shape[0] - 1):
            if not tot_x_idxs[x_idx]:
                continue

            for y_idx in range(y_crds.shape[0] - 1):
                if not tot_y_idxs[y_idx]:
                    continue

                ring = ogr.Geometry(ogr.wkbLinearRing)

                ring.AddPoint(x_crds[x_idx], y_crds[y_idx])
                ring.AddPoint(x_crds[x_idx + 1], y_crds[y_idx])
                ring.AddPoint(x_crds[x_idx + 1], y_crds[y_idx + 1])
                ring.AddPoint(x_crds[x_idx], y_crds[y_idx + 1])
                ring.AddPoint(x_crds[x_idx], y_crds[y_idx])

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                itsct_poly = poly.Intersection(geom)
                itsct_cell_area = itsct_poly.Area()

                assert 0.0 < itsct_cell_area < np.inf

                min_area_thresh = (
                    (self._min_itsct_area_pct_thresh / 100.0) * poly.Area())

                if itsct_cell_area < min_area_thresh:
                    continue

                n_cells_acptd += 1

                x_crds_acptd_idxs.append(x_idx)
                y_crds_acptd_idxs.append(y_idx)

                itsct_rel_areas.append(itsct_cell_area / geom_area)

        assert n_cells_acptd > 0
        assert n_cells_acptd == len(x_crds_acptd_idxs)
        assert n_cells_acptd == len(y_crds_acptd_idxs)
        assert n_cells_acptd == itsct_rel_areas

        return {
            'x':np.array(x_crds_acptd_idxs, dtype=int),
            'y': np.array(y_crds_acptd_idxs, dtype=int),
            'rel_area': np.array(itsct_rel_areas, dtype=float)}

    def compute_intersect_idxs(self):

        assert self._set_itsct_idxs_vrfd_flag

        self._itsct_idxs_cmptd_flag = False

        assert self._crds_ndims == 1  # for now

        itsct_idxs_dict = {}

        for label in self._poly_labels:
            geom = self._poly_geoms[label]

            if self._crds_ndims == 1:
                res = self._cmpt_1d_idxs(geom)

            else:
                raise NotImplementedError

            itsct_idxs_dict[label] = res

        self._itsct_idxs_dict = itsct_idxs_dict

        self._itsct_idxs_cmptd_flag = True
        return

    def get_intersect_idxs(self):

        assert self._itsct_idxs_cmptd_flag

        return self._itsct_idxs_dict

    __verify = verify
