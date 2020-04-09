#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

test simple kriging cython class
"""
import os

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import pyximport
import numpy as np
# pyximport.install()
pyximport.install()


import timeit
import time
#import math
import fiona
import pyproj
import glob
import osr
import pandas as pd
import wradlib as wrl
#import shapefile as shp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from spinterps import (OrdinaryKriging)
from scipy import spatial
from scipy.spatial import cKDTree
from pathlib import Path


plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})

use_reduced_sample_dwd = False

# def epsg wgs84 and utm32 for coordinates conversion
wgs82 = "+init=EPSG:4326"
utm32 = "+init=EPSG:32632"
# =============================================================================

main_dir = Path(r'X:\staff\elhachem\2020_10_03_Rheinland_Pfalz')
#main_dir = Path(r'/home/abbas/Documents/Python/Extremes')
# main_dir = Path(r'/home/IWS/hachem/Extremes')
os.chdir(main_dir)

path_to_data = main_dir

path_to_vgs = main_dir

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'dwd_coords_in_around_RH_utm32.csv')

path_to_netatmo_coords = (path_to_data /
                          r'netatmo_Rheinland-Pfalz_1hour_coords_utm32.csv')

# path for data filter
# in_filter_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

# path_to_dwd_stns_comb
#path_to_dwd_stns_comb = in_filter_path / r'dwd_combination_to_use.csv'

# path for interpolation grid
path_grid_interpolate = main_dir / \
    r"interpolation_grid_utm32.csv"  # _small  # _midle


# open shapefile
shp_objects_all = list(fiona.open(
    r"X:\hiwi\ElHachem\Peru_Project\ancahs_dem\DEU_adm\DEU_adm1.shp"))
shp_objects_all = [shp for shp in shp_objects_all
                   if shp['properties']['NAME_1'] == 'Rheinland-Pfalz']
# mas for BW data
mask = np.load(r"X:\staff\elhachem\2020_10_03_Rheinland_Pfalz\mask",
               allow_pickle=True)
#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================

# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging
use_on_event_filter = False
use_on_event_filter_check_drywet = False
use_tranformation = True
# if use_netatmo_gd_stns:
path_to_netatmo_gd_stns = (
    main_dir / r'indicator_correlation' /
    (r'keep_stns_all_neighbor_99_0_per_60min_s0_1st_rh.csv'))

# for second filter
# path_to_dwd_ratios = in_filter_path / 'ppt_ratios_'


# if use_reduced_sample_dwd:
#     #     path_to_dwd_coords = (path_to_data /
#     #                           r'reduced_smaple_dwd_stns_utm32.csv')
#
#     path_to_netatmo_gd_stns = (main_dir / r'indicator_correlation' /
#                                (r'keep_stns_all_neighbor_99_per_60min_s0_%s_2.csv'
#                                 % _acc_))
#==============================================================================
#
#==============================================================================
resample_frequencies = ['60min']
# '120min', '180min', '60min',  '360min',
#                         '720min',
title_ = r'max_DWD_RLP_1st_trf'


#==============================================================================
#
#==============================================================================
strt_date = '2014-06-01 00:00:00'
end_date = '2019-12-31 00:00:00'


idx_time_fmt = '%Y-%m-%d %H:%M:%S'

radius = 10000
diff_thr = 0.1
edf_thr = 0.7  # 0.9

hourly_events = ['2015-09-16 04:00:00',
                 '2015-09-16 06:00:00',
                 '2015-09-16 07:00:00',
                 '2015-09-16 11:00:00',
                 '2015-09-16 17:00:00',
                 '2015-09-16 18:00:00',
                 '2015-09-16 22:00:00',
                 '2015-09-17 00:00:00']


daily_events = ['2018-12-23 00:00:00',
                '2019-05-22 00:00:00',
                '2018-05-14 00:00:00',
                '2019-07-28 00:00:00']
#==============================================================================
#
#==============================================================================
# Interpoalte Coords


grid_interp_df = pd.read_csv(path_grid_interpolate,
                             index_col=0,
                             sep=';',
                             encoding='utf-8')

x_coords_grd = grid_interp_df.loc[:, 'X'].values.ravel()
# x_coords_grd2 = grid_interp_df.loc[:, 'right'].values.ravel()
# y_coords_grd1 = grid_interp_df.loc[:, 'bo ttom'].values.ravel()
y_coords_grd = grid_interp_df.loc[:, 'Y'].values.ravel()


# x_coords_grd = x_coords_grd1  # np.hstack((x_coords_grd1, x_coords_grd2))
# y_coords_grd = y_coords_grd2  # np.hstack((y_coords_grd1, y_coords_grd2))

# Netatmo Coords
netatmo_in_coords_df = pd.read_csv(path_to_netatmo_coords,
                                   index_col=0,
                                   sep=';',
                                   encoding='utf-8')

# DWD Coords

dwd_in_coords_df = pd.read_csv(path_to_dwd_coords,
                               index_col=0,
                               sep=';',
                               encoding='utf-8')


# Netatmo first filter
df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
                         index_col=0,
                         sep=';',
                         encoding='utf-8')


#==============================================================================
# NEEDED FUNCTIONS
#==============================================================================


def build_edf_fr_vals(data):
    """ construct empirical distribution function given data values """
    from statsmodels.distributions.empirical_distribution import ECDF
    cdf = ECDF(data)
    return cdf.x, cdf.y
#==============================================================================
#
#==============================================================================


def find_nearest(array, value):
    ''' given a value, find nearest one to it in original data array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
#==============================================================================
#
#==============================================================================


def convert_coords_fr_wgs84_to_utm32_(epgs_initial_str, epsg_final_str,
                                      first_coord, second_coord):
    """
    Purpose: Convert points from one reference system to a second
    --------
        In our case the function is used to transform WGS84 to UTM32
        (or vice versa), for transforming the DWD and Netatmo station
        coordinates to same reference system.

        Used for calculating the distance matrix between stations

    Keyword argument:
    -----------------
        epsg_initial_str: EPSG code as string for initial reference system
        epsg_final_str: EPSG code as string for final reference system
        first_coord: numpy array of X or Longitude coordinates
        second_coord: numpy array of Y or Latitude coordinates

    Returns:
    -------
        x, y: two numpy arrays containing the transformed coordinates in 
        the final coordinates system
    """
    initial_epsg = pyproj.Proj(epgs_initial_str)
    final_epsg = pyproj.Proj(epsg_final_str)
    x, y = pyproj.transform(initial_epsg, final_epsg,
                            first_coord, second_coord)
    return x, y
#==============================================================================
#
#==============================================================================


def plot_all_interplations_subplots(vals_to_plot_dwd_netatmo,
                                    vals_to_plot_dwd,
                                    vals_to_plot_netatmo,
                                    vals_to_plot_dwd_min_dwd_netatmo,
                                    vals_to_plot_dwd_min_radolan,
                                    vals_to_plot_dwd_min_netatmo,
                                    out_plot_path,
                                    temp_agg,
                                    event_date,
                                    radar_data,
                                    radar_lon,
                                    radar_lat,
                                    save_acc,
                                    nbr_netatmo,
                                    nbr_dwd,
                                    nbr_netatmo_flt):
    '''plot interpolated events, grid wise '''

    if temp_agg == '60min':
        min_val, max_val = 0, 30
        clbr_label = 'mm/h'  # 'Hourly precipitation [m]'
        bound_ppt = [0., 1, 2, 4, 8, 10, 15, 20, 25, 30]  # , 40, 45]
    if temp_agg == '1440min':
        min_val, max_val = 0, 45
        clbr_label = 'mm/d'
        bound_ppt = [0., 1, 2, 4, 8, 10, 15, 20, 25, 30, 40, 45]

    plt.ioff()

#     bound_ppt = [0., 1, 2, 4, 8, 10, 15, 20, 25, 30]  # , 40, 45]

    interval_ppt = np.linspace(0.05, 0.95)
    colors_ppt = plt.get_cmap('jet_r')(interval_ppt)
    cmap_ppt = LinearSegmentedColormap.from_list('name', colors_ppt)
    #cmap_ppt = plt.get_cmap('jet_r')
    cmap_ppt.set_over('navy')
    norm_ppt = mcolors.BoundaryNorm(bound_ppt, cmap_ppt.N)

    bound_diff = [-15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15]
    _fontsize = 8
    color_fontsize = 12
    # Remove the middle 10% of the RdBu_r colormap
    #interval = np.hstack([np.linspace(0.0, 0.35), np.linspace(0.7, 1)])
    #colors = plt.get_cmap('PiYG')(interval)  #
    #cmap_diff = LinearSegmentedColormap.from_list('diff', colors)

    # Concatenating colormaps
    cmap_diff = LinearSegmentedColormap.from_list(
        'custom',
        [(0,    'darkmagenta'),
         (0.1,    'mediumorchid'),
         (0.2, 'hotpink'),
         (0.3, 'pink'),
         (0.4, 'lavender'),  # mintcream lavender
         (0.5, 'lavender'),
         (0.6, 'lavender'),
         (0.7, 'yellowgreen'),
         (0.8, 'lawngreen'),
         (0.9, 'green'),
         (1,    'darkgreen')], N=150)
    cmap_diff.set_over('darkslategrey')  # darkslategrey
    cmap_diff.set_under('crimson')

    #cmap_diff = plt.get_cmap('PiYG')
    norm_diff = mcolors.BoundaryNorm(bound_diff, cmap_diff.N)

    fig = plt.figure(figsize=(20, 12), constrained_layout=False, dpi=600)
    gs = gridspec.GridSpec(2, 9, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1])

#     gs.update(left=0.25, right=1, wspace=0.01)
    # dwd-netatmo

    ax1 = fig.add_subplot(gs[:1, :2])
    ax1.scatter(x_coords_grd, y_coords_grd,
                c=vals_to_plot_dwd_netatmo,
                marker=',', s=30, cmap=cmap_ppt,
                vmin=min_val,
                norm=norm_ppt,
                vmax=max_val)

#     ax1.scatter(shp_xs, shp_ys, c='k',
#                 marker='.', s=10, alpha=0.05)
#     ax1.scatter(dwd_xcoords, dwd_ycoords, c='darkgreen',
#                 marker='x', s=10, alpha=0.25)

    ax1.legend(title='DWD+Netatmo (a)\n%d+%d' % (nbr_dwd, nbr_netatmo_flt),
               loc='upper left',  # upper
               frameon=False, fontsize=_fontsize)._legend_box.align = 'left'
    # dwd
    ax2 = fig.add_subplot(gs[:1, 2:4])
    ax2.scatter(x_coords_grd, y_coords_grd,
                c=vals_to_plot_dwd,
                marker=',', s=30, cmap=cmap_ppt,
                vmin=min_val,
                norm=norm_ppt,
                vmax=max_val)
#     ax2.scatter(dwd_xcoords, dwd_ycoords, c='darkgreen',
#                 marker='x', s=10, alpha=0.25)
    ax2.legend(title='DWD (b)\n%d' % nbr_dwd, loc='upper left',
               frameon=False, fontsize=_fontsize)._legend_box.align = 'left'

    # radolan
    ax_rad = fig.add_subplot(gs[:1, 4:6])

    _ = ax_rad.scatter(radar_lon, radar_lat,
                       c=radar_data, cmap=cmap_ppt, s=20, marker=',',
                       vmin=min_val, norm=norm_ppt, vmax=max_val)
    ax_rad.legend(title='Radolan (c)', loc='upper left',
                  frameon=False, fontsize=_fontsize)._legend_box.align = 'left'

    # netatmo
    ax3 = fig.add_subplot(gs[:1, 6:8])

    im3 = ax3.scatter(x_coords_grd, y_coords_grd,
                      c=vals_to_plot_netatmo,
                      marker=',', s=30, cmap=cmap_ppt,
                      vmin=min_val,
                      norm=norm_ppt,
                      vmax=max_val)
#     ax3.scatter(netatmo_xcoords0, netatmo_ycoords0, c='m',
#                 marker='1', s=10, alpha=0.25)
    ax3.legend(title='Netatmo(d) \n%d' % nbr_netatmo,
               loc='upper left',
               frameon=False, fontsize=_fontsize)._legend_box.align = 'left'

    # colorbar
#     cax0 = fig.add_subplot(gs[:1, 6:7])
    cax0 = fig.add_subplot(gs[:1, 8:9])

    divider0 = make_axes_locatable(cax0)
    cax20 = divider0.append_axes("left", size="8%", pad=0.00001)
#     divider0 = make_axes_locatable(ax3)
#     cax0 = divider0.append_axes("right", size="5%", pad=0.15)

    cb0 = fig.colorbar(im3, ax=ax3, cax=cax20, norm=norm_ppt,
                       ticks=bound_ppt, label=clbr_label,
                       extend='max')

    cb0.set_ticks(bound_ppt)
    cb0.ax.tick_params(labelsize=color_fontsize)

    #==========================================================================
    # # second row
    #==========================================================================
    # dwd-dwd_netatmo
    ax5 = fig.add_subplot(gs[1:, 1:3])
    ax5.scatter(x_coords_grd, y_coords_grd,
                c=vals_to_plot_dwd_min_dwd_netatmo,
                marker=',', s=30, cmap=cmap_diff,
                vmin=bound_diff[0],
                norm=norm_diff,
                vmax=bound_diff[-1])
    ax5.legend(title='(a)-(b)', loc='upper left',
               frameon=False, fontsize=_fontsize)._legend_box.align = 'left'

    # dwd-radolan
    ax_rad2 = fig.add_subplot(gs[1:, 3:5])
    _ = ax_rad2.scatter(x_coords_grd, y_coords_grd,
                        c=vals_to_plot_dwd_min_radolan,
                        marker=',', s=30, cmap=cmap_diff,
                        vmin=bound_diff[0],
                        norm=norm_diff,
                        vmax=bound_diff[-1])
    ax_rad2.legend(title='(c)-(b)', loc='upper left',
                   frameon=False, fontsize=_fontsize)._legend_box.align = 'left'

    # dwd-netatmo
    ax6 = fig.add_subplot(gs[1:, 5:7])
    im6 = ax6.scatter(x_coords_grd, y_coords_grd,
                      c=vals_to_plot_dwd_min_netatmo,
                      marker=',', s=30, cmap=cmap_diff,
                      vmin=bound_diff[0],
                      norm=norm_diff,
                      vmax=bound_diff[-1])
    ax6.legend(title='(d)-(b)', loc='upper left',
               frameon=False, fontsize=_fontsize)._legend_box.align = 'left'

    ###
    cax = fig.add_subplot(gs[1:, 7:8])

    divider = make_axes_locatable(cax)
    cax2 = divider.append_axes("left", size="8%", pad=0.00001)

    cb1 = fig.colorbar(im6, ax=ax6, cax=cax2, norm=norm_diff,
                       ticks=bound_diff, label=clbr_label,
                       extend='both')
    cb1.set_ticks(bound_diff)
    cb1.ax.tick_params(labelsize=color_fontsize)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

#     fig.patch.set_visible(False)
    ax1.axis('off'), ax2.axis('off'), ax3.axis('off')
    ax5.axis('off'), ax6.axis('off'), ax_rad.axis('off')
    cax.axis('off'), cax0.axis('off'), ax_rad2.axis('off')
#     plt.tight_layout()
    # plt.show()
    plt.savefig((
        out_plot_path / (
                r'%s_%s_%s_event_3' % (save_acc, temp_agg, str(
                    event_date).replace(
                        '-', '_').replace(':', '_').replace(' ', '_')))),
                papertype='a4',
                bbox_inches='tight',
                pad_inches=0.05)
    plt.close()

    pass

#==============================================================================
#
#==============================================================================


lon_coords_grd, lat_coords_grd = convert_coords_fr_wgs84_to_utm32_(
    epgs_initial_str=utm32,
    epsg_final_str=wgs82,
    first_coord=x_coords_grd,
    second_coord=y_coords_grd)

#==============================================================================
#
#==============================================================================
for temp_agg in resample_frequencies:

    # out path directory

    dir_path = title_ + '_' + temp_agg

    #dir_path = title_ + temp_agg
    out_plots_path = main_dir / dir_path

    if not os.path.exists(out_plots_path):
        os.mkdir(out_plots_path)
    print(out_plots_path)

    # path to data
    #=========================================================================
    out_save_csv = '_%s_ppt_edf_' % temp_agg

    path_to_dwd_edf = (path_to_data /
                       (r'edf_ppt_all_dwd_%s_.csv' % temp_agg))

    path_to_dwd_ppt = (path_to_data /
                       (r'ppt_dwd_2014_2019_%s_no_freezing_5deg.csv'
                           % temp_agg))

#     path_to_dwd_edf_old = (path_to_data /
#                            (r'edf_ppt_all_dwd_old_%s_.csv' % temp_agg))
#
#     path_to_dwd_ppt_old = (path_to_data /
#                            (r'ppt_all_dwd_old_%s_.csv' % temp_agg))
#     path_dwd_ratios = (path_to_dwd_ratios /
#                        (r'dwd_ratios_%s_data.csv' % temp_agg))
    path_to_netatmo_edf = (
        path_to_data /
        (r'edf_ppt_all_netatmo_%s_.csv' % temp_agg))

    path_to_netatmo_ppt = (
        path_to_data /
        (r'ppt_all_netatmo_rh_2014_2019_%s_no_freezing_5deg.csv'
            % temp_agg))

    path_to_dwd_vgs = (
        path_to_vgs / r'vg_strs_max100_hours_60min_vg1.csv')
    # r'vg_strs_special_events_5mm_60min.csv')
    #('vg_strs_max100_hours_%s2.csv' % temp_agg))
    #         ('vg_strs_special_events_%s.csv' % temp_agg))
#         (r'vg_strs_max100_hours_%s.csv' % temp_agg))

#     path_to_dwd_vgs = (
#         r"X:\exchange\ElHachem\Events_HBV\Echaz\df_vgs_events2.csv")

    path_dwd_extremes_df = (
        main_dir / (r'dwd_%s_maximum_100_hours.csv' % temp_agg))
    # r'dwd_60min_special_events_5mm_.csv')
    #"Data_Bardossy/EventsRLP.csv")
#         (r'dwd_%s_maximum_100_hours.csv' % temp_agg)
#         (r'dwd_%s_special_events_10mm_.csv' % temp_agg)
    #(r'dwd_%s_maximum_100_hours.csv' % temp_agg)

    df_radar_events_to_keep = pd.read_csv(
        r'X:\staff\elhachem\2020_10_03_Rheinland_Pfalz'
        #         r'X:\exchange\ElHachem'
        r'\%s_max_events_radolan_files.csv'
        # r'\%s_intense_events_5mm_radolan_files.csv'
        # r'\%s_intense_events_radolan_files.csv'
        #         r'\%s_intense_events_radolan_files_RLP.csv'
        # r'\%s_intense_events_5mm_radolan_files.csv'
        % (temp_agg), index_col=0,
        sep=';', engine='c',
        parse_dates=True,
        infer_datetime_format=True)

    # RADOLAN
    base_path = (r'X:\exchange\ElHachem\radolan_%s_data\*' % temp_agg)

    files = glob.glob(base_path)

    # Files to use
    # =========================================================================

    netatmo_data_to_use = path_to_netatmo_edf
    dwd_data_to_use = path_to_dwd_edf

    print(title_)
    # DWD DATA
    # =========================================================================
    dwd_in_vals_df = pd.read_csv(
        path_to_dwd_edf, sep=';', index_col=0, encoding='utf-8')

    dwd_in_vals_df.index = pd.to_datetime(
        dwd_in_vals_df.index, format='%Y-%m-%d')

    dwd_in_vals_df = dwd_in_vals_df.loc[strt_date:end_date, :]
    dwd_in_vals_df.dropna(how='all', axis=0, inplace=True)

    # DWD ppt
    dwd_in_ppt_vals_df = pd.read_csv(
        path_to_dwd_ppt, sep=';', index_col=0, encoding='utf-8')

    dwd_in_ppt_vals_df.index = pd.to_datetime(
        dwd_in_ppt_vals_df.index, format='%Y-%m-%d')

    dwd_in_ppt_vals_df = dwd_in_ppt_vals_df.loc[strt_date:end_date, :]
    dwd_in_ppt_vals_df.dropna(how='all', axis=0, inplace=True)

    # DWD old edf
#
#     dwd_in_vals_edf_old = pd.read_csv(
#         path_to_dwd_edf_old, sep=';', index_col=0, encoding='utf-8')
#
#     dwd_in_vals_edf_old.index = pd.to_datetime(
#         dwd_in_vals_edf_old.index, format='%Y-%m-%d')
#
#     dwd_in_vals_edf_old.dropna(how='all', axis=0, inplace=True)
#     # dwd ppt old
#     dwd_in_ppt_vals_df_old = pd.read_csv(
#         path_to_dwd_ppt_old, sep=';', index_col=0, encoding='utf-8')
#
#     dwd_in_ppt_vals_df_old.index = pd.to_datetime(
#         dwd_in_ppt_vals_df_old.index, format='%Y-%m-%d')

    # DWD ratios for second filter
#     dwd_ratios_df = pd.read_csv(path_dwd_ratios,
#                                 index_col=0,
#                                 sep=';',
#                                 parse_dates=True,
#                                 infer_datetime_format=True,
#                                 encoding='utf-8')

    # NETAMO DATA
    #=========================================================================
    netatmo_in_vals_df = pd.read_csv(path_to_netatmo_edf,
                                     sep=';',
                                     index_col=0,
                                     encoding='utf-8', engine='c')

    netatmo_in_vals_df.index = pd.to_datetime(
        netatmo_in_vals_df.index, format='%Y-%m-%d')

    netatmo_in_vals_df = netatmo_in_vals_df.loc[strt_date:end_date, :]
    netatmo_in_vals_df.dropna(how='all', axis=0, inplace=True)

    cmn_stns = netatmo_in_coords_df.index.intersection(
        netatmo_in_vals_df.columns)

    netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_stns]

    # ppt data
    netatmo_in_ppt_vals_df = pd.read_csv(
        path_to_netatmo_ppt, sep=';',
        index_col=0,
        encoding='utf-8',
        engine='c')

    netatmo_in_ppt_vals_df.index = pd.to_datetime(
        netatmo_in_ppt_vals_df.index,
        format='%Y-%m-%d')

    netatmo_in_ppt_vals_df.dropna(how='all', axis=0, inplace=True)

    netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.loc[:, cmn_stns]
    # #####
    if use_netatmo_gd_stns:
        good_netatmo_stns = df_gd_stns.loc[
            :, 'Stations'].values.ravel()

        cmn_gd_stns = netatmo_in_vals_df.columns.intersection(
            good_netatmo_stns)
        netatmo_in_vals_df_gd = netatmo_in_vals_df.loc[
            :, cmn_gd_stns]
        netatmo_in_ppt_vals_df_gd = netatmo_in_ppt_vals_df.loc[:, cmn_gd_stns]
    else:
        netatmo_in_vals_df_gd = netatmo_in_vals_df
        netatmo_in_ppt_vals_df_gd = netatmo_in_ppt_vals_df
    # DWD Extremes
    #=========================================================================
    dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,  # path_dwd_extremes_df
                                     index_col=0,
                                     sep=';',
                                     parse_dates=True,
                                     infer_datetime_format=True,
                                     encoding='utf-8',
                                     header=None)

    df_vgs_extremes = pd.read_csv(path_to_dwd_vgs,
                                  index_col=0,
                                  sep=';',
                                  parse_dates=True,
                                  infer_datetime_format=True,
                                  encoding='utf-8',
                                  skiprows=[0],
                                  header=None).dropna(how='all')

    # number events with good vg
    evts = []
    df_vgs = pd.DataFrame(index=dwd_in_extremes_df.index,
                          columns=['vg_model'])
    for ix in df_vgs_extremes.index:
        for val in df_vgs_extremes.loc[ix, :].values:
            if isinstance(val, str):
                if 'Nug' not in val:
                    df_vgs.loc[ix, 'vg_model'] = val
                    break
    df_vgs.dropna(how='all', inplace=True)

    dwd_in_extremes_df = dwd_in_extremes_df.loc[
        dwd_in_extremes_df.index.intersection(
            df_vgs.index).intersection(
            df_radar_events_to_keep.index), :]

    dwd_in_extremes_df = dwd_in_extremes_df.loc[strt_date:end_date, :]
#     dwd_in_extremes_df = dwd_in_extremes_df.loc['2018-05-31':'2018-06-01':]
    print('\n%d Intense Event with gd VG to interpolate\n'
          % dwd_in_extremes_df.shape[0])

    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    # all_dwd_stns = dwd_in_vals_df.columns.tolist()

    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()
    # TODO: FOS
    for event_date in dwd_in_extremes_df.index:  # hourly_events:  #

        #         if str(event_date) == '2019-07-28 13:00:00':
        #             pass
        #         break
        #         # hourly_events daily_events  # == '2018-12-23 00:00:00':
        # dwd_in_extremes_df.index:
        # dwd_in_extremes_df.index:
        if str(event_date) in dwd_in_extremes_df.index:  # :
            #             event_date = '2016-06-24 22:00:00'

            radar_evt = df_radar_events_to_keep.loc[event_date, '0']

            #event_date = pd.DatetimeIndex([event_date])
            print(event_date)

            # read radolan file
            rwdata, rwattrs = wrl.io.read_radolan_composite(radar_evt)
            # mask data
            sec = rwattrs['secondary']
            rwdata.flat[sec] = -9999
            rwdata = np.ma.masked_equal(rwdata, -9999)

            # create radolan projection object
            proj_stereo = wrl.georef.create_osr("dwd-radolan")

            # create wgs84 projection object
            proj_wgs = osr.SpatialReference()
            proj_wgs.ImportFromEPSG(4326)

            # get radolan grid
            radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
            x1 = radolan_grid_xy[:, :, 0]
            y1 = radolan_grid_xy[:, :, 1]

            # convert to lonlat
            radolan_grid_ll = wrl.georef.reproject(radolan_grid_xy,
                                                   projection_source=proj_stereo,
                                                   projection_target=proj_wgs)

            radolan_grid_ll[mask] = -1

            lon1 = radolan_grid_ll[:, :, 0]
            lat1 = radolan_grid_ll[:, :, 1]

#             rwdata[mask] = -1

            lon1_maskes = lon1[~mask]
            lat1_maskes = lat1[~mask]
            rw_maskes = rwdata[~mask]

#             rw_maskes = np.ma.masked_array(rwdata, rwdata < 0.)
            rw_maskes[rw_maskes.data < 0.] = np.nan

            radolan_coords = np.array([(lo, la) for lo, la in zip(
                lon1_maskes, lat1_maskes)])

            radolan_coords_tree = cKDTree(radolan_coords)

            df_ppt_radolan = pd.DataFrame(
                index=range(len(grid_interp_df.index)))

            #print('\nGetting station data\n')
            ix_lon_loc, ix_lat_loc, ppt_locs = [], [], []

            for ix, (x0, y0) in enumerate(zip(lon_coords_grd, lat_coords_grd)):

                dd, ii = radolan_coords_tree.query([x0, y0], k=1)

                grd_lon_loc = lon1_maskes[ii]
                grd_lat_loc = lat1_maskes[ii]
                ppt_loc = rw_maskes[ii]

                df_ppt_radolan.loc[ix, 'xlon'] = grd_lon_loc
                df_ppt_radolan.loc[ix, 'ylat'] = grd_lat_loc
                df_ppt_radolan.loc[ix, 'ppt'] = ppt_loc

            # df_ppt_radolan.ppt.plot()
            # plt.plot(rw_maskes.data)

            #print('Done getting station data\n')

            #print('Plotting extracted Radolan data and coordinates')

            #_stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
#             if len(_stn_id_event_) < 5:
#                 _stn_id_event_ = (5 - len(_stn_id_event_)) * \
#                     '0' + _stn_id_event_

            _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]

            #==============================================================
            # # DWD PPT
            #==============================================================

            # dwd vals and coords
            edf_dwd_vals = dwd_in_vals_df.loc[event_date, :].dropna().values
            ppt_dwd_vals_sr = dwd_in_ppt_vals_df.loc[
                event_date, :].dropna()

            ppt_dwd_vals = ppt_dwd_vals_sr.values
            dwd_xcoords = dwd_in_coords_df.loc[
                ppt_dwd_vals_sr.index, 'X'].values
            dwd_ycoords = dwd_in_coords_df.loc[
                ppt_dwd_vals_sr.index, 'Y'].values
            dwd_stn_ids = ppt_dwd_vals_sr.index.to_list()

#             ppt_dwd_vals = []
#             dwd_xcoords = []
#             dwd_ycoords = []
#             dwd_stn_ids = []
#
#             for stn_id in all_dwd_stns:
#                 #print('station is', stn_id)
#
#                 ppt_stn_vals = dwd_in_ppt_vals_df.loc[
#                     event_date, stn_id]
#                 if ppt_stn_vals >= 0:
#                     ppt_dwd_vals.append(np.round(ppt_stn_vals, 4))
#                     dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
#                     dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
#                     dwd_stn_ids.append(stn_id)

#             dwd_xcoords = np.array(dwd_xcoords)
#             dwd_ycoords = np.array(dwd_ycoords)
#             ppt_dwd_vals = np.array(ppt_dwd_vals)

            vgs_model_dwd_ppt = df_vgs.loc[event_date, 'vg_model']

            try:
                vgs_model_dwd_ppt = vgs_model_dwd_ppt.values[0]
            except Exception:
                vgs_model_dwd_ppt = vgs_model_dwd_ppt

            if ('Exp' in vgs_model_dwd_ppt or
                    'Sph' in vgs_model_dwd_ppt):

                # print(vgs_model_dwd_ppt)
                #print('\n+++ KRIGING PPT at DWD +++\n')

                vg_sill = float(vgs_model_dwd_ppt.split(" ")[0])
                dwd_vals_var = np.var(ppt_dwd_vals)
                vg_scaling_ratio = dwd_vals_var / vg_sill

                if vg_scaling_ratio == 0:
                    vg_scaling_ratio = 1
                # netatmo stns for this event
                # Netatmo data and coords

                #vg_scaling_ratio = 1

                netatmo_df = netatmo_in_ppt_vals_df.loc[
                    event_date, :].dropna(how='all')

                #==========================================================
                # NO FILTER USED
                #==========================================================

                #print('NETATMO NOT FILTERED')

                netatmo_ppt_vals_fr_dwd_interp = netatmo_df.values

                x_netatmo_ppt_vals_fr_dwd_interp = netatmo_in_coords_df.loc[
                    netatmo_df.index, 'X'].values
                y_netatmo_ppt_vals_fr_dwd_interp = netatmo_in_coords_df.loc[
                    netatmo_df.index, 'Y'].values

                #======================================================
                # Transform everything to arrays and combine
                # dwd-netatmo

                netatmo_xcoords0 = np.array(
                    x_netatmo_ppt_vals_fr_dwd_interp).ravel()
                netatmo_ycoords0 = np.array(
                    y_netatmo_ppt_vals_fr_dwd_interp).ravel()

                ppt_netatmo_vals = np.round(np.array(
                    netatmo_ppt_vals_fr_dwd_interp).ravel(), 2)

#                 netatmo_dwd_x_coords = np.concatenate([netatmo_xcoords0,
#                                                        dwd_xcoords])
#                 netatmo_dwd_y_coords = np.concatenate([netatmo_ycoords0,
#                                                        dwd_ycoords])
#                 netatmo_dwd_ppt_vals = np.round(np.hstack(
#                     (ppt_netatmo_vals,
#                      ppt_dwd_vals)), 2).ravel()

                #======================================================
                # # Krigging PPT
                #======================================================
                # using DWD data
                ordinary_kriging_dwd_ppt = OrdinaryKriging(
                    xi=dwd_xcoords,
                    yi=dwd_ycoords,
                    zi=ppt_dwd_vals,
                    xk=x_coords_grd,
                    yk=y_coords_grd,
                    model=vgs_model_dwd_ppt)

                # using Netatmo data
                ordinary_kriging_netatmo_ppt = OrdinaryKriging(
                    xi=netatmo_xcoords0,
                    yi=netatmo_ycoords0,
                    zi=ppt_netatmo_vals,
                    xk=x_coords_grd,
                    yk=y_coords_grd,
                    model=vgs_model_dwd_ppt)

                print('\nOK using %s DWD' % ppt_dwd_vals.size)
                ordinary_kriging_dwd_ppt.krige()

                print('\nOK using %d Netatmo' % netatmo_df.values.size)
                ordinary_kriging_netatmo_ppt.krige()

                interpolated_vals_dwd_only = ordinary_kriging_dwd_ppt.zk.copy()
                interpolated_vals_netatmo_only = ordinary_kriging_netatmo_ppt.zk.copy()

                interpolated_vals_dwd_only[
                    interpolated_vals_dwd_only < 0] = 0

                interpolated_vals_netatmo_only[
                    interpolated_vals_netatmo_only < 0] = 0

                #==========================================================
                # FIRST AND SECOND FILTER
                #==========================================================

                #print('\n**using Netatmo gd stns**')

                netatmo_df_gd = netatmo_in_ppt_vals_df_gd.loc[
                    event_date, :].dropna(how='all')
                netatmo_edf = netatmo_in_vals_df_gd.loc[
                    event_date, :].dropna(how='all')
                print('%d Netatmo 1st and 2nd Filter' % netatmo_df_gd.size)

                netatmo_xcoords = netatmo_in_coords_df.loc[
                    netatmo_df_gd.index, 'X'].values.ravel()
                netatmo_ycoords = netatmo_in_coords_df.loc[
                    netatmo_df_gd.index, 'Y'].values.ravel()

                ids_netatmo_stns_gd = netatmo_df_gd.index

                if use_on_event_filter:
                    ordinary_kriging_filter_netamto = OrdinaryKriging(
                        xi=dwd_xcoords,
                        yi=dwd_ycoords,
                        zi=edf_dwd_vals,
                        xk=netatmo_xcoords,
                        yk=netatmo_ycoords,
                        model=vgs_model_dwd_ppt)

                    try:
                        ordinary_kriging_filter_netamto.krige()
                    except Exception as msg:
                        print('Error while Error Kriging', msg)

                    # interpolated vals
                    interpolated_vals = ordinary_kriging_filter_netamto.zk

                    # calcualte standard deviation of estimated values
                    std_est_vals = np.sqrt(
                        ordinary_kriging_filter_netamto.est_vars)
                    # calculate difference observed and estimated
                    # values
                    diff_obsv_interp = np.abs(
                        netatmo_edf - interpolated_vals)

                    #==================================================
                    # # use additional temporal filter
                    #==================================================
                    idx_good_stns = np.where(
                        diff_obsv_interp <= 3 * std_est_vals)
                    idx_bad_stns = np.where(
                        diff_obsv_interp > 3 * std_est_vals)

                    if len(idx_bad_stns[0]) or len(idx_good_stns[0]) > 0:
                        #                     print('Number of Stations with bad index \n',
                        #                           len(idx_bad_stns[0]))
                        #                     print('Number of Stations with good index \n',
                        #                           len(idx_good_stns[0]))

                        #                                 print('**Removing bad stations and kriging**')

                        # use additional filter
                        try:
                            ids_netatmo_stns_gd = np.take(netatmo_df_gd.index,
                                                          idx_good_stns).ravel()
                            ids_netatmo_stns_bad = np.take(netatmo_df_gd.index,
                                                           idx_bad_stns).ravel()

                        except Exception as msg:
                            print(msg)

                    assert idx_good_stns[0].size == ids_netatmo_stns_gd.size
                    assert idx_bad_stns[0].size == ids_netatmo_stns_bad.size

                # netatmo stations to correct
                if use_on_event_filter_check_drywet:
                    # apply on event filter
                    ordinary_kriging_filter_netamto = OrdinaryKriging(
                        xi=dwd_xcoords,
                        yi=dwd_ycoords,
                        zi=edf_dwd_vals,
                        xk=netatmo_xcoords,
                        yk=netatmo_ycoords,
                        model=vgs_model_dwd_ppt)

                    try:
                        ordinary_kriging_filter_netamto.krige()
                    except Exception as msg:
                        print('Error while Error Kriging', msg)

                    # interpolated vals
                    interpolated_vals = ordinary_kriging_filter_netamto.zk

                    # calcualte standard deviation of estimated values
                    std_est_vals = np.sqrt(
                        ordinary_kriging_filter_netamto.est_vars)
                    # calculate difference observed and estimated
                    # values
                    diff_obsv_interp = np.abs(
                        netatmo_edf - interpolated_vals)

                    #==================================================
                    # # use additional temporal filter
                    #==================================================
                    idx_good_stns = np.where(
                        diff_obsv_interp <= 3 * std_est_vals)
                    idx_bad_stns = np.where(
                        diff_obsv_interp > 3 * std_est_vals)

                    if len(idx_bad_stns[0]) or len(idx_good_stns[0]) > 0:
                        #                     print('Number of Stations with bad index \n',
                        #                           len(idx_bad_stns[0]))
                        #                     print('Number of Stations with good index \n',
                        #                           len(idx_good_stns[0]))

                        #                                 print('**Removing bad stations and kriging**')

                        # use additional filter
                        try:
                            ids_netatmo_stns_gd = np.take(netatmo_df_gd.index,
                                                          idx_good_stns).ravel()
                            ids_netatmo_stns_bad = np.take(netatmo_df_gd.index,
                                                           idx_bad_stns).ravel()

                        except Exception as msg:
                            print(msg)

                    assert idx_good_stns[0].size == ids_netatmo_stns_gd.size
                    assert idx_bad_stns[0].size == ids_netatmo_stns_bad.size

                    try:
                        edf_gd_vals_df = netatmo_edf.loc[ids_netatmo_stns_gd]
                        ppt_gd_vals_df = netatmo_df_gd.loc[ids_netatmo_stns_gd]

                        edf_bad_vals_df = netatmo_edf.loc[ids_netatmo_stns_bad]
                    except Exception as msg:
                        print(msg, 'error while second filter')

                    # coords of Netatmo neighbors with good values

                    netatmo_x_stns_gd = netatmo_in_coords_df.loc[
                        ids_netatmo_stns_gd, 'X'].values
                    netatmo_y_stns_gd = netatmo_in_coords_df.loc[
                        ids_netatmo_stns_gd, 'Y'].values

                    netatmo_coords = np.array(
                        [(x, y) for x, y in zip(netatmo_x_stns_gd,
                                                netatmo_y_stns_gd)])
                    # coords of DWD
                    dwd_coords = np.array(
                        [(x, y) for x, y in zip(dwd_xcoords, dwd_ycoords)])
                    neighbors_coords_dwd_netatmo = np.concatenate((netatmo_coords,
                                                                   dwd_coords))
                    # create a tree from coordinates
                    points_tree = spatial.KDTree(neighbors_coords_dwd_netatmo)

                    # get the ppt and edf data
                    edf_netatmo_dwd_vals = np.concatenate((edf_gd_vals_df.values,
                                                           edf_dwd_vals))
                    ppt_netatmo_dwd_vals = np.concatenate((ppt_gd_vals_df.values,
                                                           ppt_dwd_vals))
                    #==========================================================
                    # check if bad are really bad, look at neighborhood
                    #==========================================================
                    for stn_ in edf_bad_vals_df.index:

                        # coords of stns self
                        netatmo_x_stn = netatmo_in_coords_df.loc[stn_, 'X']
                        netatmo_y_stn = netatmo_in_coords_df.loc[stn_, 'Y']

                        stn_coords = np.array([(netatmo_x_stn,
                                                netatmo_y_stn)])
                        # ppt and edf of stn
                        ppt_stn = netatmo_in_ppt_vals_df.loc[event_date, stn_]
                        edf_stn = netatmo_in_vals_df.loc[event_date, stn_]

                        # This finds the index of all points within
                        # radius of 5 km
                        idxs_neighbours = points_tree.query_ball_point(
                            np.array((netatmo_x_stn, netatmo_y_stn)), 5e3)

                        # if there are any neighbors
                        if len(idxs_neighbours) > 0:
                            # go through neighbors and check if correct
                            edf_all_ngbrs = edf_netatmo_dwd_vals[idxs_neighbours]
                            try:
                                ppt_all_ngbrs = ppt_netatmo_dwd_vals[idxs_neighbours]
                            except Exception as msg:
                                print(msg)
                            # this means that the station and all of its neighbors are
                            # eiter wet or dry but not conflicting !

                            if min(edf_all_ngbrs.min(), edf_stn) > edf_thr:
                                # all are wet
                                if stn_ not in ids_netatmo_stns_gd:

                                    ids_netatmo_stns_gd = np.append(
                                        ids_netatmo_stns_gd,
                                        stn_)
                                    ids_netatmo_stns_bad = list(
                                        filter(lambda x: x != stn_,
                                               ids_netatmo_stns_bad))

                                    #print('added bad wet to good stns \n')
                            if max(edf_all_ngbrs.min(), edf_stn) < edf_thr:
                                # all are dry
                                if stn_ not in ids_netatmo_stns_gd:

                                    ids_netatmo_stns_gd = np.append(
                                        ids_netatmo_stns_gd,
                                        stn_)
                                    ids_netatmo_stns_bad = list(
                                        filter(lambda x: x != stn_,
                                               ids_netatmo_stns_bad))
                                    #print('added bad dry to good stns \n')

                """
                try:
                    edf_gd_vals_df = netatmo_df_gd.loc[ids_netatmo_stns_gd]
                except Exception as msg:
                    print(msg, 'error while second filter')

                netatmo_dry_gd = edf_gd_vals_df[
                    edf_gd_vals_df.values < edf_thr]
                # gd

                cmn_wet_dry_stns = netatmo_in_coords_df.index.intersection(
                    netatmo_dry_gd.index)
                x_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
                    cmn_wet_dry_stns, 'X'].values.ravel()
                y_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
                    cmn_wet_dry_stns, 'Y'].values.ravel()

                netatmo_wet_gd = edf_gd_vals_df[
                    edf_gd_vals_df.values >= edf_thr]
                cmn_wet_gd_stns = netatmo_in_coords_df.index.intersection(
                    netatmo_wet_gd.index)

                x_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
                    cmn_wet_gd_stns, 'X'].values.ravel()
                y_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
                    cmn_wet_gd_stns, 'Y'].values.ravel()

                assert (netatmo_wet_gd.size ==
                        x_coords_gd_netatmo_wet.size ==
                        y_coords_gd_netatmo_wet.size)
                #====================================
                #
                #====================================
                edf_bad_vals_df = netatmo_df_gd.loc[ids_netatmo_stns_bad]

                netatmo_dry_bad = edf_bad_vals_df[
                    edf_bad_vals_df.values < edf_thr]
                cmn_dry_bad_stns = netatmo_in_coords_df.index.intersection(
                    netatmo_dry_bad.index)
                # netatmo bad
                netatmo_wet_bad = edf_bad_vals_df[
                    edf_bad_vals_df.values >= edf_thr]
                cmn_wet_bad_stns = netatmo_in_coords_df.index.intersection(
                    netatmo_wet_bad.index)
                # dry bad
                x_coords_bad_netatmo_dry = netatmo_in_coords_df.loc[
                    cmn_dry_bad_stns, 'X'].values.ravel()
                y_coords_bad_netatmo_dry = netatmo_in_coords_df.loc[
                    cmn_dry_bad_stns, 'Y'].values.ravel()
                # wet bad
                x_coords_bad_netatmo_wet = netatmo_in_coords_df.loc[
                    cmn_wet_bad_stns, 'X'].values.ravel()
                y_coords_bad_netatmo_wet = netatmo_in_coords_df.loc[
                    cmn_wet_bad_stns, 'Y'].values.ravel()

                # DWD data
                dwd_dry = edf_dwd_vals[edf_dwd_vals < edf_thr]
                dwd_wet = edf_dwd_vals[edf_dwd_vals >= edf_thr]

                x_coords_dwd_wet = dwd_xcoords[np.where(
                    edf_dwd_vals >= edf_thr)]
                y_coords_dwd_wet = dwd_ycoords[np.where(
                    edf_dwd_vals >= edf_thr)]

                assert (dwd_wet.size ==
                        x_coords_dwd_wet.size ==
                        y_coords_dwd_wet.size)

                x_coords_dwd_dry = dwd_xcoords[np.where(
                    edf_dwd_vals < edf_thr)]
                y_coords_dwd_dry = dwd_ycoords[np.where(
                    edf_dwd_vals < edf_thr)]

                # find if wet bad is really wet bad
                # find neighboring netatmo stations wet good

                # combine dwd and netatmo wet gd

                x_gd_dwd_netatmo = np.concatenate([
                    x_coords_gd_netatmo_wet,
                    x_coords_dwd_wet])
                y_gd_dwd_netatmo = np.concatenate([
                    y_coords_gd_netatmo_wet,
                    y_coords_dwd_wet])

                dwd_netatmo_wet_gd = np.concatenate([
                    netatmo_wet_gd,
                    dwd_wet])

                if netatmo_wet_gd.size > 0:

                    for stn_, edf_stn, netatmo_x_stn, netatmo_y_stn in zip(
                        netatmo_wet_bad.index,
                        netatmo_wet_bad.values,
                            x_coords_bad_netatmo_wet,
                            y_coords_bad_netatmo_wet):
                        #                         print('Trying to correct %s bad wet '
                        #                               % stn_)
                        # coords of stns self
                        stn_coords = np.array([(netatmo_x_stn,
                                                netatmo_y_stn)])

                        # coords of neighbors
                        neighbors_coords = np.array(
                            [(x, y) for x, y
                             in zip(x_gd_dwd_netatmo,
                                    y_gd_dwd_netatmo)])

                        # create a tree from coordinates
                        points_tree = spatial.KDTree(
                            neighbors_coords)

                        # This finds the index of all points within
                        # radius
                        idxs_neighbours = points_tree.query_ball_point(
                            np.array(
                                (netatmo_x_stn, netatmo_y_stn)),
                            1e4)
                        if len(idxs_neighbours) > 0:

                            for i, ix_nbr in enumerate(idxs_neighbours):

                                try:
                                    edf_neighbor = dwd_netatmo_wet_gd[ix_nbr]
                                except Exception as msg:
                                    print(msg)
                                    edf_neighbor = 1.1
                                if np.abs(edf_stn - edf_neighbor) <= diff_thr:
                                    #                                     print(
                                    #                                         'bad wet netatmo station is good')
                                    # add to good wet netatmos
                                    try:
                                        netatmo_wet_gd[stn_] = edf_stn
                                        ids_netatmo_stns_gd = np.append(
                                            ids_netatmo_stns_gd,
                                            stn_)
                                        x_coords_gd_netatmo_wet = np.append(
                                            x_coords_gd_netatmo_wet,
                                            netatmo_x_stn)
                                        y_coords_gd_netatmo_wet = np.append(
                                            y_coords_gd_netatmo_wet,
                                            netatmo_y_stn)

                                        # remove from original bad
                                        # wet
                                        x_coords_bad_netatmo_wet = np.array(list(
                                            filter(lambda x: x != netatmo_x_stn,
                                                   x_coords_bad_netatmo_wet)))

                                        y_coords_bad_netatmo_wet = np.array(list(
                                            filter(lambda x: x != netatmo_y_stn,
                                                   y_coords_bad_netatmo_wet)))

                                        netatmo_wet_bad.loc[stn_] = np.nan

                                        ids_netatmo_stns_bad = list(
                                            filter(lambda x: x != stn_,
                                                   ids_netatmo_stns_bad))
                                    except Exception as msg:
                                        print(msg)
                                        pass

                                else:
                                    pass
#                                     print('bad wet is bad wet')
                        else:
                            pass
#                             print('\nStn has no near neighbors')
                """
                # print('Number of Stations with bad index \n',
                #      len(idx_bad_stns[0]))
                # print('Number of Stations with good index \n',
                #      len(idx_good_stns[0]))

                #==================================================
                # Transformation
                #==================================================

                if use_tranformation:
                    netatmo_stns_event_gd = []
                    netatmo_ppt_vals_fr_dwd_interp_gd = []
                    x_netatmo_ppt_vals_fr_dwd_interp_gd = []
                    y_netatmo_ppt_vals_fr_dwd_interp_gd = []

                    # print('correcting Netatmo Quantiles')

                    for netatmo_stn_id in ids_netatmo_stns_gd:  # netatmo_df.index:

                        netatmo_edf_event_ = netatmo_in_vals_df_gd.loc[event_date,
                                                                       netatmo_stn_id]

                        netatmo_ppt_event_ = netatmo_in_ppt_vals_df_gd.loc[
                            event_date,
                            netatmo_stn_id]

                        x_netatmo_interpolate = np.array(
                            [netatmo_in_coords_df.loc[netatmo_stn_id, 'X']])
                        y_netatmo_interpolate = np.array(
                            [netatmo_in_coords_df.loc[netatmo_stn_id, 'Y']])

                        if netatmo_edf_event_ > 0.5:  # 0.99:
                            try:
                                #==========================================
                                # # Correct Netatmo Quantiles
                                #==========================================

                                netatmo_stn_edf_df = netatmo_in_vals_df_gd.loc[
                                    :, netatmo_stn_id].dropna()

                                netatmo_start_date = netatmo_stn_edf_df.index[0]
                                netatmo_end_date = netatmo_stn_edf_df.index[-1]

                                # select dwd stations with only same period as
                                # netatmo stn
                                ppt_dwd_stn_vals = dwd_in_ppt_vals_df.loc[
                                    netatmo_start_date:netatmo_end_date,
                                    :].dropna(how='all')

                                dwd_xcoords_new = []
                                dwd_ycoords_new = []
                                ppt_vals_dwd_new_for_obsv_ppt = []

                                for dwd_stn_id_new in ppt_dwd_stn_vals.columns:

                                    ppt_vals_stn_new = ppt_dwd_stn_vals.loc[
                                        :, dwd_stn_id_new].dropna()

                                    if ppt_vals_stn_new.size > 0:

                                        ppt_stn_new, edf_stn_new = build_edf_fr_vals(
                                            ppt_vals_stn_new)

                                        if netatmo_edf_event_ == 1.0:
                                            # correct edf event, 1 is bcz
                                            # rounding error
                                            ppt_ix_edf = find_nearest(
                                                ppt_stn_new,
                                                _ppt_event_)

                                            netatmo_edf_event_ = edf_stn_new[
                                                np.where(
                                                    ppt_stn_new == ppt_ix_edf)][0]

                                        nearst_edf_new = find_nearest(
                                            edf_stn_new,
                                            netatmo_edf_event_)

                                        ppt_idx = np.where(
                                            edf_stn_new == nearst_edf_new)

                                        try:
                                            if ppt_idx[0].size > 1:
                                                ppt_for_edf = np.mean(
                                                    ppt_stn_new[ppt_idx])
                                            else:
                                                ppt_for_edf = ppt_stn_new[ppt_idx]
                                        except Exception as msg:
                                            print(msg)

                                        if ppt_for_edf[0] >= 0:

                                            ppt_vals_dwd_new_for_obsv_ppt.append(
                                                ppt_for_edf[0])

                                            stn_xcoords_new = dwd_in_coords_df.loc[
                                                dwd_stn_id_new, 'X']
                                            stn_ycoords_new = dwd_in_coords_df.loc[
                                                dwd_stn_id_new, 'Y']

                                            dwd_xcoords_new.append(
                                                stn_xcoords_new)
                                            dwd_ycoords_new.append(
                                                stn_xcoords_new)

                                # get ppt from dwd recent for netatmo stns
                                ppt_vals_dwd_new_for_obsv_ppt = np.array(
                                    ppt_vals_dwd_new_for_obsv_ppt)
                                # for kriging with uncertainty

                                dwd_xcoords_new = np.array(dwd_xcoords_new)
                                dwd_ycoords_new = np.array(dwd_ycoords_new)

                                try:

                                    ordinary_kriging_dwd_netatmo_crt = OrdinaryKriging(
                                        xi=dwd_xcoords_new,
                                        yi=dwd_ycoords_new,
                                        zi=ppt_vals_dwd_new_for_obsv_ppt,
                                        xk=x_netatmo_interpolate,
                                        yk=y_netatmo_interpolate,
                                        model=vgs_model_dwd_ppt)

                                    try:
                                        ordinary_kriging_dwd_netatmo_crt.krige()
                                    except Exception as msg:
                                        print('Error while Kriging', msg)

                                    interpolated_netatmo_prct = ordinary_kriging_dwd_netatmo_crt.zk.copy()

                                    if interpolated_netatmo_prct < 0:
                                        interpolated_netatmo_prct = np.nan

    #                                     print('**Interpolated PPT by DWD recent: \n',
    #                                           interpolated_netatmo_prct)

                                    if interpolated_netatmo_prct >= 0.:
                                        netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                            interpolated_netatmo_prct[0])

                                        x_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                            x_netatmo_interpolate[0])

                                        y_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                            y_netatmo_interpolate[0])

                                        netatmo_stns_event_gd.append(
                                            netatmo_stn_id)
                                except Exception as msg:
                                    print(
                                        msg, 'Error when getting ppt from dwd interp')
                                    continue

                            except Exception as msg:
                                print(msg, 'Error when KRIGING')
                                continue

                        else:

                            if netatmo_ppt_event_ >= 0:
                                netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                    netatmo_ppt_event_)

                                x_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                    x_netatmo_interpolate[0])

                                y_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                    y_netatmo_interpolate[0])

                                netatmo_stns_event_gd.append(netatmo_stn_id)

                else:
                    netatmo_stns_event_gd = ids_netatmo_stns_gd
                    netatmo_ppt_vals_fr_dwd_interp_gd = netatmo_in_vals_df_gd.loc[
                        event_date, ids_netatmo_stns_gd]
                    x_netatmo_ppt_vals_fr_dwd_interp_gd = netatmo_in_coords_df.loc[
                        ids_netatmo_stns_gd, 'X']
                    y_netatmo_ppt_vals_fr_dwd_interp_gd = netatmo_in_coords_df.loc[
                        ids_netatmo_stns_gd, 'Y']
                #======================================================
                # Transform everything to arrays and combine
                # dwd-netatmo

                netatmo_xcoords = np.array(
                    x_netatmo_ppt_vals_fr_dwd_interp_gd).ravel()
                netatmo_ycoords = np.array(
                    y_netatmo_ppt_vals_fr_dwd_interp_gd).ravel()

                ppt_netatmo_vals_gd = np.round(np.array(
                    netatmo_ppt_vals_fr_dwd_interp_gd).ravel(), 2)

                netatmo_dwd_x_coords = np.concatenate([netatmo_xcoords,
                                                       dwd_xcoords])
                netatmo_dwd_y_coords = np.concatenate([netatmo_ycoords,
                                                       dwd_ycoords])
                netatmo_dwd_ppt_vals_gd = np.round(np.hstack(
                    (ppt_netatmo_vals_gd,
                     ppt_dwd_vals)), 2).ravel()

                print('\n%d Netatmo after filter' % ppt_netatmo_vals_gd.size)
                # ok unc
#                 ppt_unc_term_05perc = np.array([
#                     0.0005 * p for p in ppt_netatmo_vals_gd])
#                 uncert_dwd = np.zeros(
#                     shape=ppt_dwd_vals.shape)
#
#                 ppt_dwd_netatmo_vals_uncert_05perc = np.concatenate([
#                     ppt_unc_term_05perc,
#                     uncert_dwd])

                ordinary_kriging_dwd_netatmo_ppt_unc_05perc = OrdinaryKriging(
                    xi=netatmo_dwd_x_coords,
                    yi=netatmo_dwd_y_coords,
                    zi=netatmo_dwd_ppt_vals_gd,
                    xk=x_coords_grd,
                    yk=y_coords_grd,
                    model=vgs_model_dwd_ppt)

                ordinary_kriging_dwd_netatmo_ppt_unc_05perc.krige()

                interpolated_vals_dwd_netatmo_unc = (
                    ordinary_kriging_dwd_netatmo_ppt_unc_05perc.zk.copy())

                # put negative values to 0
                interpolated_vals_dwd_netatmo_unc[
                    interpolated_vals_dwd_netatmo_unc < 0] = 0

                #======================================================
                # # Krigging PPT
                #======================================================
                print('Plotting PPT after 1st and 2nd filter')
                # nbr of used stns
                nbr_netatmo = netatmo_ppt_vals_fr_dwd_interp.size
                nbr_netatmo_flt = ppt_netatmo_vals_gd.size
                nbr_dwd = ppt_dwd_vals_sr.size
                # difference maps
                dwd_min_dwd_netatmo = interpolated_vals_dwd_netatmo_unc - interpolated_vals_dwd_only
                dwd_min_radolan = df_ppt_radolan.ppt.data - interpolated_vals_dwd_only
                dwd_min_netatmo = interpolated_vals_netatmo_only - interpolated_vals_dwd_only

                plt.ioff()

                plot_all_interplations_subplots(
                    vals_to_plot_dwd_netatmo=interpolated_vals_dwd_netatmo_unc,
                    vals_to_plot_dwd=interpolated_vals_dwd_only,
                    vals_to_plot_netatmo=interpolated_vals_netatmo_only,
                    vals_to_plot_dwd_min_dwd_netatmo=dwd_min_dwd_netatmo,
                    vals_to_plot_dwd_min_radolan=dwd_min_radolan,
                    vals_to_plot_dwd_min_netatmo=dwd_min_netatmo,
                    out_plot_path=out_plots_path,
                    temp_agg=temp_agg,
                    event_date=event_date,
                    radar_data=df_ppt_radolan.ppt.values,
                    radar_lon=df_ppt_radolan.xlon.values,
                    radar_lat=df_ppt_radolan.ylat.values,
                    save_acc='sqs',
                    nbr_netatmo=nbr_netatmo,
                    nbr_dwd=nbr_dwd,
                    nbr_netatmo_flt=nbr_netatmo_flt)


stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
