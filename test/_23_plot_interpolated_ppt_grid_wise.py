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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from spinterps import (OrdinaryKriging)
from spinterps import variograms


from pathlib import Path

VG = variograms.vgs.Variogram


plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})


# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
#main_dir = Path(r'/home/abbas/Documents/Python/Extremes')
# main_dir = Path(r'/home/IWS/hachem/Extremes')
os.chdir(main_dir)

path_to_data = main_dir / r'NetAtmo_BW'

path_to_vgs = main_dir / r'kriging_ppt_netatmo'

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# path for data filter
in_filter_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'


# distance mtx netatmo dwd
distance_matrix_netatmo_dwd_df_file = path_to_data / \
    r'distance_mtx_in_m_NetAtmo_DWD.csv'

# path_to_dwd_stns_comb
path_to_dwd_stns_comb = in_filter_path / r'dwd_combination_to_use.csv'

# path for interpolation grid
path_grid_interpolate = in_filter_path / \
    r"coords_interpolate_small.csv"  # _small

#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================

qunatile_kriging = True

# run it to filter Netatmo
use_netatmo_gd_stns = False  # general filter, Indicator kriging
use_temporal_filter_after_kriging = False  # on day filter

use_first_neghbr_as_gd_stns = False  # False
use_first_and_second_nghbr_as_gd_stns = False  # True

_acc_ = ''

if use_first_neghbr_as_gd_stns:
    _acc_ = '1st'

if use_first_and_second_nghbr_as_gd_stns:
    _acc_ = 'comb'

if use_netatmo_gd_stns:
    path_to_netatmo_gd_stns = (main_dir / r'plots_NetAtmo_ppt_DWD_ppt_correlation_' /
                               (r'keep_stns_all_neighbor_99_per_60min_s0_%s.csv'
                                % _acc_))
# for second filter
path_to_dwd_ratios = in_filter_path / 'ppt_ratios_'

#==============================================================================
#
#==============================================================================
resample_frequencies = ['14400min']
# '120min', '180min', '60min',  '360min',
#                         '720min',
title_ = r'Ppt_ok_ok_un_2'


if not use_netatmo_gd_stns:
    title_ = title_ + '_netatmo_no_flt_'

if use_netatmo_gd_stns:
    title_ = title_ + '_first_flt_'

if use_temporal_filter_after_kriging:

    title_ = title_ + '_temp_flt_'

#==============================================================================
#
#==============================================================================

strt_date = '2015-01-01 00:00:00'
end_date = '2019-09-01 00:00:00'

# min_valid_stns = 20

drop_stns = []
mdr = 0.9
perm_r_list_ = [1, 2]
fit_vgs = ['Sph', 'Exp']  # 'Sph',
fil_nug_vg = 'Nug'  # 'Nug'
n_best = 4
ngp = 5


idx_time_fmt = '%Y-%m-%d %H:%M:%S'

radius = 10000
diff_thr = 0.1

hourly_events = [  # '2016-06-25 00:00:00',
    '2018-06-11 16:00:00',
    '2018-06-11 17:00:00',
    '2018-06-11 18:00:00',
    '2018-09-23 17:00:00',
    '2018-09-23 18:00:00',
    '2018-09-23 19:00:00']

daily_events = [  # '2018-12-23 00:00:00',
    #'2019-05-22 00:00:00',
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
y_coords_grd = grid_interp_df.loc[:, 'Y'].values.ravel()


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
# added by Abbas, for DWD stations
stndwd_ix = ['0' * (5 - len(str(stn_id))) + str(stn_id)
             if len(str(stn_id)) < 5 else str(stn_id)
             for stn_id in dwd_in_coords_df.index]

dwd_in_coords_df.index = stndwd_ix
dwd_in_coords_df.index = list(map(str, dwd_in_coords_df.index))


# Netatmo first filter
if use_netatmo_gd_stns:
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


def select_season(df,  # df to slice, index should be datetime
                  month_lst  # list of month for convective season
                  ):
    """
    return dataframe without the data corresponding to the winter season
    """
    df = df.copy()
    df_conv_season = df[df.index.month.isin(month_lst)]

    return df_conv_season


#==============================================================================
# SELECT GROUP OF 10 DWD STATIONS RANDOMLY
#==============================================================================


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


cmap_data = ['darkblue', 'blue', 'deepskyblue', 'lightblue',
             'darkgreen', 'olive',
             'lightgreen', 'greenyellow', 'yellow',
             'gold', 'orange', 'darkorange', 'orangered'][::-1]

cmap = mcolors.ListedColormap(cmap_data, 'precipitation')


def plot_interp_ppt_evnt(vals_to_plot, str_title,
                         out_plot_path,
                         temp_agg,
                         event_date):
    '''plot interpolated events, grid wise '''

    plt.figure(figsize=(12, 8), dpi=200)

    plt.scatter(x_coords_grd, y_coords_grd,
                c=vals_to_plot,
                marker=',', s=40, cmap=cmap,
                vmin=min(vals_to_plot), norm=norm,
                vmax=max(vals_to_plot))

    plt.colorbar(cmap=cmap,
                 norm=norm,
                 ticks=bound,
                 label='mm/%s' % temp_agg)

    if 'DWD' not in str_title and 'Netatmo' in str_title:
        plt.scatter(netatmo_xcoords, netatmo_ycoords, c='m',
                    marker='1', s=10, label='Netatmo', alpha=0.85)

    if 'DWD' in str_title and 'Netatmo' not in str_title:
        plt.scatter(dwd_xcoords, dwd_ycoords, c='darkgreen',
                    marker='x', s=10, label='DWD', alpha=0.85)

    if 'DWD' in str_title and 'Netatmo' in str_title:
        plt.scatter(dwd_xcoords, dwd_ycoords, c='darkgreen',
                    marker='x', s=10, label='DWD')
        plt.scatter(netatmo_xcoords, netatmo_ycoords, c='m',
                    marker='1', s=10, label='Netatmo', alpha=0.85)
    plt.legend(loc=0)
    plt.title('Event Date ' + str(
        event_date) + ' Using %s' % (str_title))
#     plt.grid(alpha=.25)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
    plt.tick_params(axis='both', which='both', bottom='False', top='False',
                    labelbottom='False', right='False',
                    left='False', labelleft='False')
#     plt.axis('equal')

    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.box(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig((
        out_plot_path / (
                '%s_%s_%s_event_test' %
                (str_title, temp_agg,
                 str(event_date).replace(
                     '-', '_').replace(':',
                                       '_').replace(' ', '_')
                 ))),
                papertype='a4',
                bbox_inches='tight',
                pad_inches=0.05)
    plt.close()
    return

#==============================================================================
#
#==============================================================================



#==============================================================================
#
#==============================================================================
# read distance matrix dwd-netamot ppt
in_df_distance_netatmo_dwd = pd.read_csv(
    distance_matrix_netatmo_dwd_df_file, sep=';', index_col=0)

# read df combinations to use
df_dwd_stns_comb = pd.read_csv(
    path_to_dwd_stns_comb, index_col=0,
    sep=',', dtype=str)

#==============================================================================
#
#==============================================================================
for temp_agg in resample_frequencies:

    # out path directory

    dir_path = title_ + '_' + _acc_ + '_' + temp_agg
    #dir_path = title_ + temp_agg
    out_plots_path = in_filter_path / dir_path

    if not os.path.exists(out_plots_path):
        os.mkdir(out_plots_path)
    print(out_plots_path)

    # path to data
    #=========================================================================
    out_save_csv = '_%s_ppt_edf_' % temp_agg

    path_to_dwd_edf = (path_to_data /
                       (r'edf_ppt_all_dwd_%s_.csv' % temp_agg))

    path_to_dwd_ppt = (path_to_data /
                       (r'ppt_all_dwd_%s_.csv' % temp_agg))

    path_to_dwd_edf_old = (path_to_data /
                           (r'edf_ppt_all_dwd_old_%s_.csv' % temp_agg))

    path_to_dwd_ppt_old = (path_to_data /
                           (r'ppt_all_dwd_old_%s_.csv' % temp_agg))
    path_dwd_ratios = (path_to_dwd_ratios /
                       (r'dwd_ratios_%s_data.csv' % temp_agg))
    path_to_netatmo_edf = (path_to_data /
                           (r'edf_ppt_all_netatmo_%s_.csv' % temp_agg))

    path_to_netatmo_ppt = (path_to_data /
                           (r'ppt_all_netatmo_%s_.csv' % temp_agg))

    path_to_dwd_vgs = (path_to_vgs /
                       (r'vg_strs_dwd_%s_maximum_100_event.csv' % temp_agg))

    path_dwd_extremes_df = path_to_data / \
        (r'dwd_%s_maximum_100_event.csv' % temp_agg)

    # Files to use
    # =========================================================================

    if qunatile_kriging:
        netatmo_data_to_use = path_to_netatmo_edf
        dwd_data_to_use = path_to_dwd_edf
        path_to_dwd_vgs = path_to_dwd_vgs

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

    dwd_in_vals_edf_old = pd.read_csv(
        path_to_dwd_edf_old, sep=';', index_col=0, encoding='utf-8')

    dwd_in_vals_edf_old.index = pd.to_datetime(
        dwd_in_vals_edf_old.index, format='%Y-%m-%d')

    dwd_in_vals_edf_old.dropna(how='all', axis=0, inplace=True)
    # dwd ppt old
    dwd_in_ppt_vals_df_old = pd.read_csv(
        path_to_dwd_ppt_old, sep=';', index_col=0, encoding='utf-8')

    dwd_in_ppt_vals_df_old.index = pd.to_datetime(
        dwd_in_ppt_vals_df_old.index, format='%Y-%m-%d')

    # DWD ratios for second filter
    dwd_ratios_df = pd.read_csv(path_dwd_ratios,
                                index_col=0,
                                sep=';',
                                parse_dates=True,
                                infer_datetime_format=True,
                                encoding='utf-8')

    # NETAMO DATA
    #=========================================================================
    netatmo_in_vals_df = pd.read_csv(
        path_to_netatmo_edf, sep=';',
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
        encoding='utf-8', engine='c')

    netatmo_in_ppt_vals_df.index = pd.to_datetime(
        netatmo_in_ppt_vals_df.index, format='%Y-%m-%d')

    netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.loc[strt_date:end_date, :]
    netatmo_in_ppt_vals_df.dropna(how='all', axis=0, inplace=True)

    netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.loc[:, cmn_stns]

    # apply first filter
    if use_netatmo_gd_stns:
        print('\n**using Netatmo gd stns**')
        good_netatmo_stns = df_gd_stns.loc[:, 'Stations'].values.ravel()
        cmn_gd_stns = netatmo_in_vals_df.columns.intersection(
            good_netatmo_stns)
        netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_gd_stns]
        netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.loc[:, cmn_gd_stns]

    # DWD Extremes
    #=========================================================================
    dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,
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

    dwd_in_extremes_df = dwd_in_extremes_df.loc[
        dwd_in_extremes_df.index.intersection(
            netatmo_in_vals_df.index).intersection(
            df_vgs_extremes.index), :]

    cmn_netatmo_stns = in_df_distance_netatmo_dwd.index.intersection(
        netatmo_in_vals_df.columns)
    in_df_distance_netatmo_dwd = in_df_distance_netatmo_dwd.loc[
        cmn_netatmo_stns, :]

    print('\n%d Extreme Event to interpolate\n' % dwd_in_extremes_df.shape[0])
    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()

    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()

    for event_date in dwd_in_extremes_df.index:

        # hourly_events daily_events  # == '2018-12-23 00:00:00':
        if str(event_date) in hourly_events:
            print(event_date)
            _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
            if len(_stn_id_event_) < 5:
                _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                    '0' + _stn_id_event_

            _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
            _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]

            # find minimal and maximal ratio for filtering netatmo
            min_ratio = dwd_ratios_df.loc[event_date, :].min()
            max_ratio = dwd_ratios_df.loc[event_date, :].max()

            print('**Calculating for Date ',
                  event_date, '\n Rainfall: ',  _ppt_event_,
                  'Quantile: ', _edf_event_, ' **\n')

            # CREATE DFS HOLD RESULT KRIGING PER NETATMO STATION
            df_interpolated = pd.DataFrame()
            df_interpolated['X'] = x_coords_grd
            df_interpolated['Y'] = y_coords_grd

            #==============================================================
            # # DWD PPT
            #==============================================================
            ppt_dwd_vals = []
            dwd_xcoords = []
            dwd_ycoords = []
            dwd_stn_ids = []

            for stn_id in all_dwd_stns:
                #print('station is', stn_id)

                ppt_stn_vals = dwd_in_ppt_vals_df.loc[
                    event_date, stn_id]

                if ppt_stn_vals >= 0:
                    ppt_dwd_vals.append(np.round(ppt_stn_vals, 4))
                    dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
                    dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
                    dwd_stn_ids.append(stn_id)

            dwd_xcoords = np.array(dwd_xcoords)
            dwd_ycoords = np.array(dwd_ycoords)
            ppt_dwd_vals = np.array(ppt_dwd_vals)

            # print(fit_vg_tst.describe())
            # fit_vg_tst.plot()
            # get vg model for this day
            vgs_model_dwd_ppt = df_vgs_extremes.loc[event_date, 1]

            if not isinstance(vgs_model_dwd_ppt, str):
                vgs_model_dwd_ppt = df_vgs_extremes.loc[event_date, 2]

            if not isinstance(vgs_model_dwd_ppt, str):
                vgs_model_dwd_ppt = ''

            if ('Nug' in vgs_model_dwd_ppt or len(
                vgs_model_dwd_ppt) == 0) and (
                'Exp' not in vgs_model_dwd_ppt and
                    'Sph' not in vgs_model_dwd_ppt):

                try:
                    for i in range(2, len(df_vgs_extremes.loc[event_date, :])):
                        vgs_model_dwd_ppt = df_vgs_extremes.loc[event_date, i]
                        if type(vgs_model_dwd_ppt) == np.float:
                            continue
                        if ('Nug' in vgs_model_dwd_ppt
                                or len(vgs_model_dwd_ppt) == 0) and (
                                    'Exp' not in vgs_model_dwd_ppt or
                                'Sph' not in vgs_model_dwd_ppt):
                            continue
                        else:
                            break

                except Exception as msg:
                    print(msg)
                    print(
                        'Only Nugget variogram for this day')
            print(vgs_model_dwd_ppt)

            if (isinstance(vgs_model_dwd_ppt, str)) and (
                    len(vgs_model_dwd_ppt) > 0) and (
                    'Exp' in vgs_model_dwd_ppt or
                        'Sph' in vgs_model_dwd_ppt):

                print('\n+++ KRIGING PPT at DWD +++\n')

                # netatmo stns for this event
                # Netatmo data and coords
                netatmo_df = netatmo_in_ppt_vals_df.loc[
                    event_date,
                    :].dropna(
                    how='all')

                if netatmo_df.size > 0:
                    #==========================================================
                    # NO FILTER USED
                    #==========================================================

                    if (not use_netatmo_gd_stns and
                            not use_temporal_filter_after_kriging):
                        print('NETATMO NOT FILTERED')
                        netatmo_ppt_vals_fr_dwd_interp = netatmo_df.values

                        x_netatmo_ppt_vals_fr_dwd_interp = netatmo_in_coords_df.loc[
                            netatmo_df.index, 'X'].values
                        y_netatmo_ppt_vals_fr_dwd_interp = netatmo_in_coords_df.loc[
                            netatmo_df.index, 'Y'].values

                        #======================================================
                        # Transform everything to arrays and combine
                        # dwd-netatmo

                        netatmo_xcoords = np.array(
                            x_netatmo_ppt_vals_fr_dwd_interp).ravel()
                        netatmo_ycoords = np.array(
                            y_netatmo_ppt_vals_fr_dwd_interp).ravel()

                        ppt_netatmo_vals = np.round(np.array(
                            netatmo_ppt_vals_fr_dwd_interp).ravel(), 2)

                        netatmo_dwd_x_coords = np.concatenate([netatmo_xcoords,
                                                               dwd_xcoords])
                        netatmo_dwd_y_coords = np.concatenate([netatmo_ycoords,
                                                               dwd_ycoords])
                        netatmo_dwd_ppt_vals = np.round(np.hstack(
                            (ppt_netatmo_vals,
                             ppt_dwd_vals)), 2).ravel()

                        #======================================================
                        # # Krigging PPT
                        #======================================================
                        # using Netatmo-DWD data
                        ordinary_kriging_dwd_netatmo_ppt = OrdinaryKriging(
                            xi=netatmo_dwd_x_coords,
                            yi=netatmo_dwd_y_coords,
                            zi=netatmo_dwd_ppt_vals,
                            xk=x_coords_grd,
                            yk=y_coords_grd,
                            model=vgs_model_dwd_ppt)

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
                            xi=netatmo_xcoords,
                            yi=netatmo_ycoords,
                            zi=ppt_netatmo_vals,
                            xk=x_coords_grd,
                            yk=y_coords_grd,
                            model=vgs_model_dwd_ppt)

                        print('\nOK using DWD-Netatmo')
                        ordinary_kriging_dwd_netatmo_ppt.krige()

                        print('\nOK using DWD')
                        ordinary_kriging_dwd_ppt.krige()

                        print('\nOK using Netatmo')
                        ordinary_kriging_netatmo_ppt.krige()

                        interpolated_vals_dwd_netatmo = ordinary_kriging_dwd_netatmo_ppt.zk.copy()
                        interpolated_vals_dwd_only = ordinary_kriging_dwd_ppt.zk.copy()
                        interpolated_vals_netatmo_only = ordinary_kriging_netatmo_ppt.zk.copy()

                        # put negative values to 0
                        interpolated_vals_dwd_netatmo[
                            interpolated_vals_dwd_netatmo < 0] = 0

                        interpolated_vals_dwd_only[
                            interpolated_vals_dwd_only < 0] = 0

                        interpolated_vals_netatmo_only[
                            interpolated_vals_netatmo_only < 0] = 0

                        # difference netatmo-dwd - dwd
                        diff_map_plus = interpolated_vals_dwd_only - interpolated_vals_dwd_netatmo
                        diff_map_plus2 = interpolated_vals_dwd_only - interpolated_vals_netatmo_only

                        bound = [0., 1,
                                 2, 5, 8,
                                 10, 15, 20,
                                 25, 30]  # , 35, 40, 45]

                        cmap = plt.get_cmap('jet')
                        norm = mcolors.BoundaryNorm(bound, cmap.N)
                        # interpolated_vals_dwd_netatmo
                        plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_dwd_netatmo,
                                             str_title=' DWD-Netatmo',
                                             out_plot_path=out_plots_path,
                                             temp_agg=temp_agg,
                                             event_date=event_date)

                        # interpolated_vals_dwd_only
                        plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_dwd_only,
                                             str_title=' DWD only',
                                             out_plot_path=out_plots_path,
                                             temp_agg=temp_agg,
                                             event_date=event_date)

                        # interpolated_vals_netatmo
                        plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_netatmo_only,
                                             str_title=' Netatmo',
                                             out_plot_path=out_plots_path,
                                             temp_agg=temp_agg,
                                             event_date=event_date)

                        bound = [-30, -15., -10,
                                 -5, -2, 0, 2,
                                 5, 10, 15,
                                 30]

                        cmap = plt.get_cmap('jet')
                        norm = mcolors.BoundaryNorm(bound, cmap.N)

                        plot_interp_ppt_evnt(vals_to_plot=diff_map_plus,
                                             str_title=' DWD minus DWD-Netatmo',
                                             out_plot_path=out_plots_path,
                                             temp_agg=temp_agg,
                                             event_date=event_date)

                        plot_interp_ppt_evnt(vals_to_plot=diff_map_plus2,
                                             str_title=' DWD minus Netatmo',
                                             out_plot_path=out_plots_path,
                                             temp_agg=temp_agg,
                                             event_date=event_date)

                    #==========================================================
                    # FIRST AND SECOND FILTER
                    #==========================================================

                    if (use_netatmo_gd_stns and
                            use_temporal_filter_after_kriging):

                        print('NETATMO 1st FILTERED and 2nd Filter')
                        netatmo_stns_event_ = []
                        netatmo_ppt_vals_fr_dwd_interp = []
    #                     netatmo_ppt_vals_fr_dwd_interp_unc = []

                        x_netatmo_ppt_vals_fr_dwd_interp = []
                        y_netatmo_ppt_vals_fr_dwd_interp = []
                        # netatmo stations to correct

                        for netatmo_stn_id in netatmo_df.index:

                            netatmo_edf_event_ = netatmo_in_vals_df.loc[
                                event_date,
                                netatmo_stn_id]

                            netatmo_ppt_event_ = netatmo_in_ppt_vals_df.loc[
                                event_date,
                                netatmo_stn_id]

                            x_netatmo_interpolate = np.array(
                                [netatmo_in_coords_df.loc[netatmo_stn_id, 'X']])
                            y_netatmo_interpolate = np.array(
                                [netatmo_in_coords_df.loc[netatmo_stn_id, 'Y']])

                            if netatmo_edf_event_ > 0.5:  # 0.99:
                                #                             print('Correcting Netatmo station',
                                #                                   netatmo_stn_id)
                                try:
                                    #==========================================
                                    # # Correct Netatmo Quantiles
                                    #==========================================

                                    netatmo_stn_edf_df = netatmo_in_vals_df.loc[
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
                                        #                                     print(
                                        #                                         '\n+++ KRIGING CORRECTING QT +++\n')

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

                                        if interpolated_netatmo_prct >= 0.:
                                            netatmo_ppt_vals_fr_dwd_interp.append(
                                                interpolated_netatmo_prct[0])

                                            x_netatmo_ppt_vals_fr_dwd_interp.append(
                                                x_netatmo_interpolate[0])

                                            y_netatmo_ppt_vals_fr_dwd_interp.append(
                                                y_netatmo_interpolate[0])

                                            netatmo_stns_event_.append(
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
                                    netatmo_ppt_vals_fr_dwd_interp.append(
                                        netatmo_ppt_event_)

    #                                     netatmo_ppt_vals_fr_dwd_interp_unc.append(
    #                                         netatmo_ppt_event_)
                                    x_netatmo_ppt_vals_fr_dwd_interp.append(
                                        x_netatmo_interpolate[0])

                                    y_netatmo_ppt_vals_fr_dwd_interp.append(
                                        y_netatmo_interpolate[0])

                                    netatmo_stns_event_.append(netatmo_stn_id)

                        #==================================================
                        # Applying second filter
                        #==================================================
                        stns_filtered = False
                        idxs_stns_remove = []
                        print('Applying second filter')

                        for ix_stn, netatmo_stn_to_test in enumerate(
                                netatmo_stns_event_):

                            x_netatmo_stn = np.array([netatmo_in_coords_df.loc[
                                netatmo_stn_to_test, 'X']])
                            y_netatmo_stn = np.array([netatmo_in_coords_df.loc[
                                netatmo_stn_to_test, 'Y']])

    #                             obs_netatmo_ppt_stn = netatmo_in_ppt_vals_df.loc[
    #                                 event_date, netatmo_stn_to_test]
                            try:
                                obs_netatmo_ppt_stn = netatmo_ppt_vals_fr_dwd_interp[
                                    ix_stn]
                            except Exception as msg:
                                print(msg)

                            ordinary_kriging_dwd_ppt_at_netatmo = OrdinaryKriging(
                                xi=dwd_xcoords,
                                yi=dwd_ycoords,
                                zi=ppt_dwd_vals,
                                xk=x_netatmo_stn,
                                yk=y_netatmo_stn,
                                model=vgs_model_dwd_ppt)

                            ordinary_kriging_dwd_ppt_at_netatmo.krige()
                            int_netatmo_ppt = ordinary_kriging_dwd_ppt_at_netatmo.zk.copy()[
                                0]

                            ratio_netatmo = np.abs(
                                obs_netatmo_ppt_stn / int_netatmo_ppt)

                            if min_ratio <= ratio_netatmo <= max_ratio:
                                pass

                            else:

                                idx_stn_remove = np.where(np.logical_and(
                                    (x_netatmo_ppt_vals_fr_dwd_interp ==
                                     x_netatmo_stn), (
                                        y_netatmo_ppt_vals_fr_dwd_interp ==
                                         y_netatmo_stn)))[0][0]
                                idxs_stns_remove.append(idx_stn_remove)
                                stns_filtered = True

                        netatmo_ppt_vals_fr_dwd_interp_gd = [
                            ppt for ix, ppt in enumerate(
                                netatmo_ppt_vals_fr_dwd_interp)
                            if ix not in idxs_stns_remove]

                        x_netatmo_ppt_vals_fr_dwd_interp_gd = [
                            x for ix, x in enumerate(
                                x_netatmo_ppt_vals_fr_dwd_interp)
                            if ix not in idxs_stns_remove]

                        y_netatmo_ppt_vals_fr_dwd_interp_gd = [
                            y for ix, y in enumerate(
                                y_netatmo_ppt_vals_fr_dwd_interp)
                            if ix not in idxs_stns_remove]

                        netatmo_stns_event_gd = [
                            stn for ix, stn in enumerate(
                                netatmo_stns_event_)
                            if ix not in idxs_stns_remove]

                    #======================================================
                    # Transform everything to arrays and combine
                    # dwd-netatmo

                    netatmo_xcoords = np.array(
                        x_netatmo_ppt_vals_fr_dwd_interp).ravel()
                    netatmo_ycoords = np.array(
                        y_netatmo_ppt_vals_fr_dwd_interp).ravel()

                    ppt_netatmo_vals = np.round(np.array(
                        netatmo_ppt_vals_fr_dwd_interp).ravel(), 2)

                    netatmo_dwd_x_coords = np.concatenate([netatmo_xcoords,
                                                           dwd_xcoords])
                    netatmo_dwd_y_coords = np.concatenate([netatmo_ycoords,
                                                           dwd_ycoords])
                    netatmo_dwd_ppt_vals = np.round(np.hstack(
                        (ppt_netatmo_vals,
                         ppt_dwd_vals)), 2).ravel()

                    #======================================================
                    # # Krigging PPT
                    #======================================================
                    print('Krigging PPT')
                    # using Netatmo-DWD data
                    ordinary_kriging_dwd_netatmo_ppt = OrdinaryKriging(
                        xi=netatmo_dwd_x_coords,
                        yi=netatmo_dwd_y_coords,
                        zi=netatmo_dwd_ppt_vals,
                        xk=x_coords_grd,
                        yk=y_coords_grd,
                        model=vgs_model_dwd_ppt)

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
                        xi=netatmo_xcoords,
                        yi=netatmo_ycoords,
                        zi=ppt_netatmo_vals,
                        xk=x_coords_grd,
                        yk=y_coords_grd,
                        model=vgs_model_dwd_ppt)

    #                     try:
    #                     print('\nOK using DWD-Netatmo')
                    ordinary_kriging_dwd_netatmo_ppt.krige()

    #                     print('\nOK using DWD')
                    ordinary_kriging_dwd_ppt.krige()

    #                     print('\nOK using Netatmo')
                    ordinary_kriging_netatmo_ppt.krige()
#                         except Exception as msg:
#                             print('Error while Kriging', msg)

                    interpolated_vals_dwd_netatmo = ordinary_kriging_dwd_netatmo_ppt.zk.copy()
                    interpolated_vals_dwd_only = ordinary_kriging_dwd_ppt.zk.copy()
                    interpolated_vals_netatmo_only = ordinary_kriging_netatmo_ppt.zk.copy()

                    # put negative values to 0
                    interpolated_vals_dwd_netatmo[
                        interpolated_vals_dwd_netatmo < 0] = 0

                    interpolated_vals_dwd_only[
                        interpolated_vals_dwd_only < 0] = 0

                    interpolated_vals_netatmo_only[
                        interpolated_vals_netatmo_only < 0] = 0

                    # difference netatmo-dwd - dwd
                    diff_map_plus = interpolated_vals_dwd_only - interpolated_vals_dwd_netatmo

                    diff_map_plus2 = interpolated_vals_dwd_only - interpolated_vals_netatmo_only
                    print('Plotting')
                    plt.ioff()

                    plt.ioff()
                    bound = [0., 1,
                             2, 5, 8,
                             10, 15, 20,
                             25, 30]  # , 35, 40, 45]

                    cmap = plt.get_cmap('jet')
                    norm = mcolors.BoundaryNorm(bound, cmap.N)
                    # interpolated_vals_dwd_netatmo
                    plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_dwd_netatmo,
                                         str_title=' DWD-Netatmo',
                                         out_plot_path=out_plots_path,
                                         temp_agg=temp_agg,
                                         event_date=event_date)

                    # interpolated_vals_dwd_only
                    plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_dwd_only,
                                         str_title=' DWD',
                                         out_plot_path=out_plots_path,
                                         temp_agg=temp_agg,
                                         event_date=event_date)

                    # interpolated_vals_netatmo
                    plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_netatmo_only,
                                         str_title=' Netatmo',
                                         out_plot_path=out_plots_path,
                                         temp_agg=temp_agg,
                                         event_date=event_date)
                    bound = [-30, -15., -10,
                             -5, -2, 0, 2,
                             5, 10, 15,
                             30]

                    cmap = plt.get_cmap('jet')
                    norm = mcolors.BoundaryNorm(bound, cmap.N)

                    plot_interp_ppt_evnt(vals_to_plot=diff_map_plus,
                                         str_title=' DWD minus DWD-Netatmo',
                                         out_plot_path=out_plots_path,
                                         temp_agg=temp_agg,
                                         event_date=event_date)

                    plot_interp_ppt_evnt(vals_to_plot=diff_map_plus2,
                                         str_title=' DWD minus Netatmo',
                                         out_plot_path=out_plots_path,
                                         temp_agg=temp_agg,
                                         event_date=event_date)

stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))

#
#                 #==========================================================
#                 # USE FIRST NETATMO FILTER
#                 #==========================================================
#                 if (use_netatmo_gd_stns and
#                         not use_temporal_filter_after_kriging):
#
#                     print('NETATMO 1st FILTERED')
#                     netatmo_stns_event_ = []
#                     netatmo_ppt_vals_fr_dwd_interp = []
# #                     netatmo_ppt_vals_fr_dwd_interp_unc = []
#
#                     x_netatmo_ppt_vals_fr_dwd_interp = []
#                     y_netatmo_ppt_vals_fr_dwd_interp = []
#                     # netatmo stations to correct
#
#                     for netatmo_stn_id in netatmo_df.index:
#
#                         netatmo_edf_event_ = netatmo_in_vals_df.loc[
#                             event_date,
#                             netatmo_stn_id]
#
#                         netatmo_ppt_event_ = netatmo_in_ppt_vals_df.loc[
#                             event_date,
#                             netatmo_stn_id]
#
#                         x_netatmo_interpolate = np.array(
#                             [netatmo_in_coords_df.loc[netatmo_stn_id, 'X']])
#                         y_netatmo_interpolate = np.array(
#                             [netatmo_in_coords_df.loc[netatmo_stn_id, 'Y']])
#
#                         if netatmo_edf_event_ > 0.7:  # 0.99:
#                             print('Correcting Netatmo station',
#                                   netatmo_stn_id)
#                             try:
#                                 #==========================================
#                                 # # Correct Netatmo Quantiles
#                                 #==========================================
#
#                                 netatmo_stn_edf_df = netatmo_in_vals_df.loc[
#                                     :, netatmo_stn_id].dropna()
#
#                                 netatmo_start_date = netatmo_stn_edf_df.index[0]
#                                 netatmo_end_date = netatmo_stn_edf_df.index[-1]
#
#                                 print('\nOriginal Netatmo Ppt: ',
#                                       netatmo_ppt_event_,
#                                       '\nOriginal Netatmo Edf: ',
#                                       netatmo_edf_event_)
#
#                                 # select dwd stations with only same period as
#                                 # netatmo stn
#                                 ppt_dwd_stn_vals = dwd_in_ppt_vals_df.loc[
#                                     netatmo_start_date:netatmo_end_date,
#                                     :].dropna(how='all')
#
#                                 dwd_xcoords_new = []
#                                 dwd_ycoords_new = []
#                                 ppt_vals_dwd_new_for_obsv_ppt = []
#
#                                 for dwd_stn_id_new in ppt_dwd_stn_vals.columns:
#
#                                     ppt_vals_stn_new = ppt_dwd_stn_vals.loc[
#                                         :, dwd_stn_id_new].dropna()
#
#                                     if ppt_vals_stn_new.size > 0:
#
#                                         ppt_stn_new, edf_stn_new = build_edf_fr_vals(
#                                             ppt_vals_stn_new)
#
#                                         if netatmo_edf_event_ == 1.0:
#                                             # correct edf event, 1 is bcz
#                                             # rounding error
#                                             ppt_ix_edf = find_nearest(
#                                                 ppt_stn_new,
#                                                 _ppt_event_)
#
#                                             netatmo_edf_event_ = edf_stn_new[
#                                                 np.where(
#                                                     ppt_stn_new == ppt_ix_edf)][0]
#
# #                                                 netatmo_edf_event_unc = netatmo_edf_event_ + (
# #                                                     (1 - netatmo_edf_event_) * 0.1)
#                                             print('\nChanged Original Netatmo Edf: ',
#                                                   netatmo_edf_event_)
#
#                                         nearst_edf_new = find_nearest(
#                                             edf_stn_new,
#                                             netatmo_edf_event_)
#
#                                         ppt_idx = np.where(
#                                             edf_stn_new == nearst_edf_new)
#
#                                         try:
#                                             if ppt_idx[0].size > 1:
#                                                 ppt_for_edf = np.mean(
#                                                     ppt_stn_new[ppt_idx])
#                                             else:
#                                                 ppt_for_edf = ppt_stn_new[ppt_idx]
#                                         except Exception as msg:
#                                             print(msg)
#
#                                         if ppt_for_edf[0] >= 0:
#
#                                             ppt_vals_dwd_new_for_obsv_ppt.append(
#                                                 ppt_for_edf[0])
#
#                                             stn_xcoords_new = dwd_in_coords_df.loc[
#                                                 dwd_stn_id_new, 'X']
#                                             stn_ycoords_new = dwd_in_coords_df.loc[
#                                                 dwd_stn_id_new, 'Y']
#
#                                             dwd_xcoords_new.append(
#                                                 stn_xcoords_new)
#                                             dwd_ycoords_new.append(
#                                                 stn_xcoords_new)
#
#                                 # get ppt from dwd recent for netatmo stns
#                                 ppt_vals_dwd_new_for_obsv_ppt = np.array(
#                                     ppt_vals_dwd_new_for_obsv_ppt)
#                                 # for kriging with uncertainty
#
#                                 dwd_xcoords_new = np.array(dwd_xcoords_new)
#                                 dwd_ycoords_new = np.array(dwd_ycoords_new)
#
#                                 try:
#                                     print(
#                                         '\n+++ KRIGING CORRECTING QT +++\n')
#
#                                     ordinary_kriging_dwd_netatmo_crt = OrdinaryKriging(
#                                         xi=dwd_xcoords_new,
#                                         yi=dwd_ycoords_new,
#                                         zi=ppt_vals_dwd_new_for_obsv_ppt,
#                                         xk=x_netatmo_interpolate,
#                                         yk=y_netatmo_interpolate,
#                                         model=vgs_model_dwd_ppt)
#
#                                     try:
#                                         ordinary_kriging_dwd_netatmo_crt.krige()
#                                     except Exception as msg:
#                                         print('Error while Kriging', msg)
#
#                                     interpolated_netatmo_prct = ordinary_kriging_dwd_netatmo_crt.zk.copy()
#
#                                     if interpolated_netatmo_prct < 0:
#                                         interpolated_netatmo_prct = np.nan
#
#                                     print('**Interpolated PPT by DWD recent: \n',
#                                           interpolated_netatmo_prct)
#
#                                     if interpolated_netatmo_prct >= 0.:
#                                         netatmo_ppt_vals_fr_dwd_interp.append(
#                                             interpolated_netatmo_prct[0])
#
#                                         x_netatmo_ppt_vals_fr_dwd_interp.append(
#                                             x_netatmo_interpolate[0])
#
#                                         y_netatmo_ppt_vals_fr_dwd_interp.append(
#                                             y_netatmo_interpolate[0])
#
#                                         netatmo_stns_event_.append(
#                                             netatmo_stn_id)
#                                 except Exception as msg:
#                                     print(
#                                         msg, 'Error when getting ppt from dwd interp')
#                                     continue
#
#                             except Exception as msg:
#                                 print(msg, 'Error when KRIGING')
#                                 continue
#
#                         else:
#                             print('QT to small no need to correct it\n')
#                             # check if nan, if so don't add it
#                             if netatmo_ppt_event_ >= 0:
#                                 netatmo_ppt_vals_fr_dwd_interp.append(
#                                     netatmo_ppt_event_)
#
# #                                     netatmo_ppt_vals_fr_dwd_interp_unc.append(
# #                                         netatmo_ppt_event_)
#                                 x_netatmo_ppt_vals_fr_dwd_interp.append(
#                                     x_netatmo_interpolate[0])
#
#                                 y_netatmo_ppt_vals_fr_dwd_interp.append(
#                                     y_netatmo_interpolate[0])
#
#                                 netatmo_stns_event_.append(netatmo_stn_id)
#
#                     #======================================================
#                     # Transform everything to arrays and combine
#                     # dwd-netatmo
#
#                     netatmo_xcoords = np.array(
#                         x_netatmo_ppt_vals_fr_dwd_interp).ravel()
#                     netatmo_ycoords = np.array(
#                         y_netatmo_ppt_vals_fr_dwd_interp).ravel()
#
#                     ppt_netatmo_vals = np.round(np.array(
#                         netatmo_ppt_vals_fr_dwd_interp).ravel(), 2)
#
#                     netatmo_dwd_x_coords = np.concatenate([netatmo_xcoords,
#                                                            dwd_xcoords])
#                     netatmo_dwd_y_coords = np.concatenate([netatmo_ycoords,
#                                                            dwd_ycoords])
#                     netatmo_dwd_ppt_vals = np.round(np.hstack(
#                         (ppt_netatmo_vals,
#                          ppt_dwd_vals)), 2).ravel()
#
#                     #======================================================
#                     # # Krigging PPT
#                     #======================================================
#                     # using Netatmo-DWD data
#                     ordinary_kriging_dwd_netatmo_ppt = OrdinaryKriging(
#                         xi=netatmo_dwd_x_coords,
#                         yi=netatmo_dwd_y_coords,
#                         zi=netatmo_dwd_ppt_vals,
#                         xk=x_coords_grd,
#                         yk=y_coords_grd,
#                         model=vgs_model_dwd_ppt)
#                     # using DWD data
#                     ordinary_kriging_dwd_ppt = OrdinaryKriging(
#                         xi=dwd_xcoords,
#                         yi=dwd_ycoords,
#                         zi=ppt_dwd_vals,
#                         xk=x_coords_grd,
#                         yk=y_coords_grd,
#                         model=vgs_model_dwd_ppt)
#                     # using Netatmo data
#                     ordinary_kriging_netatmo_ppt = OrdinaryKriging(
#                         xi=netatmo_xcoords,
#                         yi=netatmo_ycoords,
#                         zi=ppt_netatmo_vals,
#                         xk=x_coords_grd,
#                         yk=y_coords_grd,
#                         model=vgs_model_dwd_ppt)
#
#     #                     try:
#                     print('\nOK using DWD-Netatmo')
#                     ordinary_kriging_dwd_netatmo_ppt.krige()
#
#                     print('\nOK using DWD')
#                     ordinary_kriging_dwd_ppt.krige()
#
#                     print('\nOK using Netatmo')
#                     ordinary_kriging_netatmo_ppt.krige()
#     #                     except Exception as msg:
#     #                         print('Error while Kriging', msg)
#
#                     interpolated_vals_dwd_netatmo = ordinary_kriging_dwd_netatmo_ppt.zk.copy()
#                     interpolated_vals_dwd_only = ordinary_kriging_dwd_ppt.zk.copy()
#                     interpolated_vals_netatmo_only = ordinary_kriging_netatmo_ppt.zk.copy()
#
#                     # put negative values to 0
#                     interpolated_vals_dwd_netatmo[
#                         interpolated_vals_dwd_netatmo < 0] = 0
#
#                     interpolated_vals_dwd_only[
#                         interpolated_vals_dwd_only < 0] = 0
#
#                     interpolated_vals_netatmo_only[
#                         interpolated_vals_netatmo_only < 0] = 0
#
#                     plt.ioff()
#                     # interpolated_vals_dwd_netatmo
#                     plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_dwd_netatmo,
#                                          str_title=' DWD-Netatmo',
#                                          out_plot_path=out_plots_path,
#                                          temp_agg=temp_agg,
#                                          event_date=event_date)
#
#                     # interpolated_vals_dwd_only
#                     plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_dwd_only,
#                                          str_title=' DWD',
#                                          out_plot_path=out_plots_path,
#                                          temp_agg=temp_agg,
#                                          event_date=event_date)
#
#                     # interpolated_vals_netatmo
#                     plot_interp_ppt_evnt(vals_to_plot=interpolated_vals_netatmo_only,
#                                          str_title=' Netatmo',
#                                          out_plot_path=out_plots_path,
#                                          temp_agg=temp_agg,
#                                          event_date=event_date)
