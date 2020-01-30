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


from spinterps import (OrdinaryKriging)
from spinterps import variograms
from scipy.spatial import distance
from scipy import spatial

from pathlib import Path


VG = variograms.vgs.Variogram


plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'axes.labelsize': 18})


# =============================================================================

# =============================================================================
#
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

#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================


# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging

use_temporal_filter_b4_kriging = True  # on day filter on percentiles

use_first_neghbr_as_gd_stns = True  # False
use_first_and_second_nghbr_as_gd_stns = False  # True

plot_2nd_filter_netatmo = True  # for plotting on event filter percentiles

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
resample_frequencies = ['60min']
# '120min', '180min', '60min',  '360min',
#                         '720min',
title_ = r'Ppt_ok_ok_un_ratios'


if not use_netatmo_gd_stns:
    title_ = title_ + '_netatmo_no_flt_'

if use_netatmo_gd_stns:
    title_ = title_ + '_first_flt_'

if use_temporal_filter_b4_kriging:

    title_ = title_ + '_temp_flt_'

# def out plot path based on combination


strt_date = '2015-01-01 00:00:00'
end_date = '2019-09-01 00:00:00'

daily_evts = pd.DatetimeIndex(['2018-05-14', '2019-07-28'])

hourly_evts = pd.DatetimeIndex(['2016-06-25 00:00:00', '2018-06-11 16:00:00',
                                '2018-09-06 18:00:00'])
# min_valid_stns = 20

drop_stns = []
mdr = 0.9
perm_r_list_ = [1, 2]
fit_vgs = ['Sph', 'Exp']  # 'Sph',
fil_nug_vg = 'Nug'  # 'Nug'
n_best = 4
ngp = 5


idx_time_fmt = '%Y-%m-%d %H:%M:%S'

m_size = 20
radius = 10000
diff_thr = 0.1
edf_thr = 0.7
#==============================================================================
#
#==============================================================================


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
#
#==============================================================================
# SELECT GROUP OF 10 DWD STATIONS RANDOMLY
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# read distance matrix dwd-netamot ppt
in_df_distance_netatmo_dwd = pd.read_csv(
    distance_matrix_netatmo_dwd_df_file, sep=';', index_col=0)

# read df combinations to use
df_dwd_stns_comb = pd.read_csv(
    path_to_dwd_stns_comb, index_col=0,
    sep=',', dtype=str)

#==============================================================================
# make out dir
#==============================================================================

#==============================================================================

for temp_agg in resample_frequencies:

    # out path directory
    dir_path = title_ + '_' + _acc_ + '_' + temp_agg

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
            df_vgs_extremes.index).intersection(
            dwd_ratios_df.index), :]

    cmn_netatmo_stns = in_df_distance_netatmo_dwd.index.intersection(
        netatmo_in_vals_df.columns)
    in_df_distance_netatmo_dwd = in_df_distance_netatmo_dwd.loc[
        cmn_netatmo_stns, :]

    print('\n%d Extreme Event to interpolate\n' % dwd_in_extremes_df.shape[0])
    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()
#     shuffle(all_dwd_stns)
#     shuffled_dwd_stns_10stn = np.array(list(chunks(all_dwd_stns, 10)))
    stn_comb_order = []
    for idx_lst_comb in df_dwd_stns_comb.index:

        stn_comb_order.append([stn.replace("'", "")
                               for stn in df_dwd_stns_comb.iloc[
            idx_lst_comb, :].dropna().values])
    stn_comb_order_only = [stn for stn_co in stn_comb_order
                           for stn in stn_co]
    #==========================================================================
    # # CREATE DFS FOR RESULT; Index is Date, Columns as Stns
    #==========================================================================

    df_interpolated_dwd_netatmos_comb = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=stn_comb_order_only)

    df_interpolated_dwd_netatmos_comb_un_20perc = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=stn_comb_order_only)
    df_interpolated_dwd_netatmos_comb_un_10perc = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=stn_comb_order_only)

    df_interpolated_dwd_netatmos_comb_un_5perc = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=stn_comb_order_only)

    df_interpolated_dwd_netatmos_comb_un_2perc = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=stn_comb_order_only)

    df_interpolated_dwd_only = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=stn_comb_order_only)

    df_stns_netatmo_gd_event = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=stn_comb_order_only)
    # hourly_evts
#     dwd_in_extremes_df = dwd_in_extremes_df.loc[daily_evts, :]  # daily_evts
    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    for iev, event_date in enumerate(dwd_in_extremes_df.index):

        _stn_id_event_ = str(int(dwd_in_extremes_df.loc[event_date, 2]))
        if len(_stn_id_event_) < 5:
            _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                '0' + _stn_id_event_

        _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
        _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]
#         print('**Calculating for Date ',
#               event_date, '\n Rainfall: ',  _ppt_event_,
#               'Quantile: ', _edf_event_, ' **\n')

        # start cross validating DWD stations for this event
        for idx_lst_comb in df_dwd_stns_comb.index:

            stn_comb = [stn.replace("'", "")
                        for stn in df_dwd_stns_comb.iloc[
                        idx_lst_comb, :].dropna().values]

#             print('Interpolating for following DWD stations: \n',
#                   pprint.pformat(stn_comb))

            # ==========================================================
            # START KRIGING THIS GROUP OF STATIONS
            # ==========================================================

            for stn_nbr, stn_dwd_id in enumerate(stn_comb):

                #                 print('interpolating for DWD Station', stn_dwd_id,
                #                       ' ', i, '/', len(stn_comb))
                obs_ppt_stn_dwd = dwd_in_ppt_vals_df.loc[event_date, stn_dwd_id]
                x_dwd_interpolate = np.array(
                    [dwd_in_coords_df.loc[stn_dwd_id, 'X']])
                y_dwd_interpolate = np.array(
                    [dwd_in_coords_df.loc[stn_dwd_id, 'Y']])

                # drop stns
                all_dwd_stns_except_interp_loc = [
                    stn for stn in dwd_in_vals_df.columns
                    if stn not in stn_comb]

                # GET ALL DWD NEARBY STATIONS

                # nearby_dwd_stns =
                x_coords_dwd_except_interp_loc = dwd_in_coords_df.loc[
                    all_dwd_stns_except_interp_loc, 'X'].values
                y_coords_dwd_except_interp_loc = dwd_in_coords_df.loc[
                    all_dwd_stns_except_interp_loc, 'Y'].values
                # make dwd coords a tuple
                xy_coords_except_interp_loc = np.array(
                    [(x, y) for x, y in zip(x_coords_dwd_except_interp_loc,
                                            y_coords_dwd_except_interp_loc)])
                # create a tree from coordinates
                dwd_points_tree = spatial.KDTree(
                    xy_coords_except_interp_loc)

                # This finds the index of all points within
                # distance 1 of [1.5,2.5].
                idxs_dwd_neighbours = dwd_points_tree.query_ball_point(
                    np.array((x_dwd_interpolate[0],
                              y_dwd_interpolate[0])), 3e4)
                # get nearby dwd stns
                dwd_stns_id_nearby = dwd_in_coords_df.iloc[
                    idxs_dwd_neighbours, :].index.to_list()

                # GET ALL NETATMO NEARBY STATIONS
                # find distance to all dwd stations, sort them, select minimum
                distances_dwd_to_stns = in_df_distance_netatmo_dwd.loc[
                    :, stn_dwd_id]

                sorted_distances_ppt_dwd = distances_dwd_to_stns.sort_values(
                    ascending=True)

                # select only nearby netatmo stations below 50km
                sorted_distances_ppt_dwd = sorted_distances_ppt_dwd[
                    sorted_distances_ppt_dwd.values <= 3e4]
                netatmo_stns_near = sorted_distances_ppt_dwd.index

#                 plt.ioff()
#                 plt.scatter(dwd_stns_id_nearby.X, dwd_stns_id_nearby.Y, c='r')
#                 plt.scatter(x_dwd_interpolate, y_dwd_interpolate, c='b')
                # ppt data at other DWD stations
                ppt_dwd_vals_sr = dwd_in_ppt_vals_df.loc[
                    event_date,
                    dwd_stns_id_nearby]
                ppt_dwd_vals_nona = ppt_dwd_vals_sr[
                    ppt_dwd_vals_sr.values >= 0]

                # edf dwd vals
                edf_dwd_vals = dwd_in_vals_df.loc[
                    event_date,
                    dwd_stns_id_nearby].dropna().values

                # coords of all other stns
                x_dwd_all = dwd_in_coords_df.loc[
                    ppt_dwd_vals_nona.index, 'X'].values
                y_dwd_all = dwd_in_coords_df.loc[
                    ppt_dwd_vals_nona.index, 'Y'].values

                # netatmo ppt values and stns fot this event
                netatmo_ppt_stns_for_event = netatmo_in_ppt_vals_df.loc[
                    event_date, netatmo_stns_near].dropna(how='all')

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

                if (isinstance(vgs_model_dwd_ppt, str)) and (
                    'Nug' in vgs_model_dwd_ppt
                    or len(vgs_model_dwd_ppt) > 0) and (
                    'Exp' in vgs_model_dwd_ppt or
                        'Sph' in vgs_model_dwd_ppt):

                    # print('\n+++ KRIGING PPT at DWD +++\n')

                    # netatmo stns for this event
                    # Netatmo data and coords
                    netatmo_df = netatmo_in_vals_df.loc[
                        event_date,
                        sorted_distances_ppt_dwd.index].dropna(
                        how='all')

                    # netatmo coords

                    netatmo_xcoords = np.array(
                        [netatmo_in_coords_df.loc[
                            netatmo_df.index, 'X']]).ravel()
                    netatmo_ycoords = np.array(
                        [netatmo_in_coords_df.loc[
                            netatmo_df.index, 'Y']]).ravel()
                    edf_netatmo_vals = np.array(netatmo_df.values).ravel()

                    netatmo_stns_event_ = netatmo_df.index.to_list()

                    if netatmo_df.size > 0:

                        #==================================================
                        # filter percentiles
                        #==================================================

                        if use_temporal_filter_b4_kriging:

                            ordinary_kriging_filter_netamto = OrdinaryKriging(
                                xi=x_dwd_all,
                                yi=y_dwd_all,
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
                                netatmo_df.values - interpolated_vals)

                            #==================================================
                            # # use additional temporal filter
                            #==================================================
                            idx_good_stns = np.where(
                                diff_obsv_interp <= 3 * std_est_vals)
                            idx_bad_stns = np.where(
                                diff_obsv_interp > 3 * std_est_vals)

                            if len(idx_bad_stns[0]) or len(idx_good_stns[0]) > 0:
                                print('Number of Stations with bad index \n',
                                      len(idx_bad_stns[0]))
                                print('Number of Stations with good index \n',
                                      len(idx_good_stns[0]))

                                # use additional filter
                                try:
                                    ids_netatmo_stns_gd = np.take(netatmo_df.index,
                                                                  idx_good_stns).ravel()
                                    ids_netatmo_stns_bad = np.take(netatmo_df.index,
                                                                   idx_bad_stns).ravel()

                                except Exception as msg:
                                    print(msg)

                            try:
                                edf_gd_vals_df = netatmo_df.loc[ids_netatmo_stns_gd]
                            except Exception as msg:
                                print(msg, 'error while second filter')

#                             if len(idx_bad_stns[0]) > 2:
#                                 print('Dsd')
#                                 pass
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
                            edf_bad_vals_df = netatmo_df.loc[ids_netatmo_stns_bad]

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

                            x_coords_dwd_wet = x_dwd_all[np.where(
                                edf_dwd_vals >= edf_thr)]
                            y_coords_dwd_wet = y_dwd_all[np.where(
                                edf_dwd_vals >= edf_thr)]

                            assert (dwd_wet.size ==
                                    x_coords_dwd_wet.size ==
                                    y_coords_dwd_wet.size)

                            x_coords_dwd_dry = x_dwd_all[np.where(
                                edf_dwd_vals < edf_thr)]
                            y_coords_dwd_dry = y_dwd_all[np.where(
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
                                    print('Trying to correct %s bad wet '
                                          % stn_)
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
                                                print(
                                                    'bad wet netatmo station is good')
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
                                                print('bad wet is bad wet')
                                    else:
                                        print('\nStn has no near neighbors')

        #                         dwd_dry =

                            if plot_2nd_filter_netatmo:

                                plt.ioff()
        #                         texts = []
                                fig = plt.figure(figsize=(24, 12), dpi=300)
                                ax = fig.add_subplot(111)

                                ax.scatter(x_dwd_interpolate,
                                           y_dwd_interpolate, c='k',
                                           marker='2', s=m_size + 25,
                                           label='Interp Stn DWD %s' %
                                           stn_dwd_id)

                                ax.scatter(x_coords_gd_netatmo_dry,
                                           y_coords_gd_netatmo_dry, c='g',
                                           marker='d', s=m_size,
                                           label='%d Netatmo with dry good values' %
                                           x_coords_gd_netatmo_dry.shape[0])

                                ax.scatter(x_coords_gd_netatmo_wet,
                                           y_coords_gd_netatmo_wet, c='b',
                                           marker='d', s=m_size,
                                           label='%d Netatmo with wet good values' %
                                           x_coords_gd_netatmo_wet.shape[0])

                                ax.scatter(x_coords_bad_netatmo_dry,
                                           y_coords_bad_netatmo_dry, c='orange',
                                           marker='d', s=m_size,
                                           label='%d Netatmo with dry bad values' %
                                           x_coords_bad_netatmo_dry.shape[0])

                                ax.scatter(x_coords_bad_netatmo_wet,
                                           y_coords_bad_netatmo_wet, c='r',
                                           marker='d', s=m_size,
                                           label='%d Netatmo with wet bad values' %
                                           x_coords_bad_netatmo_wet.shape[0])

                                ax.scatter(x_coords_dwd_dry,
                                           y_coords_dwd_dry, c='g',
                                           marker='x', s=m_size,
                                           label='%d DWD with dry values' %
                                           x_coords_dwd_dry.shape[0])

                                ax.scatter(x_coords_dwd_wet,
                                           y_coords_dwd_wet, c='b',
                                           marker='x', s=m_size,
                                           label='%d DWD with wet values' %
                                           x_coords_dwd_wet.shape[0])
                                ax.grid(alpha=.25)
#                                 ax.grid(True)
                                ax.set_xticklabels([])
                                ax.set_yticklabels([])
                                # Shrink current axis by 20%
#                                 ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#                                           ncol=2, fancybox=True, shadow=True)

                                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                                          fancybox=True, shadow=True, ncol=4)
#                                 plt.legend(loc='upper right')


#                                 plt.xlabel('Longitude')
#                                 plt.ylabel('Latitude')
                                plt.axis('equal')

#                                 plt.show()

                                plt.savefig((out_plots_path / (
                                    'interp_stn_%s_%s_%s_event_%s_.png' %
                                    (stn_dwd_id, temp_agg,
                                     str(event_date).replace(
                                         '-', '_').replace(':',
                                                           '_').replace(' ', '_'),
                                     _acc_))),
                                    frameon=True, papertype='a4',
                                    bbox_inches='tight', pad_inches=.2)
                                plt.close(fig)

                            #==================================================
                            # Remaining Netatmo quantiles
                            #==================================================

                            edf_netatmo_vals_gd = edf_netatmo_vals[idx_good_stns[0]]
                            netatmo_xcoords_gd = netatmo_xcoords[idx_good_stns[0]]
                            netatmo_ycoords_gd = netatmo_ycoords[idx_good_stns[0]]

                            netatmo_stn_ids_gd = [netatmo_stns_event_[ix]
                                                  for ix in idx_good_stns[0]]

                            # keep only good stations
    #                         df_gd_stns_per_event = netatmo_df.loc[ids_netatmo_stns_gd]

                            print('\n----Keeping %d / %d Stns for event---\n'
                                  % (edf_netatmo_vals_gd.size,
                                     edf_netatmo_vals.size))

                            print('remaining: \n ', 100 * (
                                edf_netatmo_vals_gd.size /
                                edf_netatmo_vals.size))

                            df_stns_netatmo_gd_event.loc[
                                event_date,
                                stn_dwd_id] = 100 * (
                                edf_netatmo_vals_gd.size /
                                edf_netatmo_vals.size)

                break
            break
    df_stns_netatmo_gd_event.dropna(how='all', inplace=True)

    df_stns_netatmo_gd_event.to_csv(out_plots_path / (
        'ratio_netatmo_gd_bad_%s_data_%s_grp_%d_%s.csv'
        % (temp_agg, title_, idx_lst_comb, _acc_)),
        sep=';', float_format='%0.2f')

stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
