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
import pprint
#import math
import pandas as pd
import matplotlib.pyplot as plt


from spinterps import (OrdinaryKriging, OrdinaryKrigingWithUncertainty)
from spinterps import variograms

#from scipy.spatial import distance_matrix
from scipy.spatial import distance
from scipy import spatial

from pathlib import Path
from random import shuffle

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

#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================

use_dwd_stns_for_kriging = True

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


#==============================================================================
#
#==============================================================================
resample_frequencies = ['1440min']
# '120min', '180min', '60min',  '360min',
#                         '720min',
title_ = r'Ppt_ok_ok_un_2'


if not use_netatmo_gd_stns:
    title_ = title_ + '_netatmo_no_flt_'

if use_netatmo_gd_stns:
    title_ = title_ + '_first_flt_'

if use_temporal_filter_after_kriging:

    title_ = title_ + '_temp_flt_'

# def out plot path based on combination

plot_2nd_filter_netatmo = False
#==============================================================================
#
#==============================================================================

plot_events = False

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
# SELECT GROUP OF 10 DWD STATIONS RANDOMLY
#==============================================================================


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
#     shuffle(all_dwd_stns)
#     shuffled_dwd_stns_10stn = np.array(list(chunks(all_dwd_stns, 10)))

    #==========================================================================
    # # CREATE DFS FOR RESULT; Index is Date, Columns as Stns
    #==========================================================================
    df_interpolated_dwd_netatmos_comb = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)

    df_interpolated_dwd_netatmos_comb_un = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)

    df_interpolated_dwd_only = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)
    # get ratio between interpolated and obsv at DWD for 2nd filter
    df_interpolated_ratios = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)

#     df_interpolated_dwd_netatmos_comb_corr = pd.DataFrame(
#         index=dwd_in_extremes_df.index,
#         columns=all_dwd_stns)
#
#     df_interpolated_dwd_netatmos_comb_un_corr = pd.DataFrame(
#         index=dwd_in_extremes_df.index,
#         columns=all_dwd_stns)

    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    for event_date in dwd_in_extremes_df.index:

        _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
        if len(_stn_id_event_) < 5:
            _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                '0' + _stn_id_event_

        _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
        _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]
        print('**Calculating for Date ',
              event_date, '\n Rainfall: ',  _ppt_event_,
              'Quantile: ', _edf_event_, ' **\n')

        # start cross validating DWD stations for this event
#         for idx_lst_comb in range(len(shuffled_dwd_stns_10stn)):
#             stn_comb = shuffled_dwd_stns_10stn[idx_lst_comb]
        for idx_lst_comb in df_dwd_stns_comb.index:

            stn_comb = [stn.replace("'", "")
                        for stn in df_dwd_stns_comb.iloc[
                            idx_lst_comb, :].dropna().values]
            print('Interpolating for following DWD stations: \n',
                  pprint.pformat(stn_comb))

            # =================================================================
            # START KRIGING THIS GROUP OF STATIONS
            # =================================================================

            for stn_dwd_id in stn_comb:

                print('interpolating for DWD Station', stn_dwd_id)
                obs_ppt_stn_dwd = dwd_in_ppt_vals_df.loc[event_date, stn_dwd_id]
                x_dwd_interpolate = np.array(
                    [dwd_in_coords_df.loc[stn_dwd_id, 'X']])
                y_dwd_interpolate = np.array(
                    [dwd_in_coords_df.loc[stn_dwd_id, 'Y']])

                # drop stns
                all_dwd_stns_except_interp_loc = [
                    stn for stn in dwd_in_vals_df.columns
                    if stn not in stn_comb]

                # GET ALL NETATMO NEARBY STATIONS
                # find distance to all dwd stations, sort them, select minimum
                distances_dwd_to_stns = in_df_distance_netatmo_dwd.loc[
                    :, stn_dwd_id]

                # ppt data at other DWD stations
                ppt_dwd_vals_sr = dwd_in_ppt_vals_df.loc[
                    event_date,
                    all_dwd_stns_except_interp_loc]
                ppt_dwd_vals_nona = ppt_dwd_vals_sr[
                    ppt_dwd_vals_sr.values >= 0]

                # coords of all other stns
                x_dwd_all = dwd_in_coords_df.loc[
                    ppt_dwd_vals_nona.index, 'X'].values
                y_dwd_all = dwd_in_coords_df.loc[
                    ppt_dwd_vals_nona.index, 'Y'].values

                # select only nearby netatmo stations below 50km
                sorted_distances_ppt_dwd = distances_dwd_to_stns.sort_values(
                    ascending=True)

                sorted_distances_ppt_dwd = sorted_distances_ppt_dwd[
                    sorted_distances_ppt_dwd.values <= 10e4]
                netatmo_stns_near = sorted_distances_ppt_dwd.index

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

                    print('\n+++ KRIGING PPT at DWD +++\n')

                    # netatmo stns for this event
                    # Netatmo data and coords
                    netatmo_df = netatmo_in_ppt_vals_df.loc[
                        event_date,
                        netatmo_stns_near].dropna(
                        how='all')

                    if netatmo_df.size > 0:

                        # list to hold result of all netatmo station for this
                        # event
                        netatmo_ppt_vals_fr_dwd_interp = netatmo_df.values
                        # TODO ASK AGAIN PROF.
                        netatmo_ppt_vals_fr_dwd_interp_unc = (netatmo_ppt_vals_fr_dwd_interp +
                                                              netatmo_ppt_vals_fr_dwd_interp * 0.1)

                        x_netatmo_ppt_vals_fr_dwd_interp = netatmo_in_coords_df.loc[
                            netatmo_df.index, 'X'].values
                        y_netatmo_ppt_vals_fr_dwd_interp = netatmo_in_coords_df.loc[
                            netatmo_df.index, 'Y'].values

                        # Krigging PPT at DWD station
                        #======================================================
                        # Transform everything to arrays and combine
                        # dwd-netatmo

                        netatmo_xcoords = np.array(
                            x_netatmo_ppt_vals_fr_dwd_interp).ravel()
                        netatmo_ycoords = np.array(
                            y_netatmo_ppt_vals_fr_dwd_interp).ravel()
#                         plt.ioff()
#                         plt.scatter(netatmo_xcoords, netatmo_ycoords)
#                         plt.scatter(x_dwd_all, y_dwd_all)
#                         plt.scatter(x_dwd_interpolate, y_dwd_interpolate)
#                         plt.show()
                        ppt_netatmo_vals = np.round(np.array(
                            netatmo_ppt_vals_fr_dwd_interp).ravel(), 2)

                        ppt_netatmo_vals_unc = np.round(np.array(
                            netatmo_ppt_vals_fr_dwd_interp_unc).ravel(), 2)

                        netatmo_dwd_x_coords = np.concatenate([netatmo_xcoords,
                                                               x_dwd_all])
                        netatmo_dwd_y_coords = np.concatenate([netatmo_ycoords,
                                                               y_dwd_all])
                        netatmo_dwd_ppt_vals = np.round(np.hstack(
                            (ppt_netatmo_vals,
                             ppt_dwd_vals_nona.values)), 2).ravel()

                        netatmo_dwd_ppt_vals_unc = np.round(np.concatenate(
                            [ppt_netatmo_vals_unc,
                             ppt_dwd_vals_nona.values]), 2).ravel()

                        if (isinstance(vgs_model_dwd_ppt, str)) and (
                            'Nug' in vgs_model_dwd_ppt
                            or len(vgs_model_dwd_ppt) > 0) and (
                            'Exp' in vgs_model_dwd_ppt or
                                'Sph' in vgs_model_dwd_ppt):

                            print('Krigging PPT at DWD Stns')

                            #==================================================
                            # Start kriging ppt at DWD
                            #==================================================

                            ordinary_kriging_dwd_netatmo_ppt = OrdinaryKriging(
                                xi=netatmo_dwd_x_coords,
                                yi=netatmo_dwd_y_coords,
                                zi=netatmo_dwd_ppt_vals,
                                xk=x_dwd_interpolate,
                                yk=y_dwd_interpolate,
                                model=vgs_model_dwd_ppt)

                            ordinary_kriging_dwd_ppt = OrdinaryKriging(
                                xi=x_dwd_all,
                                yi=y_dwd_all,
                                zi=ppt_dwd_vals_nona.values,
                                xk=x_dwd_interpolate,
                                yk=y_dwd_interpolate,
                                model=vgs_model_dwd_ppt)

                            # kriging with uncertainty
                            ordinary_kriging_dwd_netatmo_ppt_unc = OrdinaryKriging(
                                xi=netatmo_dwd_x_coords,
                                yi=netatmo_dwd_y_coords,
                                zi=netatmo_dwd_ppt_vals_unc,
                                xk=x_dwd_interpolate,
                                yk=y_dwd_interpolate,
                                model=vgs_model_dwd_ppt)

                            try:
                                ordinary_kriging_dwd_netatmo_ppt.krige()
                                ordinary_kriging_dwd_ppt.krige()
                                ordinary_kriging_dwd_netatmo_ppt_unc.krige()
                            except Exception as msg:
                                print('Error while Kriging', msg)

                            interpolated_netatmo_dwd_ppt = ordinary_kriging_dwd_netatmo_ppt.zk.copy()[
                                0]
                            interpolated_dwd_ppt = ordinary_kriging_dwd_ppt.zk.copy()[
                                0]
                            interpolated_netatmo_dwd_ppt_unc = ordinary_kriging_dwd_netatmo_ppt_unc.zk.copy()[
                                0]

                            if interpolated_netatmo_dwd_ppt < 0:
                                interpolated_netatmo_dwd_ppt = np.nan

                            if interpolated_dwd_ppt < 0:
                                interpolated_dwd_ppt = np.nan

                            if interpolated_netatmo_dwd_ppt_unc < 0:
                                interpolated_netatmo_dwd_ppt_unc = np.nan

                            print('**Interpolated PPT by DWD_Netatmo: ',
                                  interpolated_netatmo_dwd_ppt)

                            print('**Interpolated PPT by DWD Only: ',
                                  interpolated_dwd_ppt)

                            print('**Interpolated PPT by DWD-Netatmo Uncert: ',
                                  interpolated_netatmo_dwd_ppt_unc)
                        else:
                            print('no good VG found,\n adding nans to df')
                            interpolated_netatmo_dwd_ppt = np.nan
                            interpolated_dwd_ppt = np.nan
                            interpolated_netatmo_dwd_ppt_unc = np.nan

                        print('+++ Saving result to DF +++\n')

                        df_interpolated_dwd_netatmos_comb.loc[
                            event_date,
                            stn_dwd_id] = interpolated_netatmo_dwd_ppt

                        df_interpolated_dwd_netatmos_comb_un.loc[
                            event_date,
                            stn_dwd_id] = interpolated_netatmo_dwd_ppt_unc

                        df_interpolated_dwd_only.loc[
                            event_date,
                            stn_dwd_id] = interpolated_dwd_ppt
                print('Done with this DWD Stn\n')
            print('Done with this group of DWD Stns\n')
        print('Done with this event\n')

    df_interpolated_dwd_netatmos_comb.dropna(how='all', inplace=True)
    df_interpolated_dwd_netatmos_comb_un.dropna(how='all', inplace=True)
    df_interpolated_dwd_only.dropna(how='all', inplace=True)
# #     df_interpolated_netatmo_only.dropna(how='all', inplace=True)
# #     df_interpolated_netatmo_only_un.dropna(how='all', inplace=True)
#
    df_interpolated_dwd_netatmos_comb.to_csv(out_plots_path / (
        'interpolated_ppt_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
        % (temp_agg, title_, idx_lst_comb, _acc_)),
        sep=';', float_format='%0.2f')
#
    df_interpolated_dwd_netatmos_comb_un.to_csv(out_plots_path / (
        'interpolated_ppt_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s_unc.csv'
        % (temp_agg, title_, idx_lst_comb, _acc_)),
        sep=';', float_format='%0.2f')

    df_interpolated_dwd_only.to_csv(out_plots_path / (
        'interpolated_ppt_dwd_%s_data_%s_using_dwd_only_grp_%d_%s.csv'
        % (temp_agg, title_, idx_lst_comb, _acc_)),
        sep=';', float_format='%0.2f')

#     df_interpolated_netatmo_only.to_csv(out_plots_path / (
#         'interpolated_quantiles_dwd_%s_data_%s_using_netamo_only_grp_%d_%s.csv'
#         % (temp_agg, title_, idx_lst_comb, _acc_)),
#         sep=';', float_format='%0.2f')
#
#     df_interpolated_netatmo_only_un.to_csv(out_plots_path / (
#         'interpolated_quantiles_un_dwd_%s_data_%s_using_netamo_only_grp_%d_%s.csv'
#         % (temp_agg, title_, idx_lst_comb, _acc_)),
#         sep=';', float_format='%0.2f')


stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
