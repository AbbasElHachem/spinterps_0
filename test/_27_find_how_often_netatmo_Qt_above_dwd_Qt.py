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
import math
import pandas as pd
import matplotlib.pyplot as plt


from spinterps import (OrdinaryKriging, OrdinaryKrigingWithUncertainty)
from spinterps import variograms
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from scipy import spatial

from pathlib import Path
from random import shuffle

VG = variograms.vgs.Variogram


plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})


# =============================================================================
#main_dir = Path(r'/home/abbas/Documents/Python/Extremes')
# main_dir = Path(r'/home/IWS/hachem/Extremes')
main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
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
use_netatmo_gd_stns = True  # general filter, Indicator kriging
use_temporal_filter_after_kriging = True  # on day filter

use_first_neghbr_as_gd_stns = True  # False
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
# , '180min', '360min', '720min', '1440min']
resample_frequencies = ['60min']
# '720min', '1440min']  # '60min', '360min'
# '120min', '180min',
title_ = r'Qt_max_dwd_netatmo'


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
    path_to_dwd_vgs = path_to_vgs / \
        (r'vg_strs_dwd_%s_maximum_100_event.csv' % temp_agg)

    path_dwd_extremes_df = path_to_data / \
        (r'dwd_%s_maximum_100_event.csv' % temp_agg)

    # Files to use
    #==========================================================================

    if qunatile_kriging:
        netatmo_data_to_use = path_to_netatmo_edf
        dwd_data_to_use = path_to_dwd_edf
#         path_to_dwd_vgs = path_to_dwd_vgs

    print(title_)
    # DWD DATA
    #=========================================================================
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
    # df results

    df_counts_maxs = pd.DataFrame(index=dwd_in_extremes_df.index)
    df_counts_maxs_ppt = pd.DataFrame(index=dwd_in_extremes_df.index)

    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()

    for event_date in dwd_in_extremes_df.index:
        #                 if event_date == '2016-08-18 20:00:00':
        #                             print(event_date)
        #                     raise Exception
        _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
        if len(_stn_id_event_) < 5:
            _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                '0' + _stn_id_event_

        _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
        _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]
        print('**Calculating for Date ',
              event_date, '\n Rainfall: ',  _ppt_event_,
              'Quantile: ', _edf_event_, ' **\n')

        #==============================================================
        # # DWD qunatiles
        #==============================================================
        edf_dwd_vals = []
        dwd_xcoords = []
        dwd_ycoords = []
        dwd_stn_ids = []

        ppt_dwd_vals = []
        for stn_id in all_dwd_stns:
            #print('station is', stn_id)

            edf_stn_vals = dwd_in_vals_df.loc[
                event_date, stn_id]
            ppt_stn_val = dwd_in_ppt_vals_df.loc[
                event_date, stn_id]
            if ppt_stn_val >= 0:
                ppt_dwd_vals.append(ppt_stn_val)

            if edf_stn_vals > 0:
                edf_dwd_vals.append(np.round(edf_stn_vals, 4))
                dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
                dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
                dwd_stn_ids.append(stn_id)

        dwd_xcoords = np.array(dwd_xcoords)
        dwd_ycoords = np.array(dwd_ycoords)
        edf_dwd_vals = np.array(edf_dwd_vals)

        ppt_dwd_vals = np.array(ppt_dwd_vals)
        # get vg model for this day
        vgs_model_dwd = df_vgs_extremes.loc[event_date, 1]

        if not isinstance(vgs_model_dwd, str):
            vgs_model_dwd = df_vgs_extremes.loc[event_date, 2]

        if not isinstance(vgs_model_dwd, str):
            vgs_model_dwd = ''

        if ('Nug' in vgs_model_dwd or len(
            vgs_model_dwd) == 0) and (
            'Exp' not in vgs_model_dwd and
                'Sph' not in vgs_model_dwd):

            try:
                for i in range(2, len(df_vgs_extremes.loc[event_date, :])):
                    vgs_model_dwd = df_vgs_extremes.loc[event_date, i]
                    if type(vgs_model_dwd) == np.float:
                        continue
                    if ('Nug' in vgs_model_dwd
                            or len(vgs_model_dwd) == 0) and (
                                'Exp' not in vgs_model_dwd or
                            'Sph' not in vgs_model_dwd):
                        continue
                    else:
                        break

            except Exception as msg:
                print(msg)
                print(
                    'Only Nugget variogram for this day')

        if not isinstance(vgs_model_dwd, str):
            vgs_model_dwd = ''

        # if type(vgs_model_dwd) != np.float and len(vgs_model_dwd) >
        # 0:
        if ('Nug' in vgs_model_dwd
                or len(vgs_model_dwd) > 0) and (
                'Exp' in vgs_model_dwd or
                'Sph' in vgs_model_dwd):

            print('**Variogram model **\n', vgs_model_dwd)

            # Netatmo data and coords
            netatmo_df = netatmo_in_vals_df.loc[
                event_date,
                :].dropna(
                how='all')

            if netatmo_df.size > 0:
                # =========================================================
                # # NETATMO QUANTILES
                # =========================================================

                edf_netatmo_vals = []
                netatmo_xcoords = []
                netatmo_ycoords = []
                netatmo_stn_ids = []

                ppt_netatmo_vals = []
                for i, netatmo_stn_id in enumerate(netatmo_df.index):
                    print('Station number %d / %d'
                          % (i, len(netatmo_df.index)))
                    netatmo_edf_event_ = netatmo_in_vals_df.loc[
                        event_date, netatmo_stn_id]
                    if netatmo_edf_event_ >= 0.5:
                        # print('Correcting Netatmo station',
                        #                                   netatmo_stn_id)
                        try:

                            #==========================================
                            # # Correct Netatmo Quantiles
                            #==========================================

                            x_netatmo_interpolate = np.array(
                                [netatmo_in_coords_df.loc[netatmo_stn_id, 'X']])
                            y_netatmo_interpolate = np.array(
                                [netatmo_in_coords_df.loc[netatmo_stn_id, 'Y']])

                            netatmo_stn_edf_df = netatmo_in_vals_df.loc[
                                :, netatmo_stn_id].dropna()

                            netatmo_start_date = netatmo_stn_edf_df.index[0]
                            netatmo_end_date = netatmo_stn_edf_df.index[-1]

                            netatmo_ppt_event_ = netatmo_in_ppt_vals_df.loc[event_date,
                                                                            netatmo_stn_id]

                            print('\nOriginal Netatmo Ppt: ', netatmo_ppt_event_,
                                  '\nOriginal Netatmo Edf: ', netatmo_edf_event_)

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

                                    if netatmo_edf_event_ == 1:
                                        # correct edf event, 1 is due to
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

                                    try:
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
                                    except Exception as msg:
                                        print(msg)
                                        continue

                            ppt_vals_dwd_new_for_obsv_ppt = np.array(
                                ppt_vals_dwd_new_for_obsv_ppt)
                            dwd_xcoords_new = np.array(dwd_xcoords_new)
                            dwd_ycoords_new = np.array(dwd_ycoords_new)

                            ordinary_kriging_dwd_netatmo_crt = OrdinaryKriging(
                                xi=dwd_xcoords_new,
                                yi=dwd_ycoords_new,
                                zi=ppt_vals_dwd_new_for_obsv_ppt,
                                xk=x_netatmo_interpolate,
                                yk=y_netatmo_interpolate,
                                model=vgs_model_dwd)

                            try:
                                ordinary_kriging_dwd_netatmo_crt.krige()
                            except Exception as msg:
                                print('Error while Kriging', msg)

                            interpolated_netatmo_prct = ordinary_kriging_dwd_netatmo_crt.zk.copy()

                            if interpolated_netatmo_prct < 0:
                                interpolated_netatmo_prct = np.nan

                            print('**Interpolated PPT by DWD recent: ',
                                  interpolated_netatmo_prct)

                            if interpolated_netatmo_prct >= 0.:
                                ppt_netatmo_vals.append(
                                    interpolated_netatmo_prct[0])
                                # get for each dwd stn the percentile corresponding
                                # to interpolated ppt and reinterpolate percentiles
                                # at Netatmo location

                                dwd_xcoords_old = []
                                dwd_ycoords_old = []
                                edf_vals_dwd_old_for_interp_ppt = []

                                for dwd_stn_id_old in dwd_in_ppt_vals_df_old.columns:
                                    ppt_vals_stn_old = dwd_in_ppt_vals_df_old.loc[
                                        :, dwd_stn_id_old].dropna()

                                    ppt_stn_old, edf_stn_old = build_edf_fr_vals(
                                        ppt_vals_stn_old)
                                    nearst_ppt_old = find_nearest(
                                        ppt_stn_old,
                                        interpolated_netatmo_prct[0])

                                    edf_idx = np.where(
                                        ppt_stn_old == nearst_ppt_old)

                                    try:
                                        if edf_idx[0].size > 1:
                                            edf_for_ppt = np.mean(
                                                edf_stn_old[edf_idx])
                                        else:
                                            edf_for_ppt = edf_stn_old[edf_idx]
                                    except Exception as msg:
                                        print(msg)
                                    try:
                                        edf_for_ppt = edf_for_ppt[0]

                                    except Exception as msg:
                                        edf_for_ppt = edf_for_ppt

                                    if edf_for_ppt >= 0:
                                        edf_vals_dwd_old_for_interp_ppt.append(
                                            edf_for_ppt)
                                        stn_xcoords_old = dwd_in_coords_df.loc[
                                            dwd_stn_id_old, 'X']
                                        stn_ycoords_old = dwd_in_coords_df.loc[
                                            dwd_stn_id_old, 'Y']

                                        dwd_xcoords_old.append(
                                            stn_xcoords_old)
                                        dwd_ycoords_old.append(
                                            stn_xcoords_old)

                                edf_vals_dwd_old_for_interp_ppt = np.array(
                                    edf_vals_dwd_old_for_interp_ppt)
                                dwd_xcoords_old = np.array(
                                    dwd_xcoords_old)
                                dwd_ycoords_old = np.array(
                                    dwd_ycoords_old)

                                vgs_model_dwd_old = vgs_model_dwd
                                # Kriging back again
                                ordinary_kriging_dwd_only_old = OrdinaryKriging(
                                    xi=dwd_xcoords_old,
                                    yi=dwd_ycoords_old,
                                    zi=edf_vals_dwd_old_for_interp_ppt,
                                    xk=x_netatmo_interpolate,
                                    yk=y_netatmo_interpolate,
                                    model=vgs_model_dwd_old)

                                try:
                                    ordinary_kriging_dwd_only_old.krige()

                                except Exception as msg:
                                    print('Error while Kriging', msg)

                                interpolated_vals_dwd_old = ordinary_kriging_dwd_only_old.zk.copy()

                                print('**Interpolated DWD '
                                      'Netatmo Percentile ',
                                      interpolated_vals_dwd_old)

                                if interpolated_vals_dwd_old < 0:
                                    interpolated_vals_dwd_old = np.nan

                            else:
                                #print('no good variogram found, adding nans to df')
                                interpolated_vals_dwd_old = np.nan

                            #==========================================
                            #
                            #==========================================

                            if interpolated_vals_dwd_old >= 0:
                                edf_netatmo_vals.append(
                                    np.round(interpolated_vals_dwd_old[0], 4))
                                netatmo_xcoords.append(
                                    netatmo_in_coords_df.loc[netatmo_stn_id, 'X'])
                                netatmo_ycoords.append(
                                    netatmo_in_coords_df.loc[netatmo_stn_id, 'Y'])
                                netatmo_stn_ids.append(netatmo_stn_id)

                        except KeyError:
                            continue
                    else:
                        print('*#+ No need to correct QT, to small#+*')
                        edf_netatmo_vals.append(
                            np.round(netatmo_edf_event_, 4))
                        netatmo_xcoords.append(
                            netatmo_in_coords_df.loc[netatmo_stn_id, 'X'])
                        netatmo_ycoords.append(
                            netatmo_in_coords_df.loc[netatmo_stn_id, 'Y'])
                        netatmo_stn_ids.append(netatmo_stn_id)

                netatmo_xcoords = np.array(netatmo_xcoords).ravel()
                netatmo_ycoords = np.array(netatmo_ycoords).ravel()

    #                     xy_netatmo = list(set([(x, y) for x, y in zip(
    #                         netatmo_xcoords, netatmo_ycoords)]))

                edf_netatmo_vals = np.array(edf_netatmo_vals).ravel()

                # add results to df ppt
                try:
                    max_netatmo_ppt = max(ppt_netatmo_vals)
                except Exception as msg:
                    print('Error finfing max netatmo ', msg)
                    max_netatmo_ppt = 0
                max_dwd_ppt = max(ppt_dwd_vals)
                if max_netatmo_ppt > max_dwd_ppt:
                    max_at_netatmo = 1
                    max_at_dwd = 0
                if max_netatmo_ppt < max_dwd_ppt:
                    max_at_netatmo = 0
                    max_at_dwd = 1
                if max_netatmo_ppt == max_dwd_ppt:
                    max_at_netatmo = 0
                    max_at_dwd = 0
                df_counts_maxs_ppt.loc[event_date,
                                       'max_netatmo_ppt'] = max_netatmo_ppt
                df_counts_maxs_ppt.loc[event_date, 'max_dwd_ppt'] = max_dwd_ppt
                df_counts_maxs_ppt.loc[event_date,
                                       'max_at_netatmo'] = max_at_netatmo
                df_counts_maxs_ppt.loc[event_date, 'max_at_dwd'] = max_at_dwd

                if use_temporal_filter_after_kriging:

                    print('using DWD stations to find Netatmo values')
                    print('apllying on event filter')

                    ordinary_kriging_filter_netamto = OrdinaryKriging(
                        xi=dwd_xcoords,
                        yi=dwd_ycoords,
                        zi=edf_dwd_vals,
                        xk=netatmo_xcoords,
                        yk=netatmo_ycoords,
                        model=vgs_model_dwd)

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
                        edf_netatmo_vals - interpolated_vals)

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

                        print('**Removing bad stations and kriging**')

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
                    netatmo_dry_gd = edf_gd_vals_df[
                        edf_gd_vals_df.values < 0.9]
                    # gd

                    cmn_wet_dry_stns = netatmo_in_coords_df.index.intersection(
                        netatmo_dry_gd.index)
                    x_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
                        cmn_wet_dry_stns, 'X'].values.ravel()
                    y_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
                        cmn_wet_dry_stns, 'Y'].values.ravel()

                    netatmo_wet_gd = edf_gd_vals_df[
                        edf_gd_vals_df.values >= 0.9]
                    cmn_wet_gd_stns = netatmo_in_coords_df.index.intersection(
                        netatmo_wet_gd.index)
                    x_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
                        cmn_wet_gd_stns, 'X'].values.ravel()
                    y_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
                        cmn_wet_gd_stns, 'Y'].values.ravel()
                    #====================================
                    #
                    #====================================
                    edf_bad_vals_df = netatmo_df.loc[ids_netatmo_stns_bad]

                    netatmo_dry_bad = edf_bad_vals_df[
                        edf_bad_vals_df.values < 0.9]
                    cmn_dry_bad_stns = netatmo_in_coords_df.index.intersection(
                        netatmo_dry_bad.index)
                    # netatmo bad
                    netatmo_wet_bad = edf_bad_vals_df[
                        edf_bad_vals_df.values >= 0.9]
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
                    # find if wet bad is really wet bad
                    # find neighboring netatmo stations wet good

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
                                 in zip(x_coords_gd_netatmo_wet,
                                        y_coords_gd_netatmo_wet)])

                            # get distance to neighboring wet stations
                            sep_dist = distance.cdist(stn_coords,
                                                      neighbors_coords,
                                                      'euclidean')
                            # create a tree from coordinates
                            points_tree = spatial.cKDTree(
                                neighbors_coords)

                            # This finds the index of all points within
                            # distance 1 of [1.5,2.5].
                            idxs_neighbours = points_tree.query_ball_point(
                                np.array((netatmo_x_stn,
                                          netatmo_y_stn)),
                                radius)

                        #==================================================
                        # Final QUANTILES
                        #==================================================

                        edf_netatmo_vals_gd = edf_netatmo_vals[idx_good_stns[0]]
                        netatmo_xcoords_gd = netatmo_xcoords[idx_good_stns[0]]
                        netatmo_ycoords_gd = netatmo_ycoords[idx_good_stns[0]]

                        netatmo_stn_ids_gd = [netatmo_stn_ids[ix]
                                              for ix in idx_good_stns[0]]

                        # keep only good stations
#                         df_gd_stns_per_event = netatmo_df.loc[ids_netatmo_stns_gd]

                        print('\n----Keeping %d / %d Stns for event---'
                              % (edf_netatmo_vals_gd.size,
                                 edf_netatmo_vals.size))
                        #a = distance_matrix(coords_netatmo_all, coords_netatmo_all)

                    else:
                        print('Netatmo second filter not used')
                        #==================================================
                        edf_netatmo_vals_gd = edf_netatmo_vals
                        netatmo_xcoords_gd = netatmo_xcoords
                        netatmo_ycoords_gd = netatmo_ycoords
                        netatmo_stn_ids_gd = netatmo_stn_ids

                dwd_netatmo_xcoords = np.concatenate(
                    [dwd_xcoords, netatmo_xcoords_gd])
                dwd_netatmo_ycoords = np.concatenate(
                    [dwd_ycoords, netatmo_ycoords_gd])

                dwd_netatmo_edf = np.concatenate([edf_dwd_vals,
                                                  edf_netatmo_vals_gd])
                # add results to df
                max_netatmo_edf = max(edf_netatmo_vals_gd)
                max_dwd_edf = max(edf_dwd_vals)
                if max_netatmo_edf > max_dwd_edf:
                    max_at_netatmo = 1
                    max_at_dwd = 0
                if max_netatmo_edf < max_dwd_edf:
                    max_at_netatmo = 0
                    max_at_dwd = 1
                if max_netatmo_edf == max_dwd_edf:
                    max_at_netatmo = 0
                    max_at_dwd = 0
                df_counts_maxs.loc[event_date,
                                   'max_netatmo_edf'] = max_netatmo_edf
                df_counts_maxs.loc[event_date, 'max_dwd_edf'] = max_dwd_edf
                df_counts_maxs.loc[event_date,
                                   'max_at_netatmo'] = max_at_netatmo
                df_counts_maxs.loc[event_date, 'max_at_dwd'] = max_at_dwd

    df_counts_maxs.dropna(how='all', inplace=True)
    df_counts_maxs.to_csv(
        out_plots_path / (
            'ratios_max_dwd_max_netatmo.csv'),
        sep=';', float_format='%0.6f')

    df_counts_maxs_ppt.dropna(how='all', inplace=True)
    df_counts_maxs_ppt.to_csv(
        out_plots_path / (
            'ppt_ratios_max_dwd_max_netatmo.csv'),
        sep=';', float_format='%0.6f')

stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
