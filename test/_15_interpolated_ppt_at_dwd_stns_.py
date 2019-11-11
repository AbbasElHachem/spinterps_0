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
from pykrige.ok import OrdinaryKriging as OKpy
import skgstat as skg
#from scipy.spatial import distance_matrix
from scipy.spatial import distance
from scipy import spatial

from pathlib import Path
from random import shuffle

import kriging_func as kf

VG = variograms.vgs.Variogram


plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})


# =============================================================================

# main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
main_dir = Path(r'/home/abbas/Documents/Python/Extremes')

os.chdir(main_dir)

path_to_data = main_dir / r'NetAtmo_BW'

path_to_vgs = main_dir / r'kriging_ppt_netatmo'

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# path for data filter
in_filter_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================

use_dwd_stns_for_kriging = True

qunatile_kriging = True

# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging
use_temporal_filter_after_kriging = False  # on day filter

use_first_neghbr_as_gd_stns = False  # False
use_first_and_second_nghbr_as_gd_stns = True  # True

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
resample_frequencies = ['60min',  '360min',
                        '720min', '1440min']
# '120min', '180min',
title_ = r'Qt_ok_ok_un_ppt_'


if not use_netatmo_gd_stns:
    title_ = title_ + '_netatmo_no_flt_'

if use_netatmo_gd_stns:
    title_ = title_ + '_first_flt_'

if use_temporal_filter_after_kriging:

    title_ = title_ + '_temp_flt_'

# def out plot path based on combination

plot_2nd_filter_netatmo = True
#==============================================================================
#
#==============================================================================

plot_events = False

strt_date = '2018-05-15 00:00:00'
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

    path_to_dwd_vgs = (path_to_vgs /
                       (r'vg_strs_dwd_%s_maximum_100_event.csv' % temp_agg))

    path_dwd_extremes_df = path_to_data / \
        (r'dwd_%s_maximum_100_event.csv' % temp_agg)

    # Files to use
    #==========================================================================

    if qunatile_kriging:
        netatmo_data_to_use = path_to_netatmo_edf
        dwd_data_to_use = path_to_dwd_edf
        path_to_dwd_vgs = path_to_dwd_vgs

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
    print('\n%d Extreme Event to interpolate\n' % dwd_in_extremes_df.shape[0])
    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()
    shuffle(all_dwd_stns)
    shuffled_dwd_stns_10stn = np.array(list(chunks(all_dwd_stns, 10)))

    #==========================================================================
    # CREATE DFS HOLD RESULT KRIGING PER NETATMO STATION
    #==========================================================================
    for idx_lst_comb in range(len(shuffled_dwd_stns_10stn)):
        stn_comb = shuffled_dwd_stns_10stn[idx_lst_comb]

        print('Interpolating for following DWD stations: \n',
              pprint.pformat(stn_comb))

        df_interpolated_dwd_netatmos_comb = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_netatmos_comb_un = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_only = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_netatmos_comb_corr = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_netatmos_comb_un_corr = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        #======================================================================
        # START KRIGING
        #======================================================================

        for stn_dwd_id in stn_comb:

            print('interpolating for DWD Station', stn_dwd_id)

            x_dwd_interpolate = np.array([dwd_in_coords_df.loc[stn_dwd_id, 'X']])
            y_dwd_interpolate = np.array([dwd_in_coords_df.loc[stn_dwd_id, 'Y']])

            # drop stns
            all_dwd_stns_except_interp_loc = [
                stn for stn in dwd_in_vals_df.columns if stn not in stn_comb]

            if use_temporal_filter_after_kriging:
                print('\n++Creating DF to hold filtered netatmo stns++\n')
                df_stns_netatmo_gd_event = pd.DataFrame(
                    index=dwd_in_extremes_df.index,
                    columns=netatmo_in_vals_df.columns,
                    data=np.ones(shape=(dwd_in_extremes_df.index.shape[0],
                                        netatmo_in_vals_df.columns.shape[0])))

            for event_date in dwd_in_extremes_df.index[58:70]:

                _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
                if len(_stn_id_event_) < 5:
                    _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                        '0' + _stn_id_event_

                _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
                _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]
                print('**Calculating for Date ',
                      event_date, '\n Rainfall: ',  _ppt_event_,
                      'Quantile: ', _edf_event_, ' **\n')

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
                
                # try to fit vg
                dwd_xy_coords = np.array([
                    (x,y) for x, y in zip(x_dwd_all, y_dwd_all)])
                fit_vg_tst = skg.Variogram(coordinates=dwd_xy_coords, bin_func='even',
                                           values=ppt_dwd_vals_nona.values)
                #print(fit_vg_tst.describe())
                #fit_vg_tst.plot()
                # get vg model for this day
                vgs_model_dwd_ppt = df_vgs_extremes.loc[event_date, 1]

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
                
                try:
                    
                    if 'Nug' in vgs_model_dwd_ppt:
                        pass
                except Exception:
                    print('Ese')
                if ('Nug' in vgs_model_dwd_ppt
                    or len(vgs_model_dwd_ppt) > 0) and (
                    'Exp' in vgs_model_dwd_ppt or
                        'Sph' in vgs_model_dwd_ppt):

                    print('\n+++ KRIGING PPT at DWD +++\n')

                    # netatmo stns for this event
                    netatmo_df = netatmo_in_vals_df.loc[event_date, :].dropna(
                        how='all')

                    # list to hold result of all netatmo station for this event
                    netatmo_ppt_vals_fr_dwd_interp = []
                    netatmo_ppt_vals_fr_dwd_interp_unc = []
                    x_netatmo_ppt_vals_fr_dwd_interp = []
                    y_netatmo_ppt_vals_fr_dwd_interp = []

                    netatmo_stns_event_ = []
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

                        if netatmo_edf_event_ > 0.9:
                            print('Correcting Netatmo station',
                                  netatmo_stn_id)
                            try:
                                #==============================================
                                # # Correct Netatmo Quantiles
                                #==============================================

                                netatmo_stn_edf_df = netatmo_in_vals_df.loc[
                                    :, netatmo_stn_id].dropna()

                                netatmo_start_date = netatmo_stn_edf_df.index[0]
                                netatmo_end_date = netatmo_stn_edf_df.index[-1]

                                print('\nOriginal Netatmo Ppt: ',
                                      netatmo_ppt_event_,
                                      '\nOriginal Netatmo Edf: ',
                                      netatmo_edf_event_)
                                # assign uncertainty to netatmo edf

                                netatmo_edf_event_unc = netatmo_edf_event_ + (
                                    (1 - netatmo_edf_event_) * 0.1)

                                # select dwd stations with only same period as
                                # netatmo stn
                                ppt_dwd_stn_vals = dwd_in_ppt_vals_df.loc[
                                    netatmo_start_date:netatmo_end_date,
                                    :].dropna(how='all')

                                dwd_xcoords_new = []
                                dwd_ycoords_new = []
                                ppt_vals_dwd_new_for_obsv_ppt = []

                                ppt_vals_dwd_new_for_obsv_ppt_unc = []

                                for dwd_stn_id_new in ppt_dwd_stn_vals.columns:

                                    ppt_vals_stn_new = ppt_dwd_stn_vals.loc[
                                        :, dwd_stn_id_new].dropna()

                                    if ppt_vals_stn_new.size > 0:

                                        ppt_stn_new, edf_stn_new = build_edf_fr_vals(
                                            ppt_vals_stn_new)

                                        if netatmo_edf_event_ == 1:
                                            # correct edf event, 1 is bcz
                                            # rounding error
                                            ppt_ix_edf = find_nearest(
                                                ppt_stn_new,
                                                _ppt_event_)
                                            netatmo_edf_event_ = edf_stn_new[
                                                np.where(
                                                    ppt_stn_new == ppt_ix_edf)]
                                            netatmo_edf_event_unc = netatmo_edf_event_ + (
                                                (1 - netatmo_edf_event_) * 0.1)

                                        nearst_edf_new = find_nearest(
                                            edf_stn_new,
                                            netatmo_edf_event_)

                                        ppt_idx = np.where(
                                            edf_stn_new == nearst_edf_new)

                                        # find for uncertainty edf
                                        nearst_edf_new_unc = find_nearest(
                                            edf_stn_new,
                                            netatmo_edf_event_unc)

                                        ppt_idx_unc = np.where(
                                            edf_stn_new == nearst_edf_new_unc)

                                        try:
                                            if ppt_idx[0].size > 1:
                                                ppt_for_edf = np.mean(
                                                    ppt_stn_new[ppt_idx])
                                            else:
                                                ppt_for_edf = ppt_stn_new[ppt_idx]
                                        except Exception as msg:
                                            print(msg)

                                        try:
                                            if ppt_idx_unc[0].size > 1:
                                                ppt_for_edf_unc = np.mean(
                                                    ppt_stn_new[ppt_idx_unc])
                                            else:
                                                ppt_for_edf_unc = ppt_stn_new[
                                                    ppt_idx_unc]
                                        except Exception as msg:
                                            print(msg)

                                        if ppt_for_edf[0] >= 0:

                                            ppt_vals_dwd_new_for_obsv_ppt.append(
                                                ppt_for_edf[0])

                                            ppt_vals_dwd_new_for_obsv_ppt_unc.append(
                                                ppt_for_edf_unc[0])

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
                                ppt_vals_dwd_new_for_obsv_ppt_unc = np.array(
                                    ppt_vals_dwd_new_for_obsv_ppt_unc)

                                dwd_xcoords_new = np.array(dwd_xcoords_new)
                                dwd_ycoords_new = np.array(dwd_ycoords_new)
                                
                                
                                try:
                                    print('\n+++ KRIGING CORRECTING QT +++\n')

                                    ordinary_kriging_dwd_netatmo_crt = OrdinaryKriging(
                                        xi=dwd_xcoords_new,
                                        yi=dwd_ycoords_new,
                                        zi=ppt_vals_dwd_new_for_obsv_ppt,
                                        xk=x_netatmo_interpolate,
                                        yk=y_netatmo_interpolate,
                                        model=vgs_model_dwd_ppt)
                                    
                                    ordinary_kriging_dwd_netatmo_crt_unc = OrdinaryKriging(
                                        xi=dwd_xcoords_new,
                                        yi=dwd_ycoords_new,
                                        zi=ppt_vals_dwd_new_for_obsv_ppt_unc,
                                        xk=x_netatmo_interpolate,
                                        yk=y_netatmo_interpolate,
                                        model=vgs_model_dwd_ppt)

                                    try:
                                        ordinary_kriging_dwd_netatmo_crt.krige()
                                        ordinary_kriging_dwd_netatmo_crt_unc.krige()
                                    except Exception as msg:
                                        print('Error while Kriging', msg)

                                    interpolated_netatmo_prct = ordinary_kriging_dwd_netatmo_crt.zk.copy()
                                    interpolated_netatmo_prct_unc = ordinary_kriging_dwd_netatmo_crt_unc.zk.copy()

                                    if interpolated_netatmo_prct < 0:
                                        interpolated_netatmo_prct = np.nan

                                    if interpolated_netatmo_prct_unc < 0:
                                        interpolated_netatmo_prct_unc = np.nan

                                    print('**Interpolated PPT by DWD recent: \n',
                                          interpolated_netatmo_prct)
                                    print('**Interpolated PPT by DWD recent Unc: ',
                                          interpolated_netatmo_prct_unc)

                                    if interpolated_netatmo_prct >= 0.:
                                        netatmo_ppt_vals_fr_dwd_interp.append(
                                            interpolated_netatmo_prct[0])

                                        netatmo_ppt_vals_fr_dwd_interp_unc.append(
                                            interpolated_netatmo_prct_unc[0])

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
                            print('QT to small no need to correct it\n')
                            # check if nan, if so don't add it
                            if netatmo_ppt_event_ >= 0:
                                netatmo_ppt_vals_fr_dwd_interp.append(
                                    netatmo_ppt_event_)

                                netatmo_ppt_vals_fr_dwd_interp_unc.append(
                                    netatmo_ppt_event_)
                                x_netatmo_ppt_vals_fr_dwd_interp.append(
                                    x_netatmo_interpolate[0])

                                y_netatmo_ppt_vals_fr_dwd_interp.append(
                                    y_netatmo_interpolate[0])

                                netatmo_stns_event_.append(netatmo_stn_id)
                
                # Transform everything to arrays and combine dwd-netatmo           

                netatmo_xcoords = np.array(
                    x_netatmo_ppt_vals_fr_dwd_interp).ravel()
                netatmo_ycoords = np.array(
                    y_netatmo_ppt_vals_fr_dwd_interp).ravel()
                
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

                netatmo_dwd_ppt_vals_unc = np.concatenate(
                    [ppt_netatmo_vals_unc,
                     ppt_dwd_vals_nona.values])
                print('Krigging PPT at DWD Stns')

                #===============================================================
                # Start kriging ppt at DWD 
                #===============================================================
           
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

                interpolated_netatmo_dwd_ppt = ordinary_kriging_dwd_netatmo_ppt.zk.copy()
                interpolated_dwd_ppt = ordinary_kriging_dwd_ppt.zk.copy()
                interpolated_netatmo_dwd_ppt_unc = ordinary_kriging_dwd_netatmo_ppt_unc.zk.copy()

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
            
            # Aplly second filter
            if interpolated_netatmo_dwd_ppt > 0:
                # TODO: Continue here
                if use_temporal_filter_after_kriging:

                    print('using DWD stations to find Netatmo values')
                    print('apllying on event filter')

                    ordinary_kriging_dwd_ppt_correct_netatmo = OrdinaryKriging(
                        xi=x_dwd_all,
                        yi=y_dwd_all,
                        zi=ppt_dwd_vals_nona,
                        xk=x_netatmo_ppt_vals_fr_dwd_interp,
                        yk=y_netatmo_ppt_vals_fr_dwd_interp,
                        model=vgs_model_dwd_ppt)

                    try:
                        ordinary_kriging_dwd_ppt_correct_netatmo.krige()
                    except Exception as msg:
                        print('Error while Error Kriging', msg)
                        continue

                    # interpolated vals
                    interpolated_vals = ordinary_kriging_dwd_ppt_correct_netatmo.zk

                    # calcualte standard deviation of estimated values
                    std_est_vals = np.sqrt(
                        ordinary_kriging_dwd_ppt_correct_netatmo.est_vars)
                    # calculate difference observed and estimated values
                    diff_obsv_interp = np.abs(
                        netatmo_ppt_vals_fr_dwd_interp - interpolated_vals)

                    #======================================================
                    # # use additional temporal filter
                    #======================================================
                    idx_good_stns = np.where(
                        diff_obsv_interp <= 3 * std_est_vals)
                    idx_bad_stns = np.where(
                        diff_obsv_interp > 3 * std_est_vals)

                    if len(idx_good_stns[0]) > 0 or len(idx_bad_stns[0]) > 0: 
                        print('Number of Stations with bad index \n',
                              len(idx_bad_stns[0]))
                        print('Number of Stations with good index \n',
                              len(idx_good_stns[0]))

                        print('**Removing bad stations and kriging**')

                        # use additional filter
                        try:
                            ids_netatmo_stns_gd = np.take(
                                netatmo_df.index,
                                idx_good_stns).ravel()
                            ids_netatmo_stns_bad = np.take(
                                netatmo_df.index,
                                idx_bad_stns).ravel()

                        except Exception as msg:
                            print(msg)

                    ppt_gd_vals_df = netatmo_in_ppt_vals_df.loc[ids_netatmo_stns_gd]
                    netatmo_dry_gd = ppt_gd_vals_df[
                        ppt_gd_vals_df.values < 0.9]
                    # gd
                    netatmo_wet_gd = ppt_gd_vals_df[
                        ppt_gd_vals_df.values >= 0.9]

                    x_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
                        netatmo_dry_gd.index, 'X'].values.ravel()
                    y_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
                        netatmo_dry_gd.index, 'Y'].values.ravel()

                    x_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
                        netatmo_wet_gd.index, 'X'].values.ravel()
                    y_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
                        netatmo_wet_gd.index, 'Y'].values.ravel()
                    # ===================================
                    #
                    # ===================================
                    ppt_bad_vals_df = netatmo_in_ppt_vals_df.loc[
                        ids_netatmo_stns_bad]

                    netatmo_dry_bad = ppt_bad_vals_df[
                        ppt_bad_vals_df.values < 0.9]
                    # netatmo bad
                    netatmo_wet_bad = ppt_bad_vals_df[
                        ppt_bad_vals_df.values >= 0.9]
                    # dry bad
                    x_coords_bad_netatmo_dry = netatmo_in_coords_df.loc[
                        netatmo_dry_bad.index, 'X'].values.ravel()
                    y_coords_bad_netatmo_dry = netatmo_in_coords_df.loc[
                        netatmo_dry_bad.index, 'Y'].values.ravel()
                    # wet bad
                    x_coords_bad_netatmo_wet = netatmo_in_coords_df.loc[
                        netatmo_wet_bad.index, 'X'].values.ravel()
                    y_coords_bad_netatmo_wet = netatmo_in_coords_df.loc[
                        netatmo_wet_bad.index, 'Y'].values.ravel()
                    # find if wet bad is really wet bad
                    # find neighboring netatmo stations wet good

                    if netatmo_wet_bad.size > 0 and netatmo_wet_gd.size > 0:

                        for stn_, ppt_stn, netatmo_x_stn, netatmo_y_stn in zip(
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
                            points_tree = spatial.cKDTree(neighbors_coords)

                            # This finds the index of all points within
                            # distance 1 of [1.5,2.5].
                            idxs_neighbours = points_tree.query_ball_point(
                                np.array((netatmo_x_stn,
                                          netatmo_y_stn)),
                                radius)
                            if len(idxs_neighbours) > 0:

                                stns_ids_bad_corrected = []
                                stns_xcoords_bad_corrected = []
                                stns_ycoords_bad_corrected = []
                                edf_bad_corrected = []

                                for i, ix_nbr in enumerate(idxs_neighbours):

                                    ppt_neighbor = netatmo_wet_gd[ix_nbr]
                                    if np.abs(ppt_stn - ppt_neighbor) <= diff_thr:
                                        print(
                                            'bad wet netatmo station is good')
                                        stns_ids_bad_corrected.append(stn_)
                                        stns_xcoords_bad_corrected.append(
                                            netatmo_x_stn)
                                        stns_ycoords_bad_corrected.append(
                                            netatmo_y_stn)

                                        edf_bad_corrected.append(ppt_stn)

                                        # remove from original bad wet
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

                                    else:
                                        print('bad wet is bad wet')
                            else:
                                print('\nStn has no near neighbors')

                    df_stns_netatmo_gd_event.loc[
                        event_date,
                        ids_netatmo_stns_bad] = -999

                    dwd_dry = ppt_dwd_vals_nona[ppt_dwd_vals_nona < 0.9]
                    dwd_wet = ppt_dwd_vals_nona[ppt_dwd_vals_nona >= 0.9]

                    x_coords_dwd_wet = x_dwd_all[np.where(
                        ppt_dwd_vals_nona >= 0.9)]
                    y_coords_dwd_wet = y_dwd_all[np.where(
                        ppt_dwd_vals_nona >= 0.9)]

                    x_coords_dwd_dry = x_dwd_all[np.where(
                        ppt_dwd_vals_nona < 0.9)]
                    y_coords_dwd_dry = y_dwd_all[np.where(
                        ppt_dwd_vals_nona < 0.9)]
#                         dwd_dry =
                    if plot_2nd_filter_netatmo:
                        print('Plotting second filter maps')
                        plt.ioff()
#                         texts = []
                        fig = plt.figure(figsize=(20, 20), dpi=75)
                        ax = fig.add_subplot(111)
                        m_size = 50
                        ax.scatter(x_dwd_interpolate,
                                   y_dwd_interpolate, c='k',
                                   marker='*', s=m_size + 25,
                                   label='Interp Stn DWD %s' %
                                   stn_dwd_id)

                        ax.scatter(x_coords_gd_netatmo_dry,
                                   y_coords_gd_netatmo_dry, c='g',
                                   marker='o', s=m_size,
                                   label='Netatmo %d stations with dry good values' %
                                   x_coords_gd_netatmo_dry.shape[0])

                        ax.scatter(x_coords_gd_netatmo_wet,
                                   y_coords_gd_netatmo_wet, c='b',
                                   marker='o', s=m_size,
                                   label='Netatmo %d stations with wet good values' %
                                   x_coords_gd_netatmo_wet.shape[0])

                        ax.scatter(x_coords_bad_netatmo_dry,
                                   y_coords_bad_netatmo_dry, c='orange',
                                   marker='D', s=m_size,
                                   label='Netatmo %d stations with dry bad values' %
                                   x_coords_bad_netatmo_dry.shape[0])

                        ax.scatter(x_coords_bad_netatmo_wet,
                                   y_coords_bad_netatmo_wet, c='r',
                                   marker='D', s=m_size,
                                   label='Netatmo %d stations with wet bad values' %
                                   x_coords_bad_netatmo_wet.shape[0])

                        ax.scatter(x_coords_dwd_dry,
                                   y_coords_dwd_dry, c='g',
                                   marker='x', s=m_size,
                                   label='DWD %d stations with dry values' %
                                   x_coords_dwd_dry.shape[0])

                        ax.scatter(x_coords_dwd_wet,
                                   y_coords_dwd_wet, c='b',
                                   marker='x', s=m_size,
                                   label='DWD %d stations with wet values' %
                                   x_coords_dwd_wet.shape[0])

                        plt.legend(loc=0)

                        plt.grid(alpha=.25)
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
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

                    #======================================================
                    # Interpolate DWD Rainfall
                    #======================================================
                    ppt_netatmo_vals_gd = ppt_netatmo_vals[idx_good_stns[0]]
                    netatmo_xcoords_gd = netatmo_xcoords[idx_good_stns[0]]
                    netatmo_ycoords_gd = netatmo_ycoords[idx_good_stns[0]]

                    netatmo_stn_ids_gd = [netatmo_stns_event_[ix]
                                          for ix in idx_good_stns[0]]

                    # keep only good stations
#                         df_gd_stns_per_event = netatmo_df.loc[ids_netatmo_stns_gd]

                    print('\n----Keeping %d / %d Stns for event---'
                          % (ppt_netatmo_vals_gd.size,
                             ppt_netatmo_vals.size))

                else:
                    print('Netatmo second filter not used')
                    #======================================================
                    ppt_netatmo_vals_gd = ppt_netatmo_vals
                    netatmo_xcoords_gd = netatmo_xcoords
                    netatmo_ycoords_gd = netatmo_ycoords
                    netatmo_stn_ids_gd = netatmo_stns_event_

            # TODO: check from here

                dwd_netatmo_xcoords = np.concatenate(
                    [x_dwd_all, netatmo_xcoords_gd])
                dwd_netatmo_ycoords = np.concatenate(
                    [y_dwd_all, netatmo_ycoords_gd])

                dwd_netatmo_ppt_gd = np.concatenate([ppt_dwd_vals_nona,
                                                     ppt_netatmo_vals_gd])

                #======================================================
                # KRIGING
                #=====================================================
                print('\n+++ KRIGING Corrected PPT +++\n')
                ordinary_kriging_dwd_netatmo_comb_corr = OrdinaryKriging(
                    xi=dwd_netatmo_xcoords,
                    yi=dwd_netatmo_ycoords,
                    zi=dwd_netatmo_ppt_gd,
                    xk=x_dwd_interpolate,
                    yk=y_dwd_interpolate,
                    model=vgs_model_dwd_ppt)

                try:
                    ordinary_kriging_dwd_netatmo_comb_corr.krige()

                except Exception as msg:
                    print('Error while Kriging', msg)

                interpolated_vals_dwd_netatmo_corr = ordinary_kriging_dwd_netatmo_comb_corr.zk.copy()

                print('\n**Interpolated DWD-Netatmo Corr: ',
                      interpolated_vals_dwd_netatmo_corr)

                if interpolated_vals_dwd_netatmo_corr < 0:
                    interpolated_vals_dwd_netatmo_corr = np.nan

            else:
                print('no good variogram found, adding nans to df')
                interpolated_vals_dwd_netatmo_corr = np.nan

            print('+++ Saving result to DF +++\n')

            df_interpolated_dwd_netatmos_comb.loc[
                event_date,
                stn_dwd_id] = interpolated_netatmo_dwd_ppt

            df_interpolated_dwd_netatmos_comb_corr.loc[
                event_date,
                stn_dwd_id] = interpolated_vals_dwd_netatmo_corr

            df_interpolated_dwd_only.loc[
                event_date,
                stn_dwd_id] = interpolated_dwd_ppt

#             df_interpolated_netatmo_only.loc[
#                 event_date,
#                 stn_dwd_id] = interpolated_vals_netatmo_only
#
#             df_interpolated_netatmo_only_un.loc[
#                 event_date,
#                 stn_dwd_id] = interpolated_vals_netatmo_only_un

        if use_temporal_filter_after_kriging:
            df_stns_netatmo_gd_event.to_csv(out_plots_path / (
                'netatmo_2nd_filter_stn_ppt_%s_%s_data_%s_grp_%d_.csv'
                % (stn_dwd_id, temp_agg, title_, idx_lst_comb)),
                sep=';', float_format='%0.2f')

    df_interpolated_dwd_netatmos_comb.dropna(how='all', inplace=True)
    df_interpolated_dwd_netatmos_comb_corr.dropna(how='all', inplace=True)
    df_interpolated_dwd_only.dropna(how='all', inplace=True)
#     df_interpolated_netatmo_only.dropna(how='all', inplace=True)
#     df_interpolated_netatmo_only_un.dropna(how='all', inplace=True)

    df_interpolated_dwd_netatmos_comb.to_csv(out_plots_path / (
        'interpolated_ppt_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
        % (temp_agg, title_, idx_lst_comb, _acc_)),
        sep=';', float_format='%0.2f')

    df_interpolated_dwd_netatmos_comb_corr.to_csv(out_plots_path / (
        'interpolated_ppt_corr_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
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
