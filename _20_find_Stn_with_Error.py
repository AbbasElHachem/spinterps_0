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

from scipy.stats import beta
from scipy.stats import norm
from scipy.special import gamma as gammaf
from scipy.optimize import fmin

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
use_temporal_filter_after_kriging = False  # on day filter

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
resample_frequencies = ['60min', '180min', '360min', '720min', '1440min']
# , '180min', '360min', '720min', '1440min'
# '720min', '1440min']  # '60min', '360min'
# '120min', '180min',
title_ = r'Qt_ok_ok_un_error'


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


def build_edf_fr_vals(ppt_data):
    # Construct EDF, need to check if it works
    """ construct empirical distribution function given data values """
    data_sorted = np.sort(ppt_data, axis=0)[::-1]
    x0 = np.round(np.squeeze(data_sorted)[::-1], 1)
    y0 = np.round((np.arange(data_sorted.size) / len(data_sorted)), 3)

    return x0, y0

#==============================================================================
#
#==============================================================================


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


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
    path_to_dwd_vgs_transf = path_to_vgs / \
        (r'dwd_edf_transf_vg_%s.csv' % temp_agg)
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

    # DWD tranformed data VG
    #==========================================================================

    df_vgs_transf = pd.read_csv(path_to_dwd_vgs_transf,
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
                df_vgs_transf.index), :]

    cmn_netatmo_stns = in_df_distance_netatmo_dwd.index.intersection(
        netatmo_in_vals_df.columns)
    in_df_distance_netatmo_dwd = in_df_distance_netatmo_dwd.loc[
        cmn_netatmo_stns, :]
    # print('\n%d Extreme Event to interpolate\n' % dwd_in_extremes_df.shape[0])

    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()
#     shuffle(all_dwd_stns)
#     shuffled_dwd_stns_10stn = np.array(list(chunks(all_dwd_stns, 10)))
#     for idx_lst_comb in range(len(shuffled_dwd_stns_10stn)):
#         stn_comb = shuffled_dwd_stns_10stn[idx_lst_comb]
    #==============================================================
    # # DWD qunatiles for every event, fit a variogram
    #==============================================================
#     df_vgs_extremes_norm = pd.DataFrame(index=dwd_in_extremes_df.index)
#
#     for event_date in dwd_in_extremes_df.index:
#         #                 if event_date == '2016-08-18 20:00:00':
#         #                     print(event_date)
#         #                     raise Exception
#         _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
#         if len(_stn_id_event_) < 5:
#             _stn_id_event_ = (5 - len(_stn_id_event_)) * \
#                 '0' + _stn_id_event_
#
#         _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
#         _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]
#         print('**Calculating for Date ',
#               event_date, '\n Rainfall: ',  _ppt_event_,
#               'Quantile: ', _edf_event_, ' **\n')
#         edf_dwd_vals = []
#         dwd_xcoords = []
#         dwd_ycoords = []
#         dwd_stn_ids = []
#         for stn_id in all_dwd_stns:
#             #print('station is', stn_id)
#
#             edf_stn_vals = dwd_in_vals_df.loc[
#                 event_date, stn_id]
#
#             if edf_stn_vals > 0:
#                 edf_dwd_vals.append(np.round(edf_stn_vals, 4))
#                 dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
#                 dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
#                 dwd_stn_ids.append(stn_id)
#
#         dwd_xcoords = np.array(dwd_xcoords)
#         dwd_ycoords = np.array(dwd_ycoords)
#         edf_dwd_vals = np.array(edf_dwd_vals)
#         # fit a beta distribution to quantiles
#         alpha1, beta1, xx, yy = beta.fit(edf_dwd_vals)  # , floc=0, fscale=1)
#         # find for every Pi the Beta_Cdf(a, b, Pi)
#         beta_edf_dwd_vals = beta.cdf(edf_dwd_vals, a=alpha1, b=beta1, loc=xx,
#                                      scale=yy)
#         # transform to normal using the inv standard normal
#         std_norm_edf_dwd_vals = norm.ppf(beta_edf_dwd_vals)
#         # fit variogram
#         print('*Done getting data* \n *Fitting variogram*\n')
#         try:
#
#             vg_dwd = VG(
#                 x=dwd_xcoords,
#                 y=dwd_ycoords,
#                 z=std_norm_edf_dwd_vals,
#                 mdr=mdr,
#                 nk=5,
#                 typ='cnst',
#                 perm_r_list=perm_r_list_,
#                 fil_nug_vg=fil_nug_vg,
#                 ld=None,
#                 uh=None,
#                 h_itrs=100,
#                 opt_meth='L-BFGS-B',
#                 opt_iters=1000,
#                 fit_vgs=fit_vgs,
#                 n_best=n_best,
#                 evg_name='robust',
#                 use_wts=False,
#                 ngp=ngp,
#                 fit_thresh=0.01)
#
#             vg_dwd.fit()
#
#             fit_vg_list = vg_dwd.vg_str_list
#
#         except Exception as msg:
#             print(msg)
#             fit_vg_list = ['']
#             # continue
#
#         vgs_model_dwd = fit_vg_list[0]
#         if len(vgs_model_dwd) > 0:
#             df_vgs_extremes_norm.loc[event_date, 1] = vgs_model_dwd
#         else:
#             df_vgs_extremes_norm.loc[event_date, 1] = np.nan
#     df_vgs_extremes_norm.dropna(how='all', inplace=True)
#     df_vgs_extremes_norm.to_csv((path_to_vgs /
#                                  ('dwd_edf_transf_vg_%s.csv' % temp_agg)),
#                                 sep=';')

    #==========================================================================
    # CREATE DFS HOLD RESULT KRIGING PER NETATMO STATION
    #==========================================================================

    for idx_lst_comb in df_dwd_stns_comb.index:

        stn_comb = [stn.replace("'", "")
                    for stn in df_dwd_stns_comb.iloc[idx_lst_comb, :].dropna().values]

#         print('Interpolating for following DWD stations: \n',
#               pprint.pformat(stn_comb))

        df_interpolated_dwd_netatmos_comb = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_netatmos_comb_un = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_only = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_netatmo_only = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_netatmo_only_un = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        # for back transformation

        df_interpolated_dwd_netatmos_comb_std_dev = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_netatmos_comb_un_std_dev = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_only_std_dev = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_netatmo_only_std_dev = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_netatmo_only_un_std_dev = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        #======================================================================
        # START KRIGING
        #======================================================================
        for event_date in dwd_in_extremes_df.index:
            #                 if event_date == '2016-08-18 20:00:00':
            #                     print(event_date)
            #                     raise Exception
            _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
            if len(_stn_id_event_) < 5:
                _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                    '0' + _stn_id_event_

            _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
            _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]
#             print('**Calculating for Date ',
#                   event_date, '\n Rainfall: ',  _ppt_event_,
#                   'Quantile: ', _edf_event_, ' **\n')

            for i, stn_dwd_id in enumerate(stn_comb):

                print('interpolating for DWD Station',
                      stn_dwd_id, i, ' from ', len(stn_comb))

                x_dwd_interpolate = np.array(
                    [dwd_in_coords_df.loc[stn_dwd_id, 'X']])
                y_dwd_interpolate = np.array(
                    [dwd_in_coords_df.loc[stn_dwd_id, 'Y']])
                # dwd ppt data for cdf
                ppt_stn_dwd = dwd_in_ppt_vals_df.loc[
                    :, stn_dwd_id].dropna(how='all').values

                if ppt_stn_dwd.size > 10:

                    _ppt_event_stn = dwd_in_ppt_vals_df.loc[
                        event_date, stn_dwd_id]

                    # build edf
                    try:
                        xppt, yppt = build_edf_fr_vals(ppt_stn_dwd)
                    except Exception as msg:
                        print('Error getting cdf', msg)
                        raise Exception
                        # drop stns
                    all_dwd_stns_except_interp_loc = [
                        stn for stn in dwd_in_vals_df.columns if stn not in stn_comb]

                    # find distance to all dwd stations, sort them, select
                    # minimum
                    distances_dwd_to_stns = in_df_distance_netatmo_dwd.loc[
                        :, stn_dwd_id]

                    sorted_distances_ppt_dwd = distances_dwd_to_stns.sort_values(
                        ascending=True)

        #             # select only neyrby netatmo stations below 30km
                    sorted_distances_ppt_dwd = sorted_distances_ppt_dwd[
                        sorted_distances_ppt_dwd.values <= 3e4]

                    if use_temporal_filter_after_kriging:
                        # print('\n++Creating DF to hold filtered netatmo stns++\n')
                        df_stns_netatmo_gd_event = pd.DataFrame(
                            index=dwd_in_extremes_df.index,
                            columns=netatmo_in_vals_df.columns,
                            data=np.ones(shape=(dwd_in_extremes_df.index.shape[0],
                                                netatmo_in_vals_df.columns.shape[0])))

                    #==========================================================
                    # # DWD qunatiles
                    #==========================================================
                    edf_dwd_vals = []
                    dwd_xcoords = []
                    dwd_ycoords = []
                    dwd_stn_ids = []

                    for stn_id in all_dwd_stns_except_interp_loc:
                        #print('station is', stn_id)

                        edf_stn_vals = dwd_in_vals_df.loc[
                            event_date, stn_id]

                        if edf_stn_vals > 0:
                            edf_dwd_vals.append(np.round(edf_stn_vals, 4))
                            dwd_xcoords.append(
                                dwd_in_coords_df.loc[stn_id, 'X'])
                            dwd_ycoords.append(
                                dwd_in_coords_df.loc[stn_id, 'Y'])
                            dwd_stn_ids.append(stn_id)

                    dwd_xcoords = np.array(dwd_xcoords)
                    dwd_ycoords = np.array(dwd_ycoords)
                    edf_dwd_vals = np.array(edf_dwd_vals)

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

                        #                         print('**Variogram model **\n', vgs_model_dwd)

                        # Netatmo data and coords
                        netatmo_df = netatmo_in_vals_df.loc[
                            event_date,
                            sorted_distances_ppt_dwd.index].dropna(
                            how='all')

                        if netatmo_df.size > 0:
                            # =========================================================
                            # # NETATMO QUANTILES
                            # =========================================================

                            edf_netatmo_vals = []
                            netatmo_xcoords = []
                            netatmo_ycoords = []
                            netatmo_stn_ids = []

                            for i, netatmo_stn_id in enumerate(netatmo_df.index):
                                #                                 print('Correcting Station number %d / %d'
                                #                                       % (i, len(netatmo_df.index)))
                                netatmo_edf_event_ = netatmo_in_vals_df.loc[
                                    event_date, netatmo_stn_id]
                                if netatmo_edf_event_ >= 1:
                                    # print('Correcting Netatmo station',
                                    # netatmo_stn_id)
                                    try:

                                        #======================================
                                        # # Correct Netatmo Quantiles
                                        #======================================

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

#                                         print('\nOriginal Netatmo Ppt: ', netatmo_ppt_event_,
#                                               '\nOriginal Netatmo Edf: ', netatmo_edf_event_)

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
                                                    print(msg, 'DWDWD')
                                                    raise Exception

                                                try:
                                                    if ppt_for_edf >= 0:
                                                        ppt_vals_dwd_new_for_obsv_ppt.append(
                                                            ppt_for_edf)
                                                        stn_xcoords_new = dwd_in_coords_df.loc[
                                                            dwd_stn_id_new, 'X']
                                                        stn_ycoords_new = dwd_in_coords_df.loc[
                                                            dwd_stn_id_new, 'Y']

                                                        dwd_xcoords_new.append(
                                                            stn_xcoords_new)
                                                        dwd_ycoords_new.append(
                                                            stn_xcoords_new)
                                                except Exception as msg:
                                                    print(msg, 'EEWEWE')
                                                    raise Exception
                                    except Exception:
                                        print('EROOR')


# stop = timeit.default_timer()  # Ending time
# print('\n\a\a\a Done with everything on %s \a\a\a' %
#       (time.asctime()))
