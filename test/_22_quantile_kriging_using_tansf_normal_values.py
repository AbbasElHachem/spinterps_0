# -*- coding: utf-8 -*-
"""
Created on 01.12.2019

@author: Abbas EL Hachem

Goal: Quantile Kriging
"""
import os
os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import pyximport

pyximport.install()


import timeit
import time
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import spatial
from scipy.stats import norm
from scipy.stats import rankdata

from pathlib import Path
from spinterps import (OrdinaryKriging,
                       OrdinaryKrigingWithUncertainty)
from spinterps import variograms


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

# path to transformed data, used for kriging
path_to_transformed_data = in_filter_path / r'normally_transfomed_values'
#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================

use_dwd_stns_for_kriging = True

qunatile_kriging = True

# run it to filter True
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
    path_to_netatmo_gd_stns = (
        main_dir / r'plots_NetAtmo_ppt_DWD_ppt_correlation_' /
        (r'keep_stns_all_neighbor_99_per_60min_s0_%s.csv'
         % _acc_))


#==============================================================================
#
#==============================================================================
resample_frequencies = ['1440min']
# '60min', '180min', '360min', '720min', '1440min'


title_ = r'Qt_ok_ok_un_3_test'


if not use_netatmo_gd_stns:
    title_ = title_ + '_netatmo_no_flt_'

    path_to_folder_wtih_transf_data = (path_to_transformed_data /
                                       r'transformed_values_no_flt')

if use_netatmo_gd_stns:
    title_ = title_ + '_first_flt_'

    path_to_folder_wtih_transf_data = (path_to_transformed_data /
                                       (r'transformed_values_first_flt__%s' % _acc_))

if use_temporal_filter_after_kriging:

    title_ = title_ + '_temp_flt_'
    path_to_folder_wtih_transf_data = (path_to_transformed_data /
                                       (r'transformed_values_first_flt_temp_flt__%s' % _acc_))

#==============================================================================
# path to folder with transformed data
#==============================================================================

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
max_dist_neighbrs = 500e4
diff_thr = 0.1

# xvals = np.linspace(0.1, 10, 1000, endpoint=True)

#==============================================================================
# Needed functions
#==============================================================================


def list_all_full_path(ext, file_dir):
    """
    Purpose: To return full path of files in all dirs of a given folder with a
    -------  given extension in ascending order.

    Keyword arguments:
    ------------------
        ext (string) = Extension of the files to list
            e.g. '.txt', '.tif'.
        file_dir (string) = Full path of the folder in which the files
            reside.
    """
    new_list = []
    patt = '*' + ext
    for root, _, files in os.walk(file_dir):
        for elm in files:
            if fnmatch.fnmatch(elm, patt):
                full_path = os.path.join(root, elm)
                new_list.append(full_path)
    return(sorted(new_list))
# =============================================================================


def build_edf_fr_vals(ppt_data):
    """ construct empirical distribution function given data values """
    data_sorted = np.sort(ppt_data, axis=0)[::-1]

    x0 = np.round(np.squeeze(data_sorted)[::-1], 3)
    y0 = np.round((np.arange(data_sorted.size) / len(data_sorted)), 6)

    return x0, y0

# =============================================================================


def find_nearest(array, value):
    '''given an array, find in the array 
        the nearest value to a given value'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# =============================================================================


def test_if_vg_model_is_suitable(vg_model, df_vgs, event_date):
    '''check if vg model is suitable'''
    if not isinstance(vg_model, str):
        vg_model = df_vgs.loc[event_date, 2]

    if not isinstance(vg_model, str):
        vg_model = ''

    if ('Nug' in vg_model or len(
        vg_model) == 0) and (
        'Exp' not in vg_model and
            'Sph' not in vg_model):
        try:
            for i in range(2, len(df_vgs.loc[event_date, :])):
                vg_model = df_vgs.loc[event_date, i]
                if type(vg_model) == np.float:
                    continue
                if ('Nug' in vg_model
                        or len(vg_model) == 0) and (
                            'Exp' not in vg_model or
                        'Sph' not in vg_model):
                    continue
                else:
                    break

        except Exception as msg:
            print(msg)
            print('Only Nugget variogram for this day')
    if not isinstance(vg_model, str):
        vg_model = ''
    return vg_model

# =============================================================================


def sum_calc_phi_nks(mean_val, std_val):
    xvals = np.linspace(
        -(mean_val + 5 * std_val),
        +(mean_val + 5 * std_val),
        1000, endpoint=True)

    phi_densities = [norm.pdf(x, loc=mean_val,
                              scale=std_val)[0]
                     for x in xvals]

    phi_densities_sr = pd.Series(index=xvals, data=phi_densities)
    phi_densities_sr.plot()

    # plt.show()
    sum_densities = np.sum(phi_densities)
    #print('sum_densities: ', sum_densities)
    return sum_densities

# =============================================================================


def calc_sum_inv_F_norm_nk_times_phi_k(mean_val, std_val,
                                       ppt_obsv, edf_obsv):

    xvals = np.linspace(
        -(mean_val + 5 * std_val),
        +(mean_val + 5 * std_val),
        1000, endpoint=True)

    sum_densities = sum_calc_phi_nks(mean_val, std_val)

    sum_weighted_ppts = []

#     phi_densities = []
    inv_obs_std_norm = []
    ppt_nks = []

    for x in xvals:

        density_nk = norm.pdf(
            x,
            loc=mean_val,
            scale=std_val)[0]

        weight_ppt = density_nk / sum_densities

        std_normal_inv_nk = norm.cdf(
            x,
            loc=0,
            scale=1)  # [0]
        # Z score transformation to std normal
#         std_normal_inv_nk = (x-mean_val) / std_val
        # phi_densities.append(density_nk)

        inv_obs_std_norm.append(std_normal_inv_nk)

        ppt_nk = ppt_obsv[np.where(
            edf_obsv == find_nearest(
                edf_obsv, std_normal_inv_nk))][0]

        weighted_ppt = ppt_nk * weight_ppt
#         if weighted_ppt >
#             print(weighted_ppt)
        ppt_nks.append(weighted_ppt)

        sum_weighted_ppts.append(
            weighted_ppt)

    sum_densities_sr = pd.DataFrame(index=inv_obs_std_norm)
    sum_densities_sr['sum_weighted_ppts'] = sum_weighted_ppts
#     sum_densities_sr['phi_nk'] = phi_densities
#     sum_densities_sr['inv_obs_std_norm'] = inv_obs_std_norm
#     sum_densities_sr['ppt_nks'] = ppt_nks
#     onsc_edf = pd.Series(index=edf_obsv, data=ppt_obsv)
#     onsc_edf.plot()
    # TODO:  check again why phi nk small

#     sum_densities_sr.plot()
    # plt.show()

    sum_ppts = np.sum(
        sum_weighted_ppts)

    return sum_ppts

#==============================================================================
#
#==============================================================================


def cal_inv_ppt_rosenbluth_mtd(n_star, std_star, ppt_obsv, edf_obsv):

    ppt_nk2 = ppt_obsv[np.where(
        edf_obsv == find_nearest(
            edf_obsv, norm.cdf(
                n_star + std_star,
                loc=0,
                scale=1)[0]))][0]
    ppt_nk1 = ppt_obsv[np.where(
        edf_obsv == find_nearest(
            edf_obsv, norm.cdf(
                n_star - std_star,
                loc=0,
                scale=1)[0]))][0]
    ppt_mean = (ppt_nk1 + ppt_nk2) / 2
    return ppt_mean

#==============================================================================
#
#==============================================================================


def qt_cal_inv_ppt_rosenbluth_mtd(n_star, std_star, ppt_obsv, edf_obsv):
    ppt_nk2 = ppt_obsv[np.where(
        edf_obsv == find_nearest(
            edf_obsv, n_star + std_star))][0]
    ppt_nk1 = ppt_obsv[np.where(
        edf_obsv == find_nearest(
            edf_obsv, n_star - std_star))][0]
    ppt_mean = (ppt_nk1 + ppt_nk2) / 2
#     ppt_mean = ppt_nk2
    return ppt_mean


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
#
#==============================================================================
# read distance matrix dwd-netamot ppt
in_df_distance_netatmo_dwd = pd.read_csv(
    distance_matrix_netatmo_dwd_df_file, sep=';', index_col=0)

# read df combinations to use
df_dwd_stns_comb = pd.read_csv(
    path_to_dwd_stns_comb, index_col=0,
    sep=',', dtype=str)

# get all transformed files for this filter combination

assert os.path.exists(path_to_folder_wtih_transf_data)
all_transf_data_files = list_all_full_path(
    '.csv', path_to_folder_wtih_transf_data)
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

    # transformed normal data, dwd; dwd_netatmo
    path_transformed_dwd_vals = Path([pth for pth in all_transf_data_files if
                                      'dwd_' + temp_agg in pth][0])

    path_transformed_dwd_netatmo_vals = Path([pth for pth in all_transf_data_files
                                              if 'dwd_netatmo_' + temp_agg in pth][0])
    # Files to use
    #==========================================================================

    netatmo_data_to_use = path_to_netatmo_edf
    dwd_data_to_use = path_to_dwd_edf

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

    # dwd transformed normal data
    dwd_in_transf_data = pd.read_csv(
        path_transformed_dwd_vals, sep=';', index_col=0, encoding='utf-8',
        parse_dates=True, infer_datetime_format=True)

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

    # ppt data
    netatmo_in_ppt_vals_df = pd.read_csv(
        path_to_netatmo_ppt, sep=';',
        index_col=0,
        encoding='utf-8', engine='c')

    netatmo_in_ppt_vals_df.index = pd.to_datetime(
        netatmo_in_ppt_vals_df.index, format='%Y-%m-%d')

    netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.loc[strt_date:end_date, :]
    netatmo_in_ppt_vals_df.dropna(how='all', axis=0, inplace=True)

    # netatmo-dwd transformed values
    dwd_netatmo_in_transf_data = pd.read_csv(
        path_transformed_dwd_netatmo_vals, sep=';', index_col=0, encoding='utf-8',
        parse_dates=True, infer_datetime_format=True)
    # apply first filter
    if use_netatmo_gd_stns:
        print('\n**using Netatmo gd stns**')
        good_netatmo_stns = df_gd_stns.loc[:, 'Stations'].values.ravel()
        cmn_gd_stns = netatmo_in_vals_df.columns.intersection(
            good_netatmo_stns)
        netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_gd_stns]

    cmn_stns = netatmo_in_vals_df.columns.intersection(
        netatmo_in_coords_df.index)

    netatmo_in_coords_df = netatmo_in_coords_df.loc[
        netatmo_in_coords_df.index.intersection(cmn_stns), :]
    netatmo_in_vals_df = netatmo_in_vals_df.loc[
        :, netatmo_in_vals_df.columns.intersection(cmn_stns)]
    netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.loc[:, cmn_stns]

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
                dwd_netatmo_in_transf_data.index).intersection(
                df_vgs_transf.index), :]

    cmn_netatmo_stns = in_df_distance_netatmo_dwd.index.intersection(
        netatmo_in_vals_df.columns)
    in_df_distance_netatmo_dwd = in_df_distance_netatmo_dwd.loc[
        cmn_netatmo_stns, :]

    # DWD, Netatmo-DWD station names
    dwd_stns_names = dwd_in_transf_data.columns.to_list()
    dwd_netatmo_stns_names = dwd_netatmo_in_transf_data.columns.to_list()
    #==========================================================================
    # CREATE DFS HOLD RESULT KRIGING PER NETATMO STATION
    #==========================================================================

    for idx_lst_comb in df_dwd_stns_comb.index:

        stn_comb = [stn.replace("'", "")
                    for stn in df_dwd_stns_comb.iloc[
                        idx_lst_comb, :].dropna().values]

        # create DFs for saving results
        df_interpolated_dwd_netatmos_comb = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_netatmos_comb_un = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_only = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        # for direkt quantile interpolation
        df_interpolated_dwd_netatmos_comb_qt = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_only_qt = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        # for direkt quantile interpolation rosenbluth
        df_interpolated_dwd_netatmos_comb_qt_ros = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_dwd_only_qt_ros = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        # remove the 10 stations from DWD transformed vals
        dwd_stns_names_rem = [stn for stn in dwd_stns_names
                              if stn not in stn_comb]
        dwd_netatmo_stns_names_rem = [stn for stn in dwd_netatmo_stns_names
                                      if stn not in stn_comb]

        dwd_in_transf_data_rem = dwd_in_transf_data.loc[:, dwd_stns_names_rem]
        dwd_netatmo_in_transf_data_rem = dwd_netatmo_in_transf_data.loc[
            :, dwd_netatmo_stns_names_rem]

        #======================================================================
        # START KRIGING EVENTS
        #======================================================================
        for ievt, event_date in enumerate(dwd_in_extremes_df.index[50:]):

            _stn_id_event_ = str(int(dwd_in_extremes_df.loc[event_date, 2]))
            if len(_stn_id_event_) < 5:
                _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                    '0' + _stn_id_event_

            _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
            _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]

            # transf dwd stations and data  for event
            trans_dwd_vals_evt = dwd_in_transf_data_rem.loc[
                event_date, :].dropna()

            # transf dwd-netatmo stations and data for event
            trans_dwd_netatmo_vals_evt = dwd_netatmo_in_transf_data_rem.loc[
                event_date, :].dropna()

            netatmo_stns_names_evt = [
                stn for stn in trans_dwd_netatmo_vals_evt.index
                if stn not in trans_dwd_vals_evt.index]

            if len(netatmo_stns_names_evt) > 0:
                for i, stn_dwd_id in enumerate(stn_comb[2:3]):

                    # interpolate loc
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
                        _edf_event_stn = dwd_in_vals_df.loc[
                            event_date, stn_dwd_id]
#                         if _ppt_event_stn > 30:
#                             print('enter debug')
#                             pass
                        trans_edf_event_stn = dwd_in_transf_data.loc[
                            event_date, stn_dwd_id]

                        # print('\n** Obersved Transf Quantile at DWD Stn: ',
                        #      trans_edf_event_stn)

                        # build edf
                        try:
                            xppt, yppt = build_edf_fr_vals(ppt_stn_dwd)
                        except Exception as msg:
                            print('Error getting cdf', msg)
                            continue

                        # coords for interpolation
                        x_dwd_coords = dwd_in_coords_df.loc[
                            trans_dwd_vals_evt.index, 'X'].values.ravel()
                        y_dwd_coords = dwd_in_coords_df.loc[
                            trans_dwd_vals_evt.index, 'Y'].values.ravel()

                        x_netatmo_coords = netatmo_in_coords_df.loc[
                            netatmo_stns_names_evt, 'X'].values.ravel()
                        y_netatmo_coords = netatmo_in_coords_df.loc[
                            netatmo_stns_names_evt, 'Y'].values.ravel()
                        print('\n*+Netatmo stns:', len(netatmo_stns_names_evt))
                        # combine coordinates of DWD and Netatmo
                        dwd_netatmo_xcoords = np.concatenate(
                            [x_dwd_coords, x_netatmo_coords])
                        dwd_netatmo_ycoords = np.concatenate(
                            [y_dwd_coords, y_netatmo_coords])

                        # dwd data
                        dwd_vals = trans_dwd_vals_evt.values.ravel()
                        dwd_netatmo_vals = trans_dwd_netatmo_vals_evt.values.ravel()
                        #
                        # quantiles observed
                        dwd_qts = dwd_in_vals_df.loc[
                            event_date,
                            dwd_stns_names_rem].dropna().values.ravel()
                        netatmo_qts = netatmo_in_vals_df.loc[
                            event_date,
                            netatmo_stns_names_evt].dropna().values.ravel()
                        dwd_netatmo_qts = np.concatenate(
                            [dwd_qts, netatmo_qts])
                        # get VG model from transformed DWD values
                        vgs_model_dwd_transf = df_vgs_transf.loc[event_date, 1]

                        #==============================================
                        # KRIGING
                        #==============================================
    #                             print('\n+++ KRIGING +++\n')
                        ordinary_kriging_dwd_netatmo_comb = OrdinaryKriging(
                            xi=dwd_netatmo_xcoords,
                            yi=dwd_netatmo_ycoords,
                            zi=dwd_netatmo_vals,
                            xk=x_dwd_interpolate,
                            yk=y_dwd_interpolate,
                            model=vgs_model_dwd_transf)

                        # qts
                        ordinary_kriging_dwd_netatmo_comb_qt = OrdinaryKriging(
                            xi=dwd_netatmo_xcoords,
                            yi=dwd_netatmo_ycoords,
                            zi=dwd_netatmo_qts,
                            xk=x_dwd_interpolate,
                            yk=y_dwd_interpolate,
                            model=vgs_model_dwd_transf)
    #                     ordinary_kriging_un_dwd_netatmo_comb = OrdinaryKrigingWithUncertainty(
    #                         xi=dwd_netatmo_xcoords,
    #                         yi=dwd_netatmo_ycoords,
    #                         zi=std_norm_edf_dwd_netatmo_vals,
    #                         uncert=edf_dwd_netatmo_vals_uncert,
    #                         xk=x_dwd_interpolate,
    #                         yk=y_dwd_interpolate,
    #                         model=vgs_model_dwd_transf)

                        ordinary_kriging_dwd_only = OrdinaryKriging(
                            xi=x_dwd_coords,
                            yi=y_dwd_coords,
                            zi=dwd_vals,
                            xk=x_dwd_interpolate,
                            yk=y_dwd_interpolate,
                            model=vgs_model_dwd_transf)

                        # qts
                        ordinary_kriging_dwd_only_qt = OrdinaryKriging(
                            xi=x_dwd_coords,
                            yi=y_dwd_coords,
                            zi=dwd_qts,
                            xk=x_dwd_interpolate,
                            yk=y_dwd_interpolate,
                            model=vgs_model_dwd_transf)
                        try:
                            ordinary_kriging_dwd_netatmo_comb.krige()
    #                         ordinary_kriging_un_dwd_netatmo_comb.krige()
                            ordinary_kriging_dwd_only.krige()

                            ordinary_kriging_dwd_netatmo_comb_qt.krige()
    #                         ordinary_kriging_un_dwd_netatmo_comb.krige()
                            ordinary_kriging_dwd_only_qt.krige()
                        except Exception as msg:
                            print('Error while Kriging 4', msg)
                            continue
                        # get interpolated values
                        interpolated_vals_dwd_netatmo = ordinary_kriging_dwd_netatmo_comb.zk.copy()
    #                     interpolated_vals_dwd_netatmo_un = ordinary_kriging_un_dwd_netatmo_comb.zk.copy()
                        interpolated_vals_dwd_only = ordinary_kriging_dwd_only.zk.copy()

                        # qts
                        interpolated_vals_dwd_netatmo_qt = ordinary_kriging_dwd_netatmo_comb_qt.zk.copy()
                        interpolated_vals_dwd_only_qt = ordinary_kriging_dwd_only_qt.zk.copy()

                        std_est_vals_dwd_netatmo_qt = np.sqrt(
                            ordinary_kriging_dwd_netatmo_comb_qt.est_vars)

                        std_est_vals_dwd_only_qt = np.sqrt(
                            ordinary_kriging_dwd_only_qt.est_vars)
                        # qt +- std
                        ppt_netatmo_dwd_ros = qt_cal_inv_ppt_rosenbluth_mtd(
                            n_star=interpolated_vals_dwd_netatmo_qt,
                            std_star=std_est_vals_dwd_netatmo_qt,
                            ppt_obsv=xppt,
                            edf_obsv=yppt)

                        ppt_dwd_ros = qt_cal_inv_ppt_rosenbluth_mtd(
                            n_star=interpolated_vals_dwd_only_qt,
                            std_star=std_est_vals_dwd_only_qt,
                            ppt_obsv=xppt,
                            edf_obsv=yppt)

                        # without std, direkt backtransformation
                        ppt_nk2 = xppt[np.where(
                            yppt == find_nearest(
                                yppt, interpolated_vals_dwd_netatmo_qt))][0]

                        ppt_nk3 = xppt[np.where(
                            yppt == find_nearest(
                                yppt, interpolated_vals_dwd_only_qt))][0]

#                         print('\n**Interpolated Transf DWD: ',
#                               interpolated_vals_dwd_only,
#                               '\n**Interpolated Transf DWD-Netatmo: ',
#                               interpolated_vals_dwd_netatmo)

                        # calcualte standard deviation of estimated
                        # values
                        std_est_vals_dwd_netatmo = np.sqrt(
                            ordinary_kriging_dwd_netatmo_comb.est_vars)

    #                     std_est_vals_dwd_netatmo_un = np.sqrt(
    #                         ordinary_kriging_un_dwd_netatmo_comb.est_vars)

                        std_est_vals_dwd_only = np.sqrt(
                            ordinary_kriging_dwd_only.est_vars)
    #
                        if (- np.inf < interpolated_vals_dwd_netatmo < np.inf) and (
                            #                             - np.inf < interpolated_vals_dwd_netatmo_un < np.inf) and (
                                - np.inf < interpolated_vals_dwd_only < np.inf):

                            #==========================================
                            # # backtransform everything to ppt
                            #==========================================

                            # discritize distribution
                            # TODO: CONTINUE HERE

                            # sum phi nks
                            sum_densities_netatmo_dwd = sum_calc_phi_nks(
                                mean_val=interpolated_vals_dwd_netatmo,
                                std_val=std_est_vals_dwd_netatmo)

    #                         sum_densities_netatmo_dwd_unc = sum_calc_phi_nks(
    #                             mean_val=interpolated_vals_dwd_netatmo_un,
    #                             std_val=std_est_vals_dwd_netatmo_un)

                            sum_densities_dwd_only = sum_calc_phi_nks(
                                mean_val=interpolated_vals_dwd_only,
                                std_val=std_est_vals_dwd_only)

                            # sum F^-1obsv(Phi(nk))*phi(nk)
                            sum_ppts_netatmo_dwd = calc_sum_inv_F_norm_nk_times_phi_k(
                                mean_val=interpolated_vals_dwd_netatmo,
                                std_val=std_est_vals_dwd_netatmo,
                                ppt_obsv=xppt,
                                edf_obsv=yppt)

    #                         sum_ppts_netatmo_dwd_unc = calc_sum_inv_F_norm_nk_times_phi_k(
    #                             mean_val=interpolated_vals_dwd_netatmo_un,
    #                             std_val=std_est_vals_dwd_netatmo_un,
    #                             ppt_obsv=xppt,
    #                             edf_obsv=yppt)

                            sum_ppts_dwd_only = calc_sum_inv_F_norm_nk_times_phi_k(
                                mean_val=interpolated_vals_dwd_only,
                                std_val=std_est_vals_dwd_only,
                                ppt_obsv=xppt,
                                edf_obsv=yppt)

                            interpolated_ppt_dwd_netatmo = (
                                sum_ppts_netatmo_dwd)

#                             interpolated_ppt_dwd_netatmo_un = (
#                                 sum_ppts_netatmo_dwd_unc /
#                                 sum_densities_netatmo_dwd_unc)

                            interpolated_ppt_dwd_only = (
                                sum_ppts_dwd_only)  # / sum_densities_dwd_only)
                            # another method
#                             ppt_dwd_ros2 = cal_inv_ppt_rosenbluth_mtd(
#                                 n_star=interpolated_vals_dwd_only,
#                                 std_star=std_est_vals_dwd_only,
#                                 ppt_obsv=xppt,
#                                 edf_obsv=yppt)
#                             ppt_netatmo_dwd_ros2 = cal_inv_ppt_rosenbluth_mtd(
#                                 n_star=interpolated_vals_dwd_netatmo,
#                                 std_star=std_est_vals_dwd_netatmo,
#                                 ppt_obsv=xppt,
#                                 edf_obsv=yppt)

                            print('\n** Obersved Rainfall at DWD Stn: ',
                                  _ppt_event_stn)

                            print('\n**Interpolated DWD: ',
                                  interpolated_ppt_dwd_only,
                                  '\n**Interpolated DWD-Netatmo: ',
                                  interpolated_ppt_dwd_netatmo)

#                             print('\n**Interpolated DWD 2: ',
#                                   ppt_dwd_ros2,
#                                   '\n**Interpolated DWD-Netatmo 2: ',
#                                   ppt_netatmo_dwd_ros2)

#                             print('\n**Interpolated DWD 3: ',
#                                   ppt_dwd_ros,
#                                   '\n**Interpolated DWD-Netatmo 3: ',
#                                   ppt_netatmo_dwd_ros)

                            print('\n**Interpolated DWD 4: ',
                                  ppt_nk3,
                                  '\n**Interpolated DWD-Netatmo 4: ',
                                  ppt_nk2)

#                                   '\n**Interpolated DWD-Netatmo Un: ',
#                                   interpolated_ppt_dwd_netatmo_un)

                            # Saving result to DF
                            df_interpolated_dwd_netatmos_comb.loc[
                                event_date,
                                stn_dwd_id] = interpolated_ppt_dwd_netatmo

#                             df_interpolated_dwd_netatmos_comb_un.loc[
#                                 event_date,
#                                 stn_dwd_id] = interpolated_ppt_dwd_netatmo_un

                            df_interpolated_dwd_only.loc[
                                event_date,
                                stn_dwd_id] = interpolated_ppt_dwd_only

                            # qts direkt back
                            df_interpolated_dwd_netatmos_comb_qt.loc[
                                event_date,
                                stn_dwd_id] = ppt_nk2
                            df_interpolated_dwd_only_qt.loc[
                                event_date,
                                stn_dwd_id] = ppt_nk3

                            # qts rosenbluth
                            df_interpolated_dwd_netatmos_comb_qt_ros.loc[
                                event_date,
                                stn_dwd_id] = ppt_netatmo_dwd_ros
                            df_interpolated_dwd_only_qt_ros.loc[
                                event_date,
                                stn_dwd_id] = ppt_dwd_ros

        print('Done for this stn')

#
#==============================================================================
# #         # drop nans from df and save results
#==============================================================================
        df_interpolated_dwd_netatmos_comb.dropna(how='all', inplace=True)
#         df_interpolated_dwd_netatmos_comb_un.dropna(how='all', inplace=True)
        df_interpolated_dwd_only.dropna(how='all', inplace=True)

        df_interpolated_dwd_netatmos_comb_qt.dropna(how='all', inplace=True)
#         df_interpolated_dwd_netatmos_comb_un.dropna(how='all', inplace=True)
        df_interpolated_dwd_only_qt.dropna(how='all', inplace=True)

        df_interpolated_dwd_netatmos_comb_qt_ros.dropna(
            how='all', inplace=True)
#         df_interpolated_dwd_netatmos_comb_un.dropna(how='all', inplace=True)
        df_interpolated_dwd_only_qt_ros.dropna(how='all', inplace=True)

#==============================================================================
#
#==============================================================================
#         df_interpolated_dwd_only.loc[:, stn_dwd_id].plot()
        df_interpolated_dwd_netatmos_comb.to_csv(out_plots_path / (
            'interpolated_quantiles_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
            % (temp_agg, title_, idx_lst_comb, _acc_)),
            sep=';', float_format='%0.2f')
#
#         df_interpolated_dwd_netatmos_comb_un.to_csv(out_plots_path / (
#             'interpolated_quantiles_un_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
#             % (temp_agg, title_, idx_lst_comb, _acc_)),
#             sep=';', float_format='%0.2f')
#
        df_interpolated_dwd_only.to_csv(out_plots_path / (
            'interpolated_quantiles_dwd_%s_data_%s_using_dwd_only_grp_%d_%s.csv'
            % (temp_agg, title_, idx_lst_comb, _acc_)),
            sep=';', float_format='%0.2f')
        #======================================================================
        # direkt QT interpolation
        #======================================================================
        df_interpolated_dwd_netatmos_comb_qt.to_csv(out_plots_path / (
            'dk_interpolated_quantiles_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
            % (temp_agg, title_, idx_lst_comb, _acc_)),
            sep=';', float_format='%0.2f')
#
#         df_interpolated_dwd_netatmos_comb_un.to_csv(out_plots_path / (
#             'interpolated_quantiles_un_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
#             % (temp_agg, title_, idx_lst_comb, _acc_)),
#             sep=';', float_format='%0.2f')
#
        df_interpolated_dwd_only_qt.to_csv(out_plots_path / (
            'dk_interpolated_quantiles_dwd_%s_data_%s_using_dwd_only_grp_%d_%s.csv'
            % (temp_agg, title_, idx_lst_comb, _acc_)),
            sep=';', float_format='%0.2f')
        #======================================================================
        # direkt Rosenbluth method
        #======================================================================
        df_interpolated_dwd_netatmos_comb_qt_ros.to_csv(out_plots_path / (
            'dk_ros_interpolated_quantiles_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
            % (temp_agg, title_, idx_lst_comb, _acc_)),
            sep=';', float_format='%0.2f')
#
#         df_interpolated_dwd_netatmos_comb_un.to_csv(out_plots_path / (
#             'interpolated_quantiles_un_dwd_%s_data_%s_using_dwd_netamo_grp_%d_%s.csv'
#             % (temp_agg, title_, idx_lst_comb, _acc_)),
#             sep=';', float_format='%0.2f')
#
        df_interpolated_dwd_only_qt_ros.to_csv(out_plots_path / (
            'dkros__interpolated_quantiles_dwd_%s_data_%s_using_dwd_only_grp_%d_%s.csv'
            % (temp_agg, title_, idx_lst_comb, _acc_)),
            sep=';', float_format='%0.2f')

#
#
# stop = timeit.default_timer()  # Ending time
# print('\n\a\a\a Done with everything on %s \a\a\a' %
#       (time.asctime()))
