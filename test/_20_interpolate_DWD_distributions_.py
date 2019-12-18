# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    Interpolate DWD distributions
Purpose: using DWD neighboring stations interpolate DWD distributions

Created on: 2019-12-16

Using the neighboring DWD stations interpolate the DWD distribution for every
DWD station. Do it both ways:
    1) using rainfall data interpolate quantiles
    2) using quantile interpolate rainfall
    3) using qunatiles interpolate qunatiles

Parameters
----------

Input Files
    DWD station data
    DWD coordinates data
    
Returns
-------

    
Df_distributions: df containing for every DWD station, the interpolated
    DWD distributions

"""


__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

import os
os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import pyximport

pyximport.install()


import timeit
import time

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


# path for data filter
in_filter_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'


# path_to_dwd_stns_comb
path_to_dwd_stns_comb = in_filter_path / r'dwd_combination_to_use.csv'

# read df combinations to use
df_dwd_stns_comb = pd.read_csv(
    path_to_dwd_stns_comb, index_col=0,
    sep=',', dtype=str)
#==============================================================================
resample_frequencies = ['1440min']
# '60min', '180min', '360min', '720min', '1440min'


title_ = r'interpolate_DWD_distributions'

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

list_percentiles = np.round(np.arange(0.5, 1.00001, 0.001), 10)
# print(list_percentiles)
#==============================================================================
# Needed functions
#==============================================================================


def calculate_probab_ppt_below_thr(ppt_data, ppt_thr):
    """ calculate probability of values being below threshold """
    origin_count = ppt_data.shape[0]
    count_below_thr = ppt_data[ppt_data <= ppt_thr].shape[0]
    p0 = np.divide(count_below_thr, origin_count)
    return p0

#==============================================================================
#
#==============================================================================


def build_edf_fr_vals(ppt_data):
    """ construct empirical distribution function given data values """
    data_sorted = np.sort(ppt_data, axis=0)[::-1]

    x0 = np.round(np.squeeze(data_sorted)[::-1], 2)
    y0 = np.round((np.arange(data_sorted.size) / len(data_sorted)), 5)

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
    return vg_model


#==============================================================================
#
#==============================================================================
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

# read df combinations to use
df_dwd_stns_comb = pd.read_csv(
    path_to_dwd_stns_comb, index_col=0,
    sep=',', dtype=str)

#==============================================================================
#
#==============================================================================
for temp_agg in resample_frequencies:

    # out path directory
    dir_path = title_ + '_' + temp_agg

    out_plots_path = in_filter_path / dir_path

    if not os.path.exists(out_plots_path):
        os.mkdir(out_plots_path)
    print(out_plots_path)

    # path to data
    #=========================================================================
    out_save_csv = 'dwd_%s_ppt_edf_' % temp_agg

    path_to_dwd_edf = (path_to_data /
                       (r'edf_ppt_all_dwd_%s_.csv' % temp_agg))

    path_to_dwd_ppt = (path_to_data /
                       (r'ppt_all_dwd_%s_.csv' % temp_agg))

    # Files to use
    #==========================================================================

    dwd_data_to_use = path_to_dwd_edf

    print(title_)
    # DWD DATA
    #=========================================================================

    # DWD ppt
    dwd_in_ppt_vals_df = pd.read_csv(
        path_to_dwd_ppt, sep=';', index_col=0, encoding='utf-8')

    dwd_in_ppt_vals_df.index = pd.to_datetime(
        dwd_in_ppt_vals_df.index, format='%Y-%m-%d')

    dwd_in_ppt_vals_df = dwd_in_ppt_vals_df.loc[strt_date:end_date, :]
    dwd_in_ppt_vals_df.dropna(how='all', axis=0, inplace=True)

    for ix, idx_lst_comb in enumerate(df_dwd_stns_comb.index):

        stn_comb = [stn.replace("'", "")
                    for stn in df_dwd_stns_comb.iloc[
                        idx_lst_comb, :].dropna().values]

        # create DFs for saving results
        df_interpolated_dwd_only = pd.DataFrame(
            index=list_percentiles,
            columns=[stn_comb])

        for stn_dwd_id in stn_comb:

            print('interpolating for DWD Station', stn_dwd_id)
            x_dwd_interpolate = np.array(
                [dwd_in_coords_df.loc[stn_dwd_id, 'X']])
            y_dwd_interpolate = np.array(
                [dwd_in_coords_df.loc[stn_dwd_id, 'Y']])

            # drop stn
            # drop stn from group
            all_dwd_stns_except_interp_loc = [
                stn for stn in dwd_in_ppt_vals_df.columns
                if stn not in stn_comb]

            ppt_data_stn = dwd_in_ppt_vals_df.loc[:, stn_dwd_id].dropna(
                how='all')

            p0_stn = calculate_probab_ppt_below_thr(ppt_data_stn, 0.1)

            for _cdf_percentile_ in list_percentiles:

                #_cdf_percentile_ = np.round(_cdf_percentile_, 3)

                print('**Calculating for percentile: ',
                      _cdf_percentile_, ' **\n')

                if _cdf_percentile_ <= p0_stn:
                    df_interpolated_dwd_only.loc[
                        _cdf_percentile_, stn_dwd_id] = p0_stn / 2

                else:
                    # DWD qunatiles
                    ppt_dwd_vals = []
                    dwd_xcoords = []
                    dwd_ycoords = []
                    dwd_stn_ids = []

                    for stn_id in all_dwd_stns_except_interp_loc:
                        # print('station is', stn_id)
                        stn_data_df = dwd_in_ppt_vals_df.loc[:, stn_id].dropna(
                            how='all')
                        if stn_data_df.size > 10:
                            ppt_stn, edf_stn = build_edf_fr_vals(
                                stn_data_df.values)

                            # get nearest percentile in dwd
                            nearst_edf_ = find_nearest(
                                edf_stn,
                                _cdf_percentile_)

                            ppt_idx = np.where(edf_stn == nearst_edf_)

                            try:
                                if ppt_idx[0].size > 1:
                                    ppt_percentile = np.mean(ppt_stn[ppt_idx])
                                else:
                                    ppt_percentile = ppt_stn[ppt_idx]
                            except Exception as msg:
                                print(msg)

                            if ppt_percentile >= 0:
                                ppt_dwd_vals.append(
                                    np.unique(ppt_percentile)[0])
                                dwd_xcoords.append(
                                    dwd_in_coords_df.loc[stn_id, 'X'])
                                dwd_ycoords.append(
                                    dwd_in_coords_df.loc[stn_id, 'Y'])
                                dwd_stn_ids.append(stn_id)

                    dwd_xcoords = np.array(dwd_xcoords)
                    dwd_ycoords = np.array(dwd_ycoords)
                    ppt_dwd_vals = np.array(ppt_dwd_vals)

                    try:

                        vg_dwd = VG(
                            x=dwd_xcoords,
                            y=dwd_ycoords,
                            z=ppt_dwd_vals,
                            mdr=mdr,
                            nk=5,
                            typ='cnst',
                            perm_r_list=perm_r_list_,
                            fil_nug_vg=fil_nug_vg,
                            ld=None,
                            uh=None,
                            h_itrs=100,
                            opt_meth='L-BFGS-B',
                            opt_iters=1000,
                            fit_vgs=fit_vgs,
                            n_best=n_best,
                            evg_name='robust',
                            use_wts=False,
                            ngp=ngp,
                            fit_thresh=0.01)

                        vg_dwd.fit()

                        fit_vg_list = vg_dwd.vg_str_list

                    except Exception as msg:
                        print(msg)
                        fit_vg_list = ['']
                        # continue

                    vgs_model_dwd = fit_vg_list[0]

                    if not isinstance(vgs_model_dwd, str):
                        vgs_model_dwd = ''

                    if ('Nug' in vgs_model_dwd or len(
                        vgs_model_dwd) == 0) and (
                        'Exp' not in vgs_model_dwd and
                            'Sph' not in vgs_model_dwd):
                        print('**Variogram %s not valid -->'
                              '\nlooking for alternative\n**'
                              % vgs_model_dwd)
                        try:
                            for i in range(1, len(fit_vg_list)):
                                vgs_model_dwd = fit_vg_list[i]
                                if type(vgs_model_dwd) == np.float:
                                    continue
                                if ('Nug' in vgs_model_dwd
                                        or len(vgs_model_dwd) == 0) and (
                                            'Exp' not in vgs_model_dwd and
                                        'Sph' not in vgs_model_dwd):
                                    continue
                                else:
                                    break

                        except Exception as msg:
                            print(msg)
                            print('Only Nugget variogram for this day')

                    if ('Nug' in vgs_model_dwd
                        or len(vgs_model_dwd) > 0) and (
                        'Exp' in vgs_model_dwd or
                            'Sph' in vgs_model_dwd):

                        #                 print('**Changed Variogram model to**\n', vgs_model_dwd)
                        #                 print('+++ KRIGING +++\n')

                        ordinary_kriging_dwd_only = OrdinaryKriging(
                            xi=dwd_xcoords,
                            yi=dwd_ycoords,
                            zi=ppt_dwd_vals,
                            xk=x_dwd_interpolate,
                            yk=y_dwd_interpolate,
                            model=vgs_model_dwd)
                        try:
                            ordinary_kriging_dwd_only.krige()

                        except Exception as msg:
                            print('Error while Kriging', msg)

                        interpolated_vals_dwd_only = ordinary_kriging_dwd_only.zk.copy()

                    else:
                        #                 print('no good variogram found, adding nans to df')
                        interpolated_vals_dwd_only = np.nan

        #             print('+++ Saving result to DF +++\n')
                    print('interpolated_vals: ', interpolated_vals_dwd_only)
                    df_interpolated_dwd_only.loc[
                        _cdf_percentile_,
                        stn_dwd_id] = interpolated_vals_dwd_only

        df_interpolated_dwd_only.dropna(how='all', inplace=True)

        df_interpolated_dwd_only.to_csv(out_plots_path / (
            'interpolated_dwd_%s_data_from_qunatiles_using_dwd_%d.csv'
            % (temp_agg, ix)),  sep=';', float_format='%0.5f')
