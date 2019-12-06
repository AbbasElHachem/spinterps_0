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


from spinterps import (OrdinaryKriging)
from spinterps import variograms


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


# path for data filter
in_filter_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'


_acc_ = ''

#==============================================================================
#
#==============================================================================
resample_frequencies = ['180min']
# '120min', '180min', '60min',  '360min',
#                         '720min',
title_ = r'ppt_ratios'


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


#==============================================================================
# SELECT GROUP OF 10 DWD STATIONS RANDOMLY
#==============================================================================


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


#==============================================================================
# make out dir
#==============================================================================
dir_path = title_ + '_' + _acc_

out_plots_path = in_filter_path / dir_path

if not os.path.exists(out_plots_path):
    os.mkdir(out_plots_path)

for temp_agg in resample_frequencies:

    # out path directory

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
            df_vgs_extremes.index), :]

    print('\n%d Extreme Event to interpolate\n' % dwd_in_extremes_df.shape[0])
    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()
    shuffle(all_dwd_stns)
    shuffled_dwd_stns_10stn = np.array(list(chunks(all_dwd_stns, 10)))

    #==========================================================================
    # # CREATE DFS FOR RESULT; Index is Date, Columns as Stns
    #==========================================================================

    # get ratio between interpolated and obsv at DWD for 2nd filter
    df_interpolated_ratios = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)

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
        for idx_lst_comb in range(len(shuffled_dwd_stns_10stn)):
            stn_comb = shuffled_dwd_stns_10stn[idx_lst_comb]

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

                    ordinary_kriging_dwd_ppt = OrdinaryKriging(
                        xi=x_dwd_all,
                        yi=y_dwd_all,
                        zi=ppt_dwd_vals_nona.values,
                        xk=x_dwd_interpolate,
                        yk=y_dwd_interpolate,
                        model=vgs_model_dwd_ppt)

                    try:

                        ordinary_kriging_dwd_ppt.krige()

                    except Exception as msg:
                        print('Error while Kriging', msg)

                    interpolated_dwd_ppt = ordinary_kriging_dwd_ppt.zk.copy()[
                        0]

                    if interpolated_dwd_ppt < 0:
                        interpolated_dwd_ppt = np.nan

                    print('**Interpolated PPT by DWD Only: ',
                          interpolated_dwd_ppt)

                    # get ratio between observed and interpolated data:
                    ratio_stn = np.abs(
                        obs_ppt_stn_dwd / interpolated_dwd_ppt)

                    df_interpolated_ratios.loc[
                        event_date,
                        stn_dwd_id] = ratio_stn

            print('Done with this group of DWD Stns')

        print('Done getting Ratios with this event')

    print('Done getting Ratios for all event')

    df_interpolated_ratios.to_csv(out_plots_path / (
        'dwd_ratios_%s_data.csv'
        % (temp_agg)),
        sep=';', float_format='%0.2f')


stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
