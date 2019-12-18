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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import rankdata

from pathlib import Path

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


_acc_ = ''


#==============================================================================
#
#==============================================================================
resample_frequencies = ['180min', '360min', '720min', '1440min']
# '60min', '180min', '360min', '720min', '1440min'


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

#==============================================================================
# Needed functions
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


#==============================================================================
#
#==============================================================================


# read df combinations to use
df_dwd_stns_comb = pd.read_csv(
    path_to_dwd_stns_comb, index_col=0,
    sep=',', dtype=str)
#==============================================================================
#
#==============================================================================
for temp_agg in resample_frequencies:

    # out path directory
    dir_path = _acc_ + '_' + temp_agg

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

    path_dwd_extremes_df = path_to_data / \
        (r'dwd_%s_maximum_100_event.csv' % temp_agg)

    # Files to use
    #==========================================================================

    dwd_data_to_use = path_to_dwd_edf

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

    # DWD Extremes
    #=========================================================================
    dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,
                                     index_col=0,
                                     sep=';',
                                     parse_dates=True,
                                     infer_datetime_format=True,
                                     encoding='utf-8',
                                     header=None)

    #==============================================================
    # # DWD qunatiles for every event, fit a variogram
    #==============================================================

    df_vgs_extremes_norm = pd.DataFrame(index=dwd_in_extremes_df.index)

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
        print('**Calculating for Date ',
              event_date, '\n Rainfall: ',  _ppt_event_,
              'Quantile: ', _edf_event_, ' **\n')
        edf_dwd_vals = []
        dwd_xcoords = []
        dwd_ycoords = []
        dwd_stn_ids = []

        for stn_id in dwd_in_vals_df.columns:
            #print('station is', stn_id)

            edf_stn_vals = dwd_in_vals_df.loc[
                event_date, stn_id]
            # ppt_stn_vals
            if edf_stn_vals > 0:
                edf_dwd_vals.append(edf_stn_vals)
                dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
                dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
                dwd_stn_ids.append(stn_id)

        dwd_xcoords = np.array(dwd_xcoords)
        dwd_ycoords = np.array(dwd_ycoords)
        edf_dwd_vals = np.array(edf_dwd_vals)

        # transform to normal
        std_norm_edf_dwd_vals = norm.ppf(
            rankdata(edf_dwd_vals) / (
                len(edf_dwd_vals) + 1))
#         plt.ioff()
#         std_norm_edf_dwd_vals_sr = pd.Series(
#             std_norm_edf_dwd_vals)
#         std_norm_edf_dwd_vals_sr.plot.hist(
#             grid=True, bins=5, rwidth=0.9,
#             alpha=0.5, color='b', label='dwd')
#         plt.show()
        # fit variogram
        print('*Done getting data* \n *Fitting variogram*\n')
        try:

            vg_dwd = VG(
                x=dwd_xcoords,
                y=dwd_ycoords,
                z=std_norm_edf_dwd_vals,
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
        if len(vgs_model_dwd) > 0:
            df_vgs_extremes_norm.loc[event_date, 1] = vgs_model_dwd
        else:
            vgs_model_dwd = fit_vg_list[1]
            if len(vgs_model_dwd) > 0:
                df_vgs_extremes_norm.loc[event_date, 1] = vgs_model_dwd
            else:
                df_vgs_extremes_norm.loc[event_date, 1] = np.nan
    df_vgs_extremes_norm.dropna(how='all', inplace=True)
    df_vgs_extremes_norm.to_csv((path_to_vgs /
                                 ('dwd_edf_transf_vg_%s.csv' % temp_agg)),
                                sep=';')


stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
