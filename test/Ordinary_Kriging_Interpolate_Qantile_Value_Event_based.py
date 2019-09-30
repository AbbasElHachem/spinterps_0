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


from spinterps import (OrdinaryKriging)
from spinterps import variograms

from pathlib import Path
from random import shuffle

VG = variograms.vgs.Variogram

plt.ioff()
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})

# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

path_to_data = main_dir / r'NetAtmo_BW'

# DAILY DATA
path_to_dwd_daily_data = (path_to_data /
                          r'edf_ppt_all_dwd_daily_.csv')

path_to_netatmo_daily_data = (path_to_data /
                              r'edf_ppt_all_netatmo_daily_.csv')

# HOURLY DATA
path_to_dwd_hourly_data = (path_to_data /
                           r'edf_ppt_all_dwd_hourly_.csv')

path_to_netatmo_hourly_data = (path_to_data /
                               r'edf_ppt_all_netatmo_hourly_.csv')

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# NETATMO FIRST FILTER
path_to_netatmo_gd_stns = (
    r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes\plots_NetAtmo_ppt_DWD_ppt_correlation_"
    r'\keep_stns_all_neighbor_90_per_60min_s0.csv')

# DWD Extreme Events Daily and Hourly

# DAILY and Hourly DATA
path_to_dwd_daily_data_extremes = (path_to_data /
                                   r'dwd_daily_maximum_100_days.csv')

path_to_dwd_hourly_data_extremes = (path_to_data /
                                    r'dwd_hourly_maximum_100_hours.csv')
# =============================================================================
strt_date = '2015-01-01'
end_date = '2019-09-01'

# warm_season_month = [5, 6, 7, 8, 9]  # mai till sep
# cold_season_month = [10, 11, 12, 1, 2, 3, 4]  # oct till april


#list_ppt_values = np.round(np.arange(0.5, 50.00, 0.5), 2)
#list_percentiles = np.round(np.arange(0.5, 1.000001, 0.0025), 4)

min_valid_stns = 10

drop_stns = []
mdr = 0.5
perm_r_list_ = [1, 2]
fit_vgs = ['Sph', 'Exp']  # 'Sph',
fil_nug_vg = 'Nug'  # 'Nug'
n_best = 4
ngp = 5


use_daily_data = True
use_hourly_data = False

use_netatmo_gd_stns = False


# =============================================================================
if use_daily_data:
    path_to_dwd_ppt_data = path_to_dwd_daily_data
    path_to_netatmo_ppt_data = path_to_netatmo_daily_data

    path_to_dwd_ppt_data_extremes = path_to_dwd_daily_data_extremes

    idx_time_fmt = '%Y-%m-%d'
    time_res = 'daily'
if use_hourly_data:
    path_to_dwd_ppt_data = path_to_dwd_hourly_data
    path_to_netatmo_ppt_data = path_to_netatmo_hourly_data

    path_to_dwd_ppt_data_extremes = path_to_dwd_hourly_data_extremes

    idx_time_fmt = '%Y-%m-%d %H:%M:%S'
    time_res = 'hourly'

#==============================================================================
# # DWD DATA
#==============================================================================

# ppt data
dwd_ppt_data_df = pd.read_csv(
    path_to_dwd_ppt_data, sep=';', index_col=0, encoding='utf-8')
dwd_ppt_data_df.index = pd.to_datetime(
    dwd_ppt_data_df.index, format=idx_time_fmt)

dwd_ppt_data_df = dwd_ppt_data_df.loc[strt_date:end_date, :]
dwd_ppt_data_df.dropna(how='all', axis=0, inplace=True)


# station coords
dwd_in_coords_df = pd.read_csv(path_to_dwd_coords,
                               index_col=0,
                               sep=';',
                               encoding='utf-8')
stndwd_ix = ['0' * (5 - len(str(stn_id))) + str(stn_id)
             if len(str(stn_id)) < 5 else str(stn_id)
             for stn_id in dwd_in_coords_df.index]

dwd_in_coords_df.index = stndwd_ix
dwd_in_coords_df.index = list(map(str, dwd_in_coords_df.index))
#==============================================================================
# # NETATMO DATA
#==============================================================================
# ppt data
netatmo_ppt_data_df = pd.read_csv(
    path_to_netatmo_ppt_data, sep=';', index_col=0, encoding='utf-8')
netatmo_ppt_data_df.index = pd.to_datetime(
    netatmo_ppt_data_df.index, format=idx_time_fmt)

netatmo_ppt_data_df = netatmo_ppt_data_df.loc[strt_date:end_date, :]
netatmo_ppt_data_df.dropna(how='all', axis=0, inplace=True)


netatmo_in_coords_df = pd.read_csv(path_to_netatmo_coords,
                                   index_col=0,
                                   sep=';',
                                   encoding='utf-8').dropna()

if use_netatmo_gd_stns:
    df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
                             index_col=0,
                             sep=';',
                             encoding='utf-8')
    good_netatmo_stns = df_gd_stns.loc[:, 'Stations'].values.ravel()

    netatmo_in_coords_df = netatmo_in_coords_df.loc[good_netatmo_stns, :].dropna(
    )
    netatmo_ppt_data_df = netatmo_ppt_data_df.loc[:,
                                                  netatmo_in_coords_df.index]

#==============================================================================
# Extreme Events Data
#==============================================================================
dwd_ppt_data_df_extremes = pd.read_csv(
    path_to_dwd_ppt_data_extremes, sep=';', index_col=0, encoding='utf-8',
    header=None)
dwd_ppt_data_df_extremes.index = pd.to_datetime(
    dwd_ppt_data_df_extremes.index, format=idx_time_fmt)

dwd_ppt_data_df_extremes = dwd_ppt_data_df_extremes.loc[strt_date:end_date, :]

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


all_dwd_stns = dwd_ppt_data_df.columns.tolist()
shuffle(all_dwd_stns)
shuffled_dwd_stns_10stn = np.array(list(chunks(all_dwd_stns, 2)))

#==============================================================================
# CREATE DFS HOLD RESULT KRIGING PER NETATMO STATION
#==============================================================================
for idx_lst_comb in range(len(shuffled_dwd_stns_10stn)):
    stn_comb = shuffled_dwd_stns_10stn[idx_lst_comb]

    print('Interpolating for following DWD stations: \n',
          pprint.pformat(stn_comb))

    df_interpolated_dwd_netatmos_comb = pd.DataFrame(
        index=dwd_ppt_data_df_extremes.index,
        columns=[stn_comb])

    df_interpolated_dwd_only = pd.DataFrame(
        index=dwd_ppt_data_df_extremes.index,
        columns=[stn_comb])

    df_interpolated_netatmo_only = pd.DataFrame(
        index=dwd_ppt_data_df_extremes.index,
        columns=[stn_comb])

    #=========================================================================
    # START KRIGING
    #=========================================================================

    for stn_dwd_id in stn_comb:

        print('interpolating for DWD Station', stn_dwd_id)

        x_dwd_interpolate = np.array([dwd_in_coords_df.loc[stn_dwd_id, 'X']])
        y_dwd_interpolate = np.array([dwd_in_coords_df.loc[stn_dwd_id, 'Y']])

        # drop stn
        all_dwd_stns_except_interp_loc = [
            stn for stn in dwd_ppt_data_df.columns if stn != stn_dwd_id]

        for event_date in dwd_ppt_data_df_extremes.index:
            _stn_id_event_ = str(dwd_ppt_data_df_extremes.loc[event_date, 2])
            if len(_stn_id_event_) < 5:
                _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                    '0' + _stn_id_event_

            _ppt_event_ = dwd_ppt_data_df_extremes.loc[event_date, 1]
            _edf_event_ = dwd_ppt_data_df.loc[event_date, _stn_id_event_]
            print('**Calculating for Date ',
                  event_date, ' Rainfall: ',  _ppt_event_,
                  'Quantile: ', _edf_event_, ' **\n')

            #==================================================================
            # # DWD qunatiles
            #==================================================================
            edf_dwd_vals = []
            dwd_xcoords = []
            dwd_ycoords = []
            dwd_stn_ids = []

            for stn_id in all_dwd_stns_except_interp_loc:
                #print('station is', stn_id)

                edf_stn_vals = dwd_ppt_data_df.loc[event_date, stn_id]

                if edf_stn_vals > 0:
                    edf_dwd_vals.append(np.round(edf_stn_vals, 4))
                    dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
                    dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
                    dwd_stn_ids.append(stn_id)

            dwd_xcoords = np.array(dwd_xcoords)
            dwd_ycoords = np.array(dwd_ycoords)
            edf_dwd_vals = np.array(edf_dwd_vals)

            #==================================================================
            # # NETATMO QUANTILES
            #==================================================================
            edf_netatmo_vals = []
            netatmo_xcoords = []
            netatmo_ycoords = []
            netatmo_stn_ids = []

            for netatmo_stn_id in netatmo_ppt_data_df.columns:
                # print('Netatmo station is', netatmo_stn_id)
                try:
                    edf_stn_vals = netatmo_ppt_data_df.loc[event_date,
                                                           netatmo_stn_id]

                    if edf_stn_vals > 0:
                        edf_netatmo_vals.append(np.round(edf_stn_vals, 4))
                        netatmo_xcoords.append(
                            netatmo_in_coords_df.loc[netatmo_stn_id, 'X'])
                        netatmo_ycoords.append(
                            netatmo_in_coords_df.loc[netatmo_stn_id, 'Y'])
                        netatmo_stn_ids.append(netatmo_stn_id)
                except KeyError:
                    continue
            # TODO: ask prof if correct
            netatmo_xcoords = np.array(netatmo_xcoords)
            netatmo_ycoords = np.array(netatmo_ycoords)
            edf_netatmo_vals = np.array(edf_netatmo_vals)

            good_stns_idx = [
                ix for ix in
                np.where(edf_netatmo_vals >= np.mean(edf_netatmo_vals))[0]]

            netatmo_xcoords = netatmo_xcoords[good_stns_idx]
            netatmo_ycoords = netatmo_ycoords[good_stns_idx]
            edf_netatmo_vals = edf_netatmo_vals[good_stns_idx]

            dwd_netatmo_xcoords = np.concatenate(
                [dwd_xcoords, netatmo_xcoords])
            dwd_netatmo_ycoords = np.concatenate(
                [dwd_ycoords, netatmo_ycoords])

            coords = np.array([(x, y) for x, y in zip(dwd_netatmo_xcoords,
                                                      dwd_netatmo_ycoords)])
            #dwd_netatmo_ppt = np.hstack((ppt_dwd_vals, ppt_netatmo_vals))
            dwd_netatmo_edf = np.concatenate([edf_dwd_vals, edf_netatmo_vals])

#             plt.scatter(x_dwd_interpolate, y_dwd_interpolate)
#             plt.scatter(netatmo_xcoords, netatmo_ycoords)
#             plt.scatter(dwd_xcoords, dwd_ycoords)
            #plt.scatter(dwd_netatmo_xcoords, dwd_netatmo_ycoords)

            print('*Done getting data and coordintates* \n *Fitting variogram*\n')
            try:

                vg_dwd = VG(
                    x=dwd_xcoords,
                    y=dwd_ycoords,
                    z=edf_dwd_vals,
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
                continue

            vgs_model_dwd = fit_vg_list[0]

            if ('Nug' in vgs_model_dwd or len(vgs_model_dwd) == 0) and (
                    'Exp' not in vgs_model_dwd and 'Sph' not in vgs_model_dwd):
                print('**Variogram %s not valid --> looking for alternative\n**'
                      % vgs_model_dwd)
                try:
                    for i in range(1, 3):
                        vgs_model_dwd = fit_vg_list[i]
                        if type(vgs_model_dwd) == np.float:
                            continue
                        if ('Nug' in vgs_model_dwd
                            or len(vgs_model_dwd) == 0) and ('Exp' not in vgs_model_dwd and
                                                             'Sph' not in vgs_model_dwd):
                            continue
                        else:
                            break
                    print('**Changed Variogram model to**\n', vgs_model_dwd)

                except Exception as msg:
                    print(msg)
                    print('Only Nugget variogram for this day')

            if type(vgs_model_dwd) != np.float and len(vgs_model_dwd) > 0:

                print('+++ KRIGING +++\n')

                ordinary_kriging_dwd_netatmo_comb = OrdinaryKriging(
                    xi=dwd_netatmo_xcoords,
                    yi=dwd_netatmo_ycoords,
                    zi=dwd_netatmo_edf,
                    xk=x_dwd_interpolate,
                    yk=y_dwd_interpolate,
                    model=vgs_model_dwd)

                ordinary_kriging_dwd_only = OrdinaryKriging(
                    xi=dwd_xcoords,
                    yi=dwd_ycoords,
                    zi=edf_dwd_vals,
                    xk=x_dwd_interpolate,
                    yk=y_dwd_interpolate,
                    model=vgs_model_dwd)

                ordinary_kriging_netatmo_only = OrdinaryKriging(
                    xi=netatmo_xcoords,
                    yi=netatmo_ycoords,
                    zi=edf_netatmo_vals,
                    xk=x_dwd_interpolate,
                    yk=y_dwd_interpolate,
                    model=vgs_model_dwd)

                try:
                    ordinary_kriging_dwd_netatmo_comb.krige()
                    ordinary_kriging_dwd_only.krige()
                    ordinary_kriging_netatmo_only.krige()
                except Exception as msg:
                    print('Error while Kriging', msg)

                interpolated_vals_dwd_netatmo = ordinary_kriging_dwd_netatmo_comb.zk.copy()
                interpolated_vals_dwd_only = ordinary_kriging_dwd_only.zk.copy()
                interpolated_vals_netatmo_only = ordinary_kriging_netatmo_only.zk.copy()

                if interpolated_vals_dwd_netatmo < 0:
                    interpolated_vals_dwd_netatmo = np.nan

                if interpolated_vals_dwd_only < 0:
                    interpolated_vals_dwd_only = np.nan

                if interpolated_vals_netatmo_only < 0:
                    interpolated_vals_netatmo_only = np.nan
            else:
                print('no good variogram found, adding nans to df')
                interpolated_vals_dwd_netatmo = np.nan
                interpolated_vals_dwd_only = np.nan
                interpolated_vals_netatmo_only = np.nan

            print('+++ Saving result to DF +++\n')

            df_interpolated_dwd_netatmos_comb.loc[
                event_date,
                stn_dwd_id] = interpolated_vals_dwd_netatmo

            df_interpolated_dwd_only.loc[
                event_date,
                stn_dwd_id] = interpolated_vals_dwd_only

            df_interpolated_netatmo_only.loc[
                event_date,
                stn_dwd_id] = interpolated_vals_netatmo_only

    df_interpolated_dwd_netatmos_comb.dropna(how='all', inplace=True)
    df_interpolated_dwd_only.dropna(how='all', inplace=True)
    df_interpolated_netatmo_only.dropna(how='all', inplace=True)

    df_interpolated_dwd_netatmos_comb.to_csv(out_plots_path / (
        'interpolated_quantiles_dwd_%s_data_basedon_quantiles_%s_season_using_dwd_netamo_grp_%d_.csv'
        % (time_res, str(event_date).replace('-', '_').replace(':', '_').replace(' ', '_'),
           idx_lst_comb)),
        sep=';', float_format='%0.2f')

    df_interpolated_dwd_only.to_csv(out_plots_path / (
        'interpolated_quantiles_dwd_%s_data_basedon_qunatiles_%s_season_using_dwd_only_grp_%d_.csv'
        % (time_res,  str(event_date).replace('-', '_').replace(':', '_').replace(' ', '_'),
           idx_lst_comb)),
        sep=';', float_format='%0.2f')

    df_interpolated_netatmo_only.to_csv(out_plots_path / (
        'interpolated_quantiles_%s_data_basedon_qunatiles_%s_season_using_netatmo_only_grp_%d_.csv'
        % (time_res, str(event_date).replace('-', '_').replace(':', '_').replace(' ', '_'),
            idx_lst_comb)),
        sep=';', float_format='%0.2f')

    break
stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
