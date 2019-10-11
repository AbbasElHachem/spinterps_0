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

path_to_vgs = main_dir / r'kriging_ppt_netatmo'

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# NETATMO FIRST FILTER
path_to_netatmo_gd_stns = (main_dir / r'plots_NetAtmo_ppt_DWD_ppt_correlation_' /
                           r'keep_stns_all_neighbor_98_per_60min_s0.csv')

#==============================================================================
#
#==============================================================================

use_dwd_stns_for_kriging = True

qunatile_kriging = True

# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging
use_temporal_filter_after_kriging = True  # on day filter

strt_date = '2015-01-01 00:00:00'
end_date = '2019-09-01 00:00:00'

min_valid_stns = 10

drop_stns = []
mdr = 0.5
perm_r_list_ = [1, 2]
fit_vgs = ['Sph', 'Exp']  # 'Sph',
fil_nug_vg = 'Nug'  # 'Nug'
n_best = 4
ngp = 5


resample_frequencies = ['60min', '120min', '180min',
                        '360min', '720min', '1440min']

idx_time_fmt = '%Y-%m-%d %H:%M:%S'

title_ = r'Quantiles'

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
df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
                         index_col=0,
                         sep=';',
                         encoding='utf-8')
#==============================================================================
# NEEDED FUNCTIONS
#==============================================================================


def build_edf_fr_vals(ppt_data):
    """ construct empirical distribution function given data values """
    data_sorted = np.sort(ppt_data, axis=0)[::-1]
    x0 = np.round(np.squeeze(data_sorted)[::-1], 1)
    y0 = np.round((np.arange(data_sorted.size) / len(data_sorted)), 3)

    return x0, y0

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

    # path to data
    #=========================================================================
    out_save_csv = '_%s_ppt_edf_' % temp_agg

    path_to_dwd_edf = (path_to_data /
                       (r'edf_ppt_all_dwd_%s_.csv' % temp_agg))

    path_to_netatmo_edf = (path_to_data /
                           (r'edf_ppt_all_netatmo_good_stns_%s_.csv' % temp_agg))

    path_to_dwd_vgs = path_to_vgs / \
        (r'vg_strs_dwd_%s_maximum_100_event.csv' % temp_agg)

    path_dwd_extremes_df = path_to_data / \
        (r'dwd_%s_maximum_100_event.csv' % temp_agg)

    # netatmo second filter
    path_to_netatmo_edf_temp_filter = (
        out_plots_path /
        (r'all_netatmo__%s_ppt_edf__using_DWD_stations_to_find_Netatmo_values__temporal_filter_98perc_.csv'
         % temp_agg))

    # Files to use
    #==========================================================================

    if qunatile_kriging:
        netatmo_data_to_use = path_to_netatmo_edf
        dwd_data_to_use = path_to_dwd_edf
        path_to_dwd_vgs = path_to_dwd_vgs

        if not use_netatmo_gd_stns:
            title_ = title_ + '_netatmo_not_filtered_'

        if not use_temporal_filter_after_kriging:
            title = title_ + '_no_Temporal_filter_used_'

        if use_temporal_filter_after_kriging:
            path_to_netatmo_temp_filter = path_to_netatmo_edf_temp_filter
            title_ = title_ + '_Temporal_filter_used_'

    # DWD DATA
    #=========================================================================
    dwd_in_vals_df = pd.read_csv(
        path_to_dwd_edf, sep=';', index_col=0, encoding='utf-8')

    dwd_in_vals_df.index = pd.to_datetime(
        dwd_in_vals_df.index, format='%Y-%m-%d')

    dwd_in_vals_df = dwd_in_vals_df.loc[strt_date:end_date, :]
    dwd_in_vals_df.dropna(how='all', axis=0, inplace=True)

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
    # apply first filter
    good_netatmo_stns = df_gd_stns.loc[:, 'Stations'].values.ravel()

    in_vals_df = netatmo_in_vals_df.loc[:, good_netatmo_stns]

    netatmo_in_coords_df = netatmo_in_coords_df.loc[good_netatmo_stns, :]
    cmn_stns = netatmo_in_coords_df.index.intersection(
        netatmo_in_vals_df.columns)

    netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_stns]

    # DWD Extremes
    #=========================================================================
    dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,
                                     index_col=0,
                                     sep=';',
                                     parse_dates=True,
                                     infer_datetime_format=True,
                                     encoding='utf-8',
                                     header=None)

    # if temporal filter for Netamo
    #=========================================================================

    if use_temporal_filter_after_kriging:
        # apply second filter
        df_all_stns_per_events = pd.read_csv(
            path_to_netatmo_temp_filter,
            sep=';', index_col=0,
            parse_dates=True,
            infer_datetime_format=True)

        dwd_in_extremes_df = dwd_in_extremes_df.loc[
            dwd_in_extremes_df.index.intersection(df_all_stns_per_events.index), :]

    # VG MODELS
    #==========================================================================
    df_vgs = pd.read_csv(path_to_dwd_vgs,
                         index_col=0,
                         sep=';',
                         encoding='utf-8')

    df_vgs.index = pd.to_datetime(df_vgs.index, format='%Y-%m-%d')
    df_vgs_models = df_vgs.iloc[:, 0]
    df_vgs_models.dropna(how='all', inplace=True)

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

        df_interpolated_dwd_only = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        df_interpolated_netatmo_only = pd.DataFrame(
            index=dwd_in_extremes_df.index,
            columns=[stn_comb])

        #======================================================================
        # START KRIGING
        #======================================================================

        for stn_dwd_id in stn_comb:

            print('interpolating for DWD Station', stn_dwd_id)

            x_dwd_interpolate = np.array(
                [dwd_in_coords_df.loc[stn_dwd_id, 'X']])
            y_dwd_interpolate = np.array(
                [dwd_in_coords_df.loc[stn_dwd_id, 'Y']])

            # drop stn
            all_dwd_stns_except_interp_loc = [
                stn for stn in dwd_in_vals_df.columns if stn != stn_dwd_id]

            for event_date in dwd_in_extremes_df.index:
                _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
                if len(_stn_id_event_) < 5:
                    _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                        '0' + _stn_id_event_

                _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
                _edf_event_ = dwd_in_vals_df.loc[event_date, _stn_id_event_]
                print('**Calculating for Date ',
                      event_date, ' Rainfall: ',  _ppt_event_,
                      'Quantile: ', _edf_event_, ' **\n')

                #==============================================================
                # # DWD qunatiles
                #==============================================================
                edf_dwd_vals = []
                dwd_xcoords = []
                dwd_ycoords = []
                dwd_stn_ids = []

                for stn_id in all_dwd_stns_except_interp_loc:
                    #print('station is', stn_id)

                    edf_stn_vals = dwd_in_vals_df.loc[event_date, stn_id]

                    if edf_stn_vals > 0:
                        edf_dwd_vals.append(np.round(edf_stn_vals, 4))
                        dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
                        dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
                        dwd_stn_ids.append(stn_id)

                dwd_xcoords = np.array(dwd_xcoords)
                dwd_ycoords = np.array(dwd_ycoords)
                edf_dwd_vals = np.array(edf_dwd_vals)

                #==============================================================
                # # NETATMO QUANTILES
                #==============================================================

                edf_netatmo_vals = []
                netatmo_xcoords = []
                netatmo_ycoords = []
                netatmo_stn_ids = []

                # Netatmo data and coords
                netatmo_df = netatmo_in_vals_df.loc[event_date, :].dropna(
                    how='all')

                # apply temp filter per event

                if use_temporal_filter_after_kriging:
                    print('apllying on event filter')

                    df_all_stns_per_event = df_all_stns_per_events.loc[event_date, :]

                    all_stns = df_all_stns_per_event.index
                    stns_keep_per_event = all_stns[np.where(
                        df_all_stns_per_event.values > 0)]
                    print('\n----Keeping %d / %d Stns for event---'
                          % (stns_keep_per_event.shape[0],
                             all_stns.shape[0]))
                    # keep only good stations
                    netatmo_df = netatmo_in_vals_df.loc[
                        event_date, stns_keep_per_event].dropna(how='all')

                netatmo_stns = netatmo_df.index

                for netatmo_stn_id in netatmo_stns:
                    # print('Netatmo station is', netatmo_stn_id)

                    try:
                        edf_stn_vals = netatmo_in_vals_df.loc[event_date,
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

                netatmo_xcoords = np.array(netatmo_xcoords)
                netatmo_ycoords = np.array(netatmo_ycoords)
                edf_netatmo_vals = np.array(edf_netatmo_vals)

                dwd_netatmo_xcoords = np.concatenate(
                    [dwd_xcoords, netatmo_xcoords])
                dwd_netatmo_ycoords = np.concatenate(
                    [dwd_ycoords, netatmo_ycoords])

                coords = np.array([(x, y) for x, y in zip(dwd_netatmo_xcoords,
                                                          dwd_netatmo_ycoords)])

                dwd_netatmo_edf = np.concatenate([edf_dwd_vals,
                                                  edf_netatmo_vals])

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
                                    or len(vgs_model_dwd) == 0) and (
                                        'Exp' not in vgs_model_dwd and
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
            'interpolated_quantiles_dwd_%s_data_%s_using_dwd_netamo_grp_%d_.csv'
            % (temp_agg, title_, idx_lst_comb)),
            sep=';', float_format='%0.2f')

        df_interpolated_dwd_only.to_csv(out_plots_path / (
            'interpolated_quantiles_dwd_%s_data_%s_using_dwd_only_grp_%d_.csv'
            % (temp_agg, title_, idx_lst_comb)),
            sep=';', float_format='%0.2f')

        df_interpolated_netatmo_only.to_csv(out_plots_path / (
            'interpolated_quantiles_dwd_%s_data_%s_using_netamo_only_grp_%d_.csv'
            % (temp_agg, title_, idx_lst_comb)),
            sep=';', float_format='%0.2f')
        break
    break

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s \a\a\a' %
          (time.asctime()))
