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
from scipy.spatial import distance_matrix

from pathlib import Path
from random import shuffle

VG = variograms.vgs.Variogram


plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})

# correct all events with edf >
extreme_evts_percentile = 0.6

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
                           r'keep_stns_all_neighbor_99_0_per_60min_s0_1st.csv')

#==============================================================================
#
#==============================================================================

use_dwd_stns_for_kriging = True

qunatile_kriging = True

# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging


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


resample_frequencies = ['60min', '1440min']

idx_time_fmt = '%Y-%m-%d %H:%M:%S'

title_ = r'Quantiles_at_Netatmo'

if not use_netatmo_gd_stns:
    title_ = title_ + '_netatmo_not_filtered_'

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


def build_edf_fr_vals(data):
    """ construct empirical distribution function given data values """
    from statsmodels.distributions.empirical_distribution import ECDF
    cdf = ECDF(data)
    x0 = cdf.x[1:]
    y0 = cdf.y[1:]
    return x0, y0

# def build_edf_fr_vals(data):
#     x = np.sort(np.unique(data))
#     n = x.size
#     y = np.arange(1, n+1) / n
#     return x, y
#==============================================================================
#
#==============================================================================


def find_nearest(array, value):
    ''' given a value, find nearest one to it in original data array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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

    dwd_in_ppt_vals_df_old.dropna(how='all', axis=0, inplace=True)
    ########################################
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
    # create df for corrected events
    netatmo_corrected_edf = netatmo_in_vals_df.copy()

    #======================================================================
    # START KRIGING
    #======================================================================
    total_stns = len(netatmo_in_vals_df.columns)
    for stn_netatmo_id in netatmo_in_vals_df.columns:

        # stn_netatmo_id = stn_comb[0]
        #print('\ninterpolating for Netatmo Station', stn_netatmo_id)
        print('\nRemaining Stations: ', total_stns)

        x_netatmo_interpolate = np.array(
            [netatmo_in_coords_df.loc[stn_netatmo_id, 'X']])
        y_netatmo_interpolate = np.array(
            [netatmo_in_coords_df.loc[stn_netatmo_id, 'Y']])

        netatmo_stn_edf_df = netatmo_in_vals_df.loc[
            :, stn_netatmo_id].dropna()

        netatmo_start_date = netatmo_stn_edf_df.index[0]
        netatmo_end_date = netatmo_stn_edf_df.index[-1]

        # select percentiles above 90%
        netatmo_stn_edf_df_high_evts = netatmo_stn_edf_df[
            netatmo_stn_edf_df.values > extreme_evts_percentile]

        # drop stn from all other stations
        all_netatmo_stns_except_interp_loc = [
            stn for stn in netatmo_in_vals_df.columns
            if stn != stn_netatmo_id]

        # select dwd stations with only same period as netatmo stn
        ppt_dwd_stn_vals = dwd_in_ppt_vals_df.loc[
            netatmo_start_date:netatmo_end_date, :].dropna(how='all')

        for event_date in netatmo_stn_edf_df_high_evts.index:

            # netatmo data at this event
            netatmo_ppt_event_ = netatmo_in_ppt_vals_df.loc[event_date,
                                                            stn_netatmo_id]
            netatmo_edf_event_ = netatmo_in_vals_df.loc[event_date,
                                                        stn_netatmo_id]

            print('Event at: ', event_date,
                  '\nObserved PPT: ', netatmo_ppt_event_,
                  '\nObserved Quantile: ', netatmo_edf_event_)

            dwd_xcoords_new = []
            dwd_ycoords_new = []
            ppt_vals_dwd_new_for_obsv_ppt = []

            for dwd_stn_id_new in ppt_dwd_stn_vals.columns:
                ppt_vals_stn_new = ppt_dwd_stn_vals.loc[
                    :, dwd_stn_id_new].dropna()
                if ppt_vals_stn_new.size > 0:
                    ppt_stn_new, edf_stn_new = build_edf_fr_vals(
                        ppt_vals_stn_new)
                    nearst_edf_new = find_nearest(
                        edf_stn_new,
                        netatmo_edf_event_)

                    ppt_idx = np.where(edf_stn_new == nearst_edf_new)

                    try:
                        if ppt_idx[0].size > 1:
                            ppt_for_edf = np.mean(ppt_stn_new[ppt_idx])
                        else:
                            ppt_for_edf = ppt_stn_new[ppt_idx]
                    except Exception as msg:
                        print(msg)

                    if ppt_for_edf[0] >= 0:
                        ppt_vals_dwd_new_for_obsv_ppt.append(ppt_for_edf[0])
                        stn_xcoords_new = dwd_in_coords_df.loc[
                            dwd_stn_id_new, 'X']
                        stn_ycoords_new = dwd_in_coords_df.loc[
                            dwd_stn_id_new, 'Y']

                        dwd_xcoords_new.append(stn_xcoords_new)
                        dwd_ycoords_new.append(stn_xcoords_new)

            ppt_vals_dwd_new_for_obsv_ppt = np.array(
                ppt_vals_dwd_new_for_obsv_ppt)
            dwd_xcoords_new = np.array(dwd_xcoords_new)
            dwd_ycoords_new = np.array(dwd_ycoords_new)

            # print('*Done getting data and coordintates* \n *Fitting variogram*\n')
            try:

                vg_dwd = VG(
                    x=dwd_xcoords_new,
                    y=dwd_ycoords_new,
                    z=ppt_vals_dwd_new_for_obsv_ppt,
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
                #fit_vg_list = ['']
                continue

            vgs_model_dwd = fit_vg_list[0]

            if ('Nug' in vgs_model_dwd or len(vgs_model_dwd) == 0) and (
                    'Exp' not in vgs_model_dwd and 'Sph' not in vgs_model_dwd):
                #                 print('**Variogram %s not valid --> looking for alternative\n**'
                #                       % vgs_model_dwd)
                try:
                    for i in range(1, len(fit_vg_list)):
                        vgs_model_dwd = fit_vg_list[i]
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
                    print('Only Nugget variogram for this day')

            # if type(vgs_model_dwd) != np.float and len(vgs_model_dwd) >
            # 0:
            if ('Nug' in vgs_model_dwd
                    or len(vgs_model_dwd) > 0) and (
                    'Exp' in vgs_model_dwd or
                    'Sph' in vgs_model_dwd):
                #                 print('**Changed Variogram model to**\n\n', vgs_model_dwd)
                print('\n+++ KRIGING +++\n')

                ordinary_kriging_dwd_only = OrdinaryKriging(
                    xi=dwd_xcoords_new,
                    yi=dwd_ycoords_new,
                    zi=ppt_vals_dwd_new_for_obsv_ppt,
                    xk=x_netatmo_interpolate,
                    yk=y_netatmo_interpolate,
                    model=vgs_model_dwd)

                try:
                    ordinary_kriging_dwd_only.krige()
                except Exception as msg:
                    print('Error while Kriging', msg)

                interpolated_vals_dwd_only = ordinary_kriging_dwd_only.zk.copy()

                if interpolated_vals_dwd_only < 0:
                    interpolated_vals_dwd_only = np.nan

                print('**Interpolated PPT by DWD recent: ',
                      interpolated_vals_dwd_only)

            else:
                #print('no good variogram found, adding nans to df')
                interpolated_vals_dwd_only = np.nan

            if interpolated_vals_dwd_only >= 0.:
                #print('\nBacktransforming interpolated ppt to per\n')
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
                        interpolated_vals_dwd_only[0])

                    edf_idx = np.where(ppt_stn_old == nearst_ppt_old)

                    try:
                        if edf_idx[0].size > 1:
                            edf_for_ppt = np.mean(edf_stn_old[edf_idx])
                        else:
                            edf_for_ppt = edf_stn_old[edf_idx]
                    except Exception as msg:
                        print(msg)
                    try:
                        edf_for_ppt = edf_for_ppt[0]

                    except Exception as msg:
                        edf_for_ppt = edf_for_ppt

                    if edf_for_ppt >= 0:
                        edf_vals_dwd_old_for_interp_ppt.append(edf_for_ppt)
                        stn_xcoords_old = dwd_in_coords_df.loc[
                            dwd_stn_id_old, 'X']
                        stn_ycoords_old = dwd_in_coords_df.loc[
                            dwd_stn_id_old, 'Y']

                        dwd_xcoords_old.append(stn_xcoords_old)
                        dwd_ycoords_old.append(stn_xcoords_old)

                edf_vals_dwd_old_for_interp_ppt = np.array(
                    edf_vals_dwd_old_for_interp_ppt)
                dwd_xcoords_old = np.array(dwd_xcoords_old)
                dwd_ycoords_old = np.array(dwd_ycoords_old)

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

                print('**Interpolated DWD: ',
                      interpolated_vals_dwd_old)

                if interpolated_vals_dwd_old < 0:
                    interpolated_vals_dwd_old = np.nan

            else:
                #print('no good variogram found, adding nans to df')
                interpolated_vals_dwd_old = np.nan

            #print('+++ Saving result to DF +++\n')
            netatmo_corrected_edf.loc[event_date,
                                      stn_netatmo_id] = interpolated_vals_dwd_old
        total_stns -= 1

    netatmo_corrected_edf.to_csv(out_plots_path / (
        'corrected_edf_netatmo_%s_1st.csv'
        % (temp_agg)),
        sep=';', float_format='%0.2f')

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s \a\a\a' %
          (time.asctime()))
