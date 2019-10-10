# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Interpolate Quantiles at Netatmo Locations using DWD
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

import pandas as pd
import matplotlib.pyplot as plt

from spinterps import (OrdinaryKriging)
from spinterps import variograms

from pathlib import Path

VG = variograms.vgs.Variogram

plt.ioff()
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})

# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

path_to_data = main_dir / r'oridinary_kriging_compare_DWD_Netatmo/needed_dfs'

# DAILY DATA
path_to_dwd_daily_data = (path_to_data /
                          r'all_dwd_daily_ppt_data_combined_2014_2019_.csv')

# HOURLY DATA
path_to_dwd_hourly_data = (path_to_data /
                           r'all_dwd_hourly_ppt_data_combined_2014_2019_.csv')
# containing CDF values
path_to_dwd_hourly_cold_season_data = (path_to_data /
                                       r'df_dwd_distributions_cold_season_hourly.csv')
path_to_dwd_hourly_warm_season_data = (path_to_data /
                                       r'df_dwd_distributions_warm_season_hourly.csv')

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# NETATMO FIRST FILTER
path_to_netatmo_gd_stns = (path_to_data /
                           r'keep_stns_all_neighbor_90_per_60min_.csv')

# =============================================================================
strt_date = '2014-01-01'
end_date = '2019-08-01'

warm_season_month = [5, 6, 7, 8, 9]  # mai till sep
cold_season_month = [10, 11, 12, 1, 2, 3, 4]  # oct till april


list_percentiles = np.round(np.arange(0.5, 1.0001, 0.01), 3)

min_valid_stns = 20

drop_stns = []
mdr = 0.8
perm_r_list = [1, 2, 3]
fit_vgs = ['Sph', 'Exp']  # 'Sph',
fil_nug_vg = 'Nug'  # 'Nug'
n_best = 4
ngp = 5


use_daily_data = True
use_hourly_data = False

use_netatmo_gd_stns = True
do_it_for_cold_season = True  # True
do_it_for_warm_season = False  # False

# =============================================================================
if use_daily_data:
    path_to_dwd_ppt_data = path_to_dwd_daily_data

    idx_time_fmt = '%Y-%m-%d'
    time_res = 'daily'
if use_hourly_data:
    path_to_dwd_ppt_data = path_to_dwd_hourly_data

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

    netatmo_in_coords_df = netatmo_in_coords_df.loc[good_netatmo_stns, :]

x_netatmo = netatmo_in_coords_df.loc[:, 'X'].values.ravel()
y_netatmo = netatmo_in_coords_df.loc[:, 'Y'].values.ravel()


#==============================================================================
# # VG MODELS
#==============================================================================
def build_edf_fr_vals(ppt_data):
    # Construct EDF, need to check if it works
    """ construct empirical distribution function given data values """
    data_sorted = np.sort(ppt_data, axis=0)[::-1]
    x0 = np.round(np.squeeze(data_sorted)[::-1], 2)
    y0 = np.round((np.arange(data_sorted.size) / len(data_sorted)), 3)

    return x0, y0

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
# GET DATAFRAME PER SEASON
#==============================================================================
if do_it_for_cold_season:
    stn_data_season = select_season(dwd_ppt_data_df, cold_season_month)
    data_season = 'cold'

if do_it_for_warm_season:
    stn_data_season = select_season(dwd_ppt_data_df, warm_season_month)
    data_season = 'warm'
#==============================================================================
# CREATE DFS HOLD RESULT KRIGING PER NETATMO STATION
#==============================================================================

df_netatmos = pd.DataFrame(index=list_percentiles,
                           columns=[netatmo_in_coords_df.index])


#==============================================================================
# START KRIGING
#==============================================================================


for _cdf_percentile_ in list_percentiles:

    _cdf_percentile_ = np.round(_cdf_percentile_, 2)
    print('**Calculating for percentile: ',  _cdf_percentile_, ' **\n')

    ppt_vals = []
    dwd_xcoords = []
    dwd_ycoords = []
    stn_ids = []

    for stn_id in stn_data_season.columns:
        # print('station is', stn_id)
        stn_data_df = stn_data_season.loc[:, stn_id].dropna()
        ppt_cold_season, edf_cold_season = build_edf_fr_vals(
            stn_data_df.values)

        ppt_percentile = ppt_cold_season[edf_cold_season == _cdf_percentile_]
        if ppt_percentile.shape[0] > 0:
            ppt_vals.append(np.unique(ppt_percentile)[0])
            dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
            dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
            stn_ids.append(stn_id)

    dwd_xcoords = np.array(dwd_xcoords)
    dwd_ycoords = np.array(dwd_ycoords)
    ppt_vals = np.array(ppt_vals)

    vgs_list_all = []
    print('*Done getting data and coordintates* \n *Fitting variogram*\n')
    try:
        vg = VG(
            x=dwd_xcoords,
            y=dwd_ycoords,
            z=ppt_vals,
            mdr=mdr,
            nk=10,
            typ='cnst',
            perm_r_list=perm_r_list,
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

        vg.fit()
    except Exception as msg:
        print(msg)

    fit_vg_list = vg.vg_str_list

    vgs_model = fit_vg_list[0]

    if ('Nug' in vgs_model or len(vgs_model) == 0) and (
            'Exp' not in vgs_model and 'Sph' not in vgs_model):
        print('**Variogram %s not valid --> looking for alternative\n**'
              % vgs_model)
        try:
            for i in range(1, 4):
                vgs_model = fit_vg_list[i]
                if type(vgs_model) == np.float:
                    continue
                if ('Nug' in vgs_model
                    or len(vgs_model) == 0) and ('Exp' not in vgs_model and
                                                 'Sph' not in vgs_model):
                    continue
                else:
                    break

        except Exception as msg:
            print(msg)
            print('Only Nugget variogram for this day')


#         evg = vg.vg_vg_arr
#         h_arr = vg.vg_h_arr
#         vg_fit = vg.vg_fit
#
#         plt.ioff()
#         plt.figure(figsize=(20, 12))
#
#         plt.plot(h_arr, evg, 'bo', alpha=0.3)
#
#         for m in range(len(fit_vg_list)):
#             plt.plot(
#                 vg_fit[m][:, 0],
#                 vg_fit[m][:, 1],
#                 c=pd.np.random.rand(3,),
#                 linewidth=4,
#                 zorder=m,
#                 label=fit_vg_list[m],
#                 alpha=0.6)
#
#         plt.grid()
#
#         plt.xlabel('Distance')
#         plt.ylabel('Variogram')
#
#         plt.title(
#             'Percentile %.3f' % (_cdf_percentile_), fontdict={'fontsize': 15})
#
#         plt.legend(loc=4, framealpha=0.7)
#         plt.show()
#         plt.savefig(
#             str(self._out_figs_path / f'{date_str}.png'),
#             bbox_inches='tight')
#
#         plt.close()

    if type(vgs_model) != np.float:
        print('**Changed Variogram model to**\n', vgs_model)
        print('+++ KRIGING +++\n')
        vgs_list_all.append(vgs_model)
        ordinary_kriging = OrdinaryKriging(
            xi=dwd_xcoords,
            yi=dwd_ycoords,
            zi=ppt_vals,
            xk=x_netatmo,
            yk=y_netatmo,
            model=vgs_model)

        try:
            ordinary_kriging.krige()
        except Exception as msg:
            print('Error while Kriging', msg)

        interpolated_vals = ordinary_kriging.zk.copy()

    else:
        interpolated_vals = np.nan

    print('+++ Saving result to DF +++\n')
    df_netatmos.loc[_cdf_percentile_, :] = interpolated_vals

df_netatmos.dropna(how='all', inplace=True)
df_netatmos.to_csv(out_plots_path / (
    'interpolated_%s_data_from_qunatiles_%s_season.csv'
    % (time_res, data_season)),
    sep=';', float_format='%0.2f')


stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
