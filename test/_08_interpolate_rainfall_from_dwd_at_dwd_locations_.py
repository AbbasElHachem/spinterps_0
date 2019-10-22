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

import shapefile as shp

from spinterps import (OrdinaryKriging)
from spinterps import variograms
from scipy.interpolate import griddata

from pathlib import Path

from random import shuffle
from matplotlib import cm, colors

VG = variograms.vgs.Variogram


plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})


# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

path_to_data = main_dir / r'NetAtmo_BW'


# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

bw_area_shp = (r"X:\hiwi\ElHachem\GitHub\extremes"
               r"\Landesgrenze_ETRS89\Landesgrenze_10000_ETRS89.shp")
#==============================================================================
#
#==============================================================================

use_dwd_stns_for_kriging = True

qunatile_kriging = True

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

title_ = r'Rainfall'

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
# NEEDED FUNCTIONS
#==============================================================================


# def build_edf_fr_vals(ppt_data):
#     """ construct empirical distribution function given data values """
#     data_sorted = np.sort(ppt_data, axis=0)[::-1]
#     x0 = np.round(np.squeeze(data_sorted)[::-1], 1)
#     y0 = np.round((np.arange(data_sorted.size) / len(data_sorted)), 3)
#
#     return x0, y0

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
    out_save_csv = '_%s_ppt_' % temp_agg

    path_to_dwd_ppt = (path_to_data /
                       (r'ppt_all_dwd_%s_.csv' % temp_agg))

    path_dwd_extremes_df = path_to_data / \
        (r'dwd_%s_maximum_100_event.csv' % temp_agg)

    # Files to use
    #==========================================================================

    print(title_)
    # DWD DATA
    #=========================================================================
    # ppt values
    dwd_in_ppt_vals = pd.read_csv(
        path_to_dwd_ppt, sep=';', index_col=0, encoding='utf-8')

    dwd_in_ppt_vals.index = pd.to_datetime(
        dwd_in_ppt_vals.index, format='%Y-%m-%d')

    dwd_in_ppt_vals = dwd_in_ppt_vals.loc[strt_date:end_date, :]
    dwd_in_ppt_vals.dropna(how='all', axis=0, inplace=True)

    # DWD Extremes
    #=========================================================================
    dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,
                                     index_col=0,
                                     sep=';',
                                     parse_dates=True,
                                     infer_datetime_format=True,
                                     encoding='utf-8',
                                     header=None)

    print('\n%d Extreme Event to interpolate\n' % dwd_in_extremes_df.shape[0])
    # shuffle and select 10 DWD stations randomly
    # =========================================================================
    all_dwd_stns = dwd_in_ppt_vals.columns.tolist()
    shuffle(all_dwd_stns)
    shuffled_dwd_stns_10stn = np.array(list(chunks(all_dwd_stns, 10)))

    #==========================================================================
    # CREATE DFS HOLD RESULT KRIGING PER NETATMO STATION
    #==========================================================================
    for idx_lst_comb in range(len(shuffled_dwd_stns_10stn)):
        stn_comb = shuffled_dwd_stns_10stn[idx_lst_comb]

        print('Interpolating for following DWD stations: \n',
              pprint.pformat(stn_comb))

        df_interpolated_dwd_only = pd.DataFrame(
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
                stn for stn in dwd_in_ppt_vals.columns if stn not in stn_comb]

            for event_date in dwd_in_extremes_df.index:
                #                 if event_date == '2016-08-18 20:00:00':
                #                     print(event_date)
                #                     raise Exception
                _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
                if len(_stn_id_event_) < 5:
                    _stn_id_event_ = (5 - len(_stn_id_event_)) * \
                        '0' + _stn_id_event_

                _ppt_event_ = dwd_in_extremes_df.loc[event_date, 1]
                _edf_event_ = dwd_in_ppt_vals.loc[event_date, _stn_id_event_]
                print('**Calculating for Date ',
                      event_date, '\n Rainfall: ',  _ppt_event_,
                      'Rainfall: ', _edf_event_, ' **\n')

                #==============================================================
                # # DWD qunatiles
                #==============================================================
                ppt_dwd_vals = []
                dwd_xcoords = []
                dwd_ycoords = []
                dwd_stn_ids = []

                for stn_id in all_dwd_stns_except_interp_loc:
                    #print('station is', stn_id)

                    ppt_stn_vals = dwd_in_ppt_vals.loc[event_date, stn_id]

                    if ppt_stn_vals >= 0:
                        ppt_dwd_vals.append(np.round(ppt_stn_vals, 4))
                        dwd_xcoords.append(dwd_in_coords_df.loc[stn_id, 'X'])
                        dwd_ycoords.append(dwd_in_coords_df.loc[stn_id, 'Y'])
                        dwd_stn_ids.append(stn_id)

                dwd_xcoords = np.array(dwd_xcoords)
                dwd_ycoords = np.array(dwd_ycoords)
                ppt_dwd_vals = np.array(ppt_dwd_vals)

                print('*Done getting data and coordintates* \n *Fitting variogram*\n')
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
                    #fit_vg_list = ['']
                    continue

                vgs_model_dwd = fit_vg_list[0]

                if ('Nug' in vgs_model_dwd or len(vgs_model_dwd) == 0) and (
                        'Exp' not in vgs_model_dwd and 'Sph' not in vgs_model_dwd):
                    print('**Variogram %s not valid --> looking for alternative\n**'
                          % vgs_model_dwd)
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
                    print('**Changed Variogram model to**\n', vgs_model_dwd)
                    print('\n+++ KRIGING +++\n')

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

                    print('**Interpolated DWD: ',
                          interpolated_vals_dwd_only)

                    if interpolated_vals_dwd_only < 0:
                        interpolated_vals_dwd_only = np.nan

                else:
                    print('no good variogram found, adding nans to df')
                    interpolated_vals_dwd_only = np.nan

                df_interpolated_dwd_only.loc[
                    event_date, stn_dwd_id] = interpolated_vals_dwd_only
        df_interpolated_dwd_only.dropna(how='all', inplace=True)

        df_interpolated_dwd_only.to_csv(out_plots_path / (
            'interpolated_ppt_dwd_%s_data_%s_using_dwd_only_grp_%d_.csv'
            % (temp_agg, title_, idx_lst_comb)),
            sep=';', float_format='%0.2f')

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s \a\a\a' %
          (time.asctime()))
