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
#import math
import fiona

import pandas as pd


from spinterps import (OrdinaryKriging)
from scipy import spatial
from scipy.spatial import cKDTree
from pathlib import Path


use_reduced_sample_dwd = False

neigbhrs_radius_dwd = 3e4
neigbhrs_radius_netatmo = 3e4
# =============================================================================

#main_dir = Path(r'X:\staff\elhachem\2020_10_03_Rheinland_Pfalz')

main_dir = Path(
    r'/run/media/abbas/EL Hachem 2019/home_office'
    r'/2020_10_03_Rheinland_Pfalz')

#main_dir = Path(r'/home/abbas/Documents/Python/Extremes')
# main_dir = Path(r'/home/IWS/hachem/Extremes')
os.chdir(main_dir)

path_to_data = main_dir

path_to_vgs = main_dir

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'dwd_coords_in_around_RH_utm32.csv')

path_to_netatmo_coords = (path_to_data /
                          r'netatmo_Rheinland-Pfalz_1hour_coords_utm32.csv')

# path for data filter
# in_filter_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

# path_to_dwd_stns_comb
path_to_dwd_stns_comb = main_dir / r'dwd_combination_to_use_RH_all_stns.csv'

# r'dwd_combination_to_use_RH.csv'  # dwd combinations leave 10 out

#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================

# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging


# if use_netatmo_gd_stns:
path_to_netatmo_gd_stns = (
    main_dir / r'indicator_correlation' /
    (r'keep_stns_99_0_per_60min_shift_10perc_10fact.csv'))

path_to_netatmo_gd_stns = (
   r"/run/media/abbas/EL Hachem 2019/home_office"
   r"/Data_Bardossy/AW__Online_Meetings_"
    r'/Good_Netatmo99.csv')  # Good
# keep_stns_99_0_per_60min_shift_10perc_10fact
# 99
#==============================================================================
#
#==============================================================================
resample_frequencies = ['60min']
# '120min', '180min', '60min',  '360min',
#                         '720min',
title_ = r'ppt_cross_valid_RH_'


#==============================================================================
#
#==============================================================================
strt_date = '2015-01-01 00:00:00'
end_date = '2019-12-31 00:00:00'


idx_time_fmt = '%Y-%m-%d %H:%M:%S'

radius = 1e4
diff_thr = 0.2
edf_thr = 0.9  # 0.9

hourly_events = ['2015-09-16 04:00:00',
                 '2015-09-16 06:00:00',
                 '2015-09-16 07:00:00',
                 '2015-09-16 11:00:00',
                 '2015-09-16 17:00:00',
                 '2015-09-16 18:00:00',
                 '2015-09-16 22:00:00',
                 '2015-09-17 00:00:00']


daily_events = ['2018-12-23 00:00:00',
                '2019-05-22 00:00:00',
                '2018-05-14 00:00:00',
                '2019-07-28 00:00:00']
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


# Netatmo first filter
df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
                         index_col=0,
                         sep=';',
                         encoding='utf-8')

# read df combinations to use
df_dwd_stns_comb = pd.read_csv(
    path_to_dwd_stns_comb, index_col=0,
    sep=',', dtype=str)

#===============================================================================
# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]
# 
#  
# # divide DWD stations into group of 10
# stns_dwd = dwd_in_coords_df.index.to_list()
# groups_of_10 = chunks(l=stns_dwd, n=1)
# grp = [gr for gr in groups_of_10]
#  
# df_dwd_group_stns = pd.DataFrame(index=range(len(grp)),
#                                  data=grp)
# df_dwd_group_stns.to_csv(main_dir / 'dwd_combination_to_use_RH_all_stns.csv',
#                          sep=',')
# pass
#===============================================================================
#==============================================================================
# NEEDED FUNCTIONS
#==============================================================================


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
#
#==============================================================================
for temp_agg in resample_frequencies:
    
    # out path directory
    
    dir_path = title_ + temp_agg

    #dir_path = title_ + temp_agg
    out_plots_path = main_dir / dir_path

    if not os.path.exists(out_plots_path):
        os.mkdir(out_plots_path)
    print(out_plots_path)

    # path to data
    #=========================================================================
    #out_save_csv = '_%s_ppt_edf_' % temp_agg

    path_to_dwd_edf = (path_to_data /
                       (r'edf_ppt_all_dwd_%s_.csv' % temp_agg))

    path_to_dwd_ppt = (path_to_data /
                       (r'ppt_dwd_2014_2019_%s_no_freezing_5deg.csv'
                           % temp_agg))

    path_to_netatmo_edf = (
        path_to_data /
        (r'edf_ppt_all_netatmo_%s_.csv' % temp_agg))

    path_to_netatmo_ppt = (
        path_to_data /
        (r'ppt_all_netatmo_rh_2014_2019_%s_no_freezing_5deg.csv'
            % temp_agg))

    # TODO: what to change
    path_to_dwd_vgs = (
        path_to_vgs /
        ('vg_strs_max100_hours_%s2.csv' % temp_agg))
    #         ('vg_strs_special_events_%s.csv' % temp_agg))
#         (r'vg_strs_max100_hours_%s.csv' % temp_agg))

#     path_to_dwd_vgs = (
#         r"X:\exchange\ElHachem\Events_HBV\Echaz\df_vgs_events2.csv")

    path_dwd_extremes_df = (
        r"/run/media/abbas/EL Hachem 2019/home_office/Data_Bardossy/EventsRLP.csv") 

#     path_dwd_extremes_df = path_to_data / \
#         (r'dwd_%s_maximum_100_hours.csv' % temp_agg)
        
        # max_100_%s_events_dwd
#         (r'dwd_%s_maximum_100_hours.csv' % temp_agg)
#         (r'dwd_%s_special_events_10mm_.csv' % temp_agg)
    #(r'dwd_%s_maximum_100_hours.csv' % temp_agg)
    # Files to use
    # =========================================================================

    netatmo_data_to_use = path_to_netatmo_edf
    dwd_data_to_use = path_to_dwd_edf

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


    # NETAMO DATA
    #=========================================================================
    netatmo_in_vals_df = pd.read_csv(path_to_netatmo_edf,
                                     sep=';',
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
        encoding='utf-8',
        engine='c')

    netatmo_in_ppt_vals_df.index = pd.to_datetime(
        netatmo_in_ppt_vals_df.index,
        format='%Y-%m-%d')

    netatmo_in_ppt_vals_df.dropna(how='all', axis=0, inplace=True)

    netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.loc[:, cmn_stns]
    
    #==========================================================================
    # # shift data
    #==========================================================================
    netatmo_in_vals_df = netatmo_in_vals_df.shift(1)
    netatmo_in_ppt_vals_df = netatmo_in_ppt_vals_df.shift(1)
    
    # #####
    # good_netatmo_stns = df_gd_stns.loc[
    #    :, 'Stations'].values.ravel()
    
    good_netatmo_stns = df_gd_stns.index
        
    cmn_gd_stns = netatmo_in_vals_df.columns.intersection(
        good_netatmo_stns)
    
    netatmo_in_vals_df_gd = netatmo_in_vals_df.loc[
        :, cmn_gd_stns]
    
    netatmo_in_ppt_vals_df_gd = netatmo_in_ppt_vals_df.loc[:, cmn_gd_stns]
    
    # DWD Extremes
    #=========================================================================
    dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,  # path_dwd_extremes_df
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
    # get the highest 100 event
    # TODO: no hardcoding
    dwd_in_extremes_df = dwd_in_extremes_df.loc[strt_date:end_date, :]
    dwd_in_extremes_df= dwd_in_extremes_df.sort_values(
        by=[1], ascending=False)  # [:100]
    
    # empty df for saving resuls
    data_arr = np.zeros(shape=(len(dwd_in_extremes_df.index),
                               len(netatmo_in_ppt_vals_df_gd.columns)))
    data_arr[data_arr == 0] = np.nan
    
    netatmo_in_ppt_vals_df_gd_corr = pd.DataFrame(
        index=dwd_in_extremes_df.index, data=data_arr, 
        columns= netatmo_in_ppt_vals_df_gd.columns)
    

    # number events with good vg
    evts = []
    df_vgs = pd.DataFrame(index=dwd_in_extremes_df.index,
                          columns=['vg_model'])
    for ix in df_vgs_extremes.index:
        for val in df_vgs_extremes.loc[ix, :].values:
            if isinstance(val, str):
                if 'Nug' not in val:
                    df_vgs.loc[ix, 'vg_model'] = val
                    break
    df_vgs.dropna(how='all', inplace=True)

    dwd_in_extremes_df = dwd_in_extremes_df.loc[
        dwd_in_extremes_df.index.intersection(
            df_vgs.index), :]
    
    print('\n%d Intense Event with gd VG to interpolate\n'
          % dwd_in_extremes_df.shape[0])
    dwd_in_extremes_df = dwd_in_extremes_df.sort_index()
    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()

    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    for iev, event_date in enumerate(dwd_in_extremes_df.index):
        # break
        print(event_date, '---', iev ,'/', len(dwd_in_extremes_df.index))
        # _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
        _ppt_event_ = float(dwd_in_extremes_df.loc[event_date, :])  # 1
        
        # start cross validating DWD stations for this event
        
        
        obs_ppt_stn_dwd = dwd_in_ppt_vals_df.loc[
            event_date, all_dwd_stns].dropna()

        # coords of all other stns
        x_dwd_all = dwd_in_coords_df.loc[
            dwd_in_coords_df.index.intersection(
                obs_ppt_stn_dwd.index),
            'X'].values
        y_dwd_all = dwd_in_coords_df.loc[
            dwd_in_coords_df.index.intersection(
                obs_ppt_stn_dwd.index),
            'Y'].values

        # get vg model for this day
        vgs_model_dwd_ppt = df_vgs.loc[event_date, 'vg_model']

        vg_sill = float(vgs_model_dwd_ppt.split(" ")[0])
        dwd_vals_var = np.var(obs_ppt_stn_dwd.values)
        vg_scaling_ratio = dwd_vals_var / vg_sill

        if vg_scaling_ratio == 0:
            vg_scaling_ratio = 1
        
        #print(vgs_model_dwd_ppt)
        
        dwd_xcoords = np.array(x_dwd_all)
        dwd_ycoords = np.array(y_dwd_all)
        ppt_dwd_vals = np.array(obs_ppt_stn_dwd.values)
        edf_dwd_vals = np.array(dwd_in_vals_df.loc[
            event_date, all_dwd_stns].dropna())
        #==========================================================
        # FIRST AND SECOND FILTER
        #==========================================================
        
        netatmo_df_gd = netatmo_in_ppt_vals_df_gd.loc[
            event_date, :].dropna(how='all')
        netatmo_edf = netatmo_in_vals_df_gd.loc[
            event_date, :].dropna(how='all')
        
        netatmo_xcoords = netatmo_in_coords_df.loc[
            netatmo_edf.index, 'X'].values.ravel()
        netatmo_ycoords = netatmo_in_coords_df.loc[
            netatmo_edf.index, 'Y'].values.ravel()

        #===============================================================
        # # apply on event filter
        #===============================================================
        #print('NETATMO 2nd Filter')
        
        ordinary_kriging_filter_netamto = OrdinaryKriging(
            xi=dwd_xcoords,
            yi=dwd_ycoords,
            zi=edf_dwd_vals,
            xk=netatmo_xcoords,
            yk=netatmo_ycoords,
            model=vgs_model_dwd_ppt)

        try:
            ordinary_kriging_filter_netamto.krige()
        except Exception as msg:
            print('Error while Error Kriging', msg)

        # interpolated vals
        interpolated_vals = ordinary_kriging_filter_netamto.zk

        # calcualte standard deviation of estimated values
        std_est_vals = np.sqrt(
            ordinary_kriging_filter_netamto.est_vars)
        # calculate difference observed and estimated  # values
        try:
            diff_obsv_interp = np.abs(
            netatmo_edf.values - interpolated_vals)
        except Exception:
            print('ERROR 2nd FILTER')
            
        idx_good_stns = np.where(
            diff_obsv_interp <= 3 * std_est_vals)
        idx_bad_stns = np.where(
            diff_obsv_interp > 3 * std_est_vals)

        if len(idx_bad_stns[0]) or len(idx_good_stns[0]) > 0:
#             print('Number of Stations with bad index \n',
#                   len(idx_bad_stns[0]))
#             print('Number of Stations with good index \n',
#                  len(idx_good_stns[0]))

            try:
                ids_netatmo_stns_gd = np.take(netatmo_df_gd.index,
                                              idx_good_stns).ravel()
                ids_netatmo_stns_bad = np.take(netatmo_df_gd.index,
                                               idx_bad_stns).ravel()

            except Exception as msg:
                print(msg)
        
        assert idx_good_stns[0].size == ids_netatmo_stns_gd.size
        assert idx_bad_stns[0].size == ids_netatmo_stns_bad.size
        
        try:
            edf_gd_vals_df = netatmo_edf.loc[ids_netatmo_stns_gd]
        except Exception as msg:
            print(msg, 'error while second filter')

        netatmo_dry_gd = edf_gd_vals_df[edf_gd_vals_df.values < edf_thr]
        # gd

        cmn_wet_dry_stns = netatmo_in_coords_df.index.intersection(
            netatmo_dry_gd.index)
        x_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
            cmn_wet_dry_stns, 'X'].values.ravel()
        y_coords_gd_netatmo_dry = netatmo_in_coords_df.loc[
            cmn_wet_dry_stns, 'Y'].values.ravel()

        netatmo_wet_gd = edf_gd_vals_df[
            edf_gd_vals_df.values >= edf_thr]
        cmn_wet_gd_stns = netatmo_in_coords_df.index.intersection(
            netatmo_wet_gd.index)

        x_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
            cmn_wet_gd_stns, 'X'].values.ravel()
        y_coords_gd_netatmo_wet = netatmo_in_coords_df.loc[
            cmn_wet_gd_stns, 'Y'].values.ravel()

        assert (netatmo_wet_gd.size ==
                x_coords_gd_netatmo_wet.size ==
                y_coords_gd_netatmo_wet.size)

        edf_bad_vals_df = netatmo_edf.loc[ids_netatmo_stns_bad]

        netatmo_dry_bad = edf_bad_vals_df[
            edf_bad_vals_df.values < edf_thr]
        cmn_dry_bad_stns = netatmo_in_coords_df.index.intersection(
            netatmo_dry_bad.index)
        # netatmo bad
        netatmo_wet_bad = edf_bad_vals_df[
            edf_bad_vals_df.values >= edf_thr]
        cmn_wet_bad_stns = netatmo_in_coords_df.index.intersection(
            netatmo_wet_bad.index)
        # dry bad
        x_coords_bad_netatmo_dry = netatmo_in_coords_df.loc[
            cmn_dry_bad_stns, 'X'].values.ravel()
        y_coords_bad_netatmo_dry = netatmo_in_coords_df.loc[
            cmn_dry_bad_stns, 'Y'].values.ravel()
        # wet bad
        x_coords_bad_netatmo_wet = netatmo_in_coords_df.loc[
            cmn_wet_bad_stns, 'X'].values.ravel()
        y_coords_bad_netatmo_wet = netatmo_in_coords_df.loc[
            cmn_wet_bad_stns, 'Y'].values.ravel()

        # DWD data
        dwd_dry = edf_dwd_vals[edf_dwd_vals < edf_thr]
        dwd_wet = edf_dwd_vals[edf_dwd_vals >= edf_thr]

        x_coords_dwd_wet = dwd_xcoords[np.where(
            edf_dwd_vals >= edf_thr)]
        y_coords_dwd_wet = dwd_ycoords[np.where(
            edf_dwd_vals >= edf_thr)]

        assert (dwd_wet.size ==
                x_coords_dwd_wet.size ==
                y_coords_dwd_wet.size)

        x_coords_dwd_dry = dwd_xcoords[np.where(
            edf_dwd_vals < edf_thr)]
        y_coords_dwd_dry = dwd_ycoords[np.where(
            edf_dwd_vals < edf_thr)]

        # find if wet bad is really wet bad
        # find neighboring netatmo stations wet good

        # combine dwd and netatmo wet gd

        x_gd_dwd_netatmo = np.concatenate([
            x_coords_gd_netatmo_wet,
            x_coords_dwd_wet])
        y_gd_dwd_netatmo = np.concatenate([
            y_coords_gd_netatmo_wet,
            y_coords_dwd_wet])

        dwd_netatmo_wet_gd = np.concatenate([
            netatmo_wet_gd,
            dwd_wet])
        
        assert (netatmo_wet_gd.size + netatmo_wet_bad.size + 
                netatmo_dry_gd.size + netatmo_dry_bad.size
                ) == netatmo_df_gd.size
        
        # check if dry stns are really dry
        if netatmo_wet_gd.size > 0:

            for stn_, edf_stn, netatmo_x_stn, netatmo_y_stn in zip(
                netatmo_wet_bad.index,
                netatmo_wet_bad.values,
                    x_coords_bad_netatmo_wet,
                    y_coords_bad_netatmo_wet):

                # coords of stns self
                stn_coords = np.array([(netatmo_x_stn,
                                        netatmo_y_stn)])

                # coords of neighbors
                neighbors_coords = np.array(
                    [(x, y) for x, y
                     in zip(x_gd_dwd_netatmo,
                            y_gd_dwd_netatmo)])

                # create a tree from coordinates
                points_tree = spatial.KDTree(
                    neighbors_coords)

                # This finds the index of all points within
                # radius
                idxs_neighbours = points_tree.query_ball_point(
                    np.array(
                        (netatmo_x_stn, netatmo_y_stn)),
                    1e4)
                if len(idxs_neighbours) > 0:

                    for i, ix_nbr in enumerate(idxs_neighbours):

                        try:
                            edf_neighbor = dwd_netatmo_wet_gd[ix_nbr]
                        except Exception as msg:
                            print(msg)
                            edf_neighbor = 1.1
                        if np.abs(edf_stn - edf_neighbor) <= diff_thr:

                            try:
                                netatmo_wet_gd[stn_] = edf_stn
                                
                                if stn_ not in ids_netatmo_stns_gd:
                                    
                                    ids_netatmo_stns_gd = np.append(
                                        ids_netatmo_stns_gd,
                                        stn_)
                                    x_coords_gd_netatmo_wet = np.append(
                                        x_coords_gd_netatmo_wet,
                                        netatmo_x_stn)
                                    y_coords_gd_netatmo_wet = np.append(
                                        y_coords_gd_netatmo_wet,
                                        netatmo_y_stn)
    
                                    # remove from original bad
                                    # wet
                                    x_coords_bad_netatmo_wet = np.array(list(
                                        filter(lambda x: x != netatmo_x_stn,
                                               x_coords_bad_netatmo_wet)))
    
                                    y_coords_bad_netatmo_wet = np.array(list(
                                        filter(lambda x: x != netatmo_y_stn,
                                               y_coords_bad_netatmo_wet)))
    
                                    netatmo_wet_bad.loc[stn_] = np.nan
    
                                    ids_netatmo_stns_bad = list(
                                        filter(lambda x: x != stn_,
                                               ids_netatmo_stns_bad))
                            except Exception as msg:
                                print(msg)
                                pass

                        else:
                            pass
                            print('bad wet is bad wet')
                else:
                    pass
                    # print('\nStn has no near neighbors')
        print('Number of Stations with bad index \n',
                  len(ids_netatmo_stns_bad), '/', len(netatmo_df_gd.index))
        print('Number of Stations with good index \n',
                 len(ids_netatmo_stns_gd), '/', len(netatmo_df_gd.index))
        
        try:
            assert len(netatmo_df_gd.index) == (len(ids_netatmo_stns_bad) + 
                                                len(ids_netatmo_stns_gd))
        except Exception as msg:
            print(msg)
        # rescale variogram
        vgs_model_dwd_ppt = str(
            np.round(vg_scaling_ratio, 4)
            ) + ' ' + vgs_model_dwd_ppt.split(" ")[1]
            
        netatmo_stns_event_gd = []
        netatmo_ppt_vals_fr_dwd_interp_gd = []
        x_netatmo_ppt_vals_fr_dwd_interp_gd = []
        y_netatmo_ppt_vals_fr_dwd_interp_gd = []
        
        # ppt data before correction
        
        netatmo_ppt_not_corrected = netatmo_in_ppt_vals_df_gd.loc[
            event_date, ids_netatmo_stns_gd].values
        #print('correcting Netatmo Quantiles')
        for netatmo_stn_id in ids_netatmo_stns_gd:  # netatmo_df.index:

            netatmo_edf_event_ = netatmo_in_vals_df_gd.loc[event_date,
                                                           netatmo_stn_id]

            netatmo_ppt_event_ = netatmo_in_ppt_vals_df_gd.loc[
                event_date,
                netatmo_stn_id]

            x_netatmo_interpolate = np.array(
                [netatmo_in_coords_df.loc[netatmo_stn_id, 'X']])
            y_netatmo_interpolate = np.array(
                [netatmo_in_coords_df.loc[netatmo_stn_id, 'Y']])

            if netatmo_edf_event_ > 0.5:  # 0.99:
                try:

                    # Correct Netatmo Quantiles

                    netatmo_stn_edf_df = netatmo_in_vals_df_gd.loc[
                        :, netatmo_stn_id].dropna()

                    netatmo_start_date = netatmo_stn_edf_df.index[0]
                    netatmo_end_date = netatmo_stn_edf_df.index[-1]

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

                            if netatmo_edf_event_ == 1.0:
                                # correct edf event, 1 is bcz
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
                                print(msg)

                            if ppt_for_edf[0] >= 0:

                                ppt_vals_dwd_new_for_obsv_ppt.append(
                                    ppt_for_edf[0])

                                stn_xcoords_new = dwd_in_coords_df.loc[
                                    dwd_stn_id_new, 'X']
                                stn_ycoords_new = dwd_in_coords_df.loc[
                                    dwd_stn_id_new, 'Y']

                                dwd_xcoords_new.append(
                                    stn_xcoords_new)
                                dwd_ycoords_new.append(
                                    stn_xcoords_new)

                    # get ppt from dwd recent for netatmo stns
                    ppt_vals_dwd_new_for_obsv_ppt = np.array(
                        ppt_vals_dwd_new_for_obsv_ppt)
                    # for kriging with uncertainty

                    dwd_xcoords_new = np.array(dwd_xcoords_new)
                    dwd_ycoords_new = np.array(dwd_ycoords_new)

                    try:

                        ordinary_kriging_dwd_netatmo_crt = OrdinaryKriging(
                            xi=dwd_xcoords_new,
                            yi=dwd_ycoords_new,
                            zi=ppt_vals_dwd_new_for_obsv_ppt,
                            xk=x_netatmo_interpolate,
                            yk=y_netatmo_interpolate,
                            model=vgs_model_dwd_ppt)

                        try:
                            ordinary_kriging_dwd_netatmo_crt.krige()
                        except Exception as msg:
                            print('Error while Kriging', msg)

                        interpolated_netatmo_prct = ordinary_kriging_dwd_netatmo_crt.zk.copy()

                        if interpolated_netatmo_prct < 0:
                            interpolated_netatmo_prct = np.nan

                        if interpolated_netatmo_prct >= 0.:
                            netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                interpolated_netatmo_prct[0])

                            x_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                x_netatmo_interpolate[0])

                            y_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                                y_netatmo_interpolate[0])

                            netatmo_stns_event_gd.append(
                                netatmo_stn_id)
                    except Exception as msg:
                        print(
                            msg,
                             'Error when getting ppt from dwd interp')
                        continue

                except Exception as msg:
                    print(msg, 'Error when KRIGING')
                    continue

            else:
                # edf too small no need to correct it
                if netatmo_ppt_event_ >= 0:
                    netatmo_ppt_vals_fr_dwd_interp_gd.append(
                        netatmo_ppt_event_)

                    x_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                        x_netatmo_interpolate[0])

                    y_netatmo_ppt_vals_fr_dwd_interp_gd.append(
                        y_netatmo_interpolate[0])

                    netatmo_stns_event_gd.append(netatmo_stn_id)
        
        # Transform everything to arrays 
        netatmo_xcoords = np.array(
            x_netatmo_ppt_vals_fr_dwd_interp_gd).ravel()
        netatmo_ycoords = np.array(
            y_netatmo_ppt_vals_fr_dwd_interp_gd).ravel()

        ppt_netatmo_vals_gd = np.round(np.array(
            netatmo_ppt_vals_fr_dwd_interp_gd).ravel(), 2)
        
        print('saving stations:', ppt_netatmo_vals_gd.size)
        #======================================================
        # # SAVING PPT
        #======================================================
        netatmo_in_ppt_vals_df_gd_corr.loc[
            event_date, netatmo_stns_event_gd] = ppt_netatmo_vals_gd
netatmo_in_ppt_vals_df_gd_corr.dropna(how='all', inplace=True)           
netatmo_in_ppt_vals_df_gd_corr.to_csv(main_dir / (
        'ppt_all_netatmo_100_intense_events_corrected_99_gd199_%s.csv'
        % (temp_agg)), sep=';', float_format='%0.2f')



stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
