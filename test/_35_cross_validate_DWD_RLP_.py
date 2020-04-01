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

import pandas as pd
import matplotlib.pyplot as plt

from spinterps import (OrdinaryKriging)
from scipy import spatial
from scipy.spatial import distance
from pathlib import Path


neigbhrs_radius_dwd = 3e4
neigbhrs_radius_netatmo = 2e4
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

path_to_dwd_coords_on_in_rh = (path_to_data /
                               r'dwd_coords_in_around_RH_utm32.csv')
                      # r'dwd_coords_only_in_RH.csv')

path_to_netatmo_coords = (path_to_data /
                          r'netatmo_Rheinland-Pfalz_1hour_coords_utm32.csv')

# path for data filter
# in_filter_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

# path_to_dwd_stns_comb
path_to_dwd_stns_comb = main_dir / r'dwd_combination_to_use_RH_all_stns.csv'
# r'dwd_combination_to_use_RH.csv'

#==============================================================================
# # NETATMO FIRST FILTER
#==============================================================================

# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging


# if use_netatmo_gd_stns:
#path_to_netatmo_gd_stns = (
#    main_dir / r'indicator_correlation' /
#    (r'keep_stns_99_0_per_60min_shift_10perc_3fact.csv'))
# 99
#==============================================================================
#
#==============================================================================
resample_frequencies = ['60min']
# '120min', '180min', '60min',  '360min',
#                         '720min',
title_ = r'ppt_cross_valid_RH_'

# for Netatmo good stations percentile_shiftpoly_shiftexp
used_data_acc = r'99_gd199'
#==============================================================================
#
#==============================================================================
strt_date = '2017-01-01 00:00:00'
end_date = '2019-12-31 00:00:00'


idx_time_fmt = '%Y-%m-%d %H:%M:%S'

radius = 1e4
diff_thr = 1
edf_thr = 0.7  # 0.9

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

dwd_in_coords_df_in_rh = pd.read_csv(path_to_dwd_coords_on_in_rh,
                                     index_col=0,
                                     sep=';',
                                     encoding='utf-8')
# Netatmo first filter    
#df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
#                         index_col=0,
#                         sep=';',
#                         encoding='utf-8')

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
# stns_dwd = dwd_in_coords_df_in_rh.index.to_list()
# groups_of_10 = chunks(l=stns_dwd, n=10)
# grp = [gr for gr in groups_of_10]
#    
# df_dwd_group_stns = pd.DataFrame(index=range(len(grp)),
#                                  data=grp)
# df_dwd_group_stns.to_csv(main_dir / 'dwd_combination_to_use_in_ar_RH.csv',
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
    
    path_to_netatmo_coorected_ppt = (
        main_dir / (
            r'ppt_all_netatmo_100_intense_events_corrected_%s_%s.csv'
                    % (used_data_acc, temp_agg)))
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
#         (r'dwd_%s_maximum_100_hours.csv' % temp_agg)
#         (r'dwd_%s_special_events_10mm_.csv' % temp_agg)
    #(r'dwd_%s_maximum_100_hours.csv' % temp_agg)

    #df_radar_events_to_keep = pd.read_csv(
    #    r'X:\staff\elhachem\2020_10_03_Rheinland_Pfalz'
        #         r'\%s_max_events_radolan_files.csv'
        #         r'\%s_intense_events_radolan_files.csv'
    #    r'\%s_intense_events_5mm_radolan_files.csv'
    #    % (temp_agg), index_col=0,
    #    sep=';', engine='c',
    #    parse_dates=True,
    #    infer_datetime_format=True)

    # RADOLAN
    #base_path = (r'X:\exchange\ElHachem\radolan_%s_data\*' % temp_agg)

    #files = glob.glob(base_path)

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
    
    # corrected Netatmo data
    path_to_netatmo_gd_stns = (
           r"/run/media/abbas/EL Hachem 2019/home_office"
           r"/Data_Bardossy/AW__Online_Meetings_"
            r'/Good_Netatmo99.csv')  # Good
    df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
                         index_col=0,
                         sep=';',
                         encoding='utf-8')
    
    # netatmo_in_ppt_vals_df_gd = netatmo_in_ppt_vals_df
    
    good_netatmo_stns = df_gd_stns.index
        
    cmn_gd_stns = netatmo_in_vals_df.columns.intersection(
        good_netatmo_stns)
    netatmo_in_vals_df_gd = netatmo_in_vals_df.loc[
        :, cmn_gd_stns]
    netatmo_in_ppt_vals_df_gd = netatmo_in_ppt_vals_df.loc[:, cmn_gd_stns]
    
#     netatmo_in_ppt_vals_df_gd = pd.read_csv(
#         path_to_netatmo_coorected_ppt,
#         sep=';', index_col=0,
#         parse_dates=True,
#         infer_datetime_format=True)
    
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
            df_vgs.index).intersection(
                netatmo_in_ppt_vals_df_gd.index), :]
    
    print('\n%d Intense Event with gd VG to interpolate\n'
          % dwd_in_extremes_df.shape[0])
    dwd_in_extremes_df = dwd_in_extremes_df.sort_index()
    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    all_dwd_stns = dwd_in_vals_df.columns.tolist()
        
    # CREATE DFS FOR RESULT; Index is Date, Columns as Stns
    df_interpolated_dwd_only = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)
    
    df_interpolated_netatmo_only = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)
    
    df_interpolated_dwd_netatmos_comb = pd.DataFrame(
        index=dwd_in_extremes_df.index,
        columns=all_dwd_stns)
    
    #===========================================================================
    # 
    #===========================================================================
    # Netatmo COORDS
#     netatmo_all_stns = netatmo_in_ppt_vals_df.columns
#     netatmo_neighbors_coords = np.array(
#                 [(x, y) for x, y
#                  in zip(netatmo_in_coords_df.loc[:, 'X'].values,
#                         netatmo_in_coords_df.loc[:, 'Y'].values)])
#     
#     # create a tree from coordinates
#     netatmo_points_tree = spatial.KDTree(netatmo_neighbors_coords)
    #==========================================================================
    # # Go thourgh events ,interpolate all DWD for this event
    #==========================================================================
    for iev, event_date in enumerate(dwd_in_extremes_df.index):

        print(event_date, '---', iev ,'/', len(dwd_in_extremes_df.index))
        # _stn_id_event_ = str(dwd_in_extremes_df.loc[event_date, 2])
        _ppt_event_ = dwd_in_extremes_df.loc[event_date, :]
        
        # start cross validating DWD stations for this event
        for idx_lst_comb in df_dwd_stns_comb.index:

            stn_comb = [stn.replace("'", "")
                        for stn in df_dwd_stns_comb.iloc[
                        idx_lst_comb, :].dropna().values]
            
        
            obs_ppt_stn_dwd = dwd_in_ppt_vals_df.loc[
                event_date, stn_comb]
            
            x_dwd_interpolate = np.array(
                    dwd_in_coords_df.loc[stn_comb, 'X'].values)
            y_dwd_interpolate = np.array(
                    dwd_in_coords_df.loc[stn_comb, 'Y'].values)
            
#             xy_dwd_interp = np.array([x_dwd_interpolate[0],
#                                        y_dwd_interpolate[0]])
            
            # drop stns
            all_dwd_stns_except_interp_loc = [
                stn for stn in dwd_in_vals_df.columns
                if stn not in stn_comb]
            
            # ppt at dwd all other stns for event
            
            ppt_dwd_vals_sr = dwd_in_ppt_vals_df.loc[
                event_date,
                all_dwd_stns_except_interp_loc].dropna(how='all')
                            
            # coords of all other stns for event
            x_dwd_all = dwd_in_coords_df.loc[
                dwd_in_coords_df.index.intersection(
                    ppt_dwd_vals_sr.index),
                'X'].values
            y_dwd_all = dwd_in_coords_df.loc[
                dwd_in_coords_df.index.intersection(
                    ppt_dwd_vals_sr.index),
                'Y'].values
            # stns for this event
            stn_dwd_all = dwd_in_coords_df.loc[
                dwd_in_coords_df.index.intersection(
                    ppt_dwd_vals_sr.index), :].index
            #===================================================================
            # # GET nearest DWD stations
            #===================================================================

            # coords of neighbors
            dwd_neighbors_coords = np.array(
                [(x, y) for x, y
                 in zip(x_dwd_all,
                        y_dwd_all)])

            # create a tree from coordinates
            points_tree = spatial.KDTree(dwd_neighbors_coords)

            # This finds the index of all points within radius
            
            dwd_idxs_neighbours = np.unique([
                ix for idx in [points_tree.query_ball_point(
                np.array((x_interp, y_interp)),
                neigbhrs_radius_dwd) for x_interp, y_interp in zip(
                    x_dwd_interpolate.flatten(),
                    y_dwd_interpolate.flatten())]
                for ix in idx])
            
            stn_dwd_all_ngbrs = stn_dwd_all[dwd_idxs_neighbours]

            
            # ppt dwd vals neighbors
            ppt_dwd_vals_nona = dwd_in_ppt_vals_df.loc[
                event_date,
                stn_dwd_all_ngbrs].dropna().values
            # edf dwd vals neighbors
            edf_dwd_vals = dwd_in_vals_df.loc[
                event_date,
                stn_dwd_all_ngbrs].dropna().values

            x_dwd_all_ngbrs = dwd_in_coords_df.loc[stn_dwd_all_ngbrs, 'X'].values
            y_dwd_all_ngbrs = dwd_in_coords_df.loc[stn_dwd_all_ngbrs, 'Y'].values
            #===================================================================
#             plt.ioff()
#             plt.scatter(x_dwd_all,y_dwd_all, c='g', label='dwd all') 
#             plt.scatter(x_dwd_all_ngbrs,y_dwd_all_ngbrs, c='b', label='dwd ngbrs')
#             plt.scatter(x_dwd_interpolate,y_dwd_interpolate, c='r', label='dwd interp')
#             plt.legend(loc=0)       
#             plt.show()
            #===================================================================
            
            #===================================================================
            # # GET nearest NETATMO stations
            #===================================================================

            # ppt data at other NETATMO stations
            ppt_netatmo_vals_sr = netatmo_in_ppt_vals_df.loc[
                event_date, :].dropna()
                
            x_netatmo_all = netatmo_in_coords_df.loc[
                ppt_netatmo_vals_sr.index, 'X'].values
            y_netatmo_all = netatmo_in_coords_df.loc[
                ppt_netatmo_vals_sr.index, 'Y'].values
                
            # coords of neighbors
            netatmo_neighbors_coords = np.array(
                [(x, y) for x, y
                 in zip(x_netatmo_all,
                        y_netatmo_all)])

            # create a tree from coordinates
            netatmo_points_tree = spatial.KDTree(netatmo_neighbors_coords)
            
            # This finds the index of all points within radius
            
            netatmo_idxs_neighbours = np.unique([
                ix for idx in [netatmo_points_tree.query_ball_point(
                np.array((x_interp, y_interp)),
                neigbhrs_radius_netatmo) for x_interp, y_interp in zip(
                    x_dwd_interpolate.flatten(),
                    y_dwd_interpolate.flatten())]
                for ix in idx])
            
            if netatmo_idxs_neighbours.size < 10:
                neigbhrs_radius_netatmo = 3e4
                netatmo_idxs_neighbours = np.unique([
                ix for idx in [netatmo_points_tree.query_ball_point(
                np.array((x_interp, y_interp)),
                neigbhrs_radius_netatmo) for x_interp, y_interp in zip(
                    x_dwd_interpolate.flatten(),
                    y_dwd_interpolate.flatten())]
                for ix in idx])
                
            stn_netatmo_all_ngbrs = ppt_netatmo_vals_sr.index[
                netatmo_idxs_neighbours]
            
            ppt_netatmo_vals_nona = netatmo_in_ppt_vals_df.loc[
                event_date, stn_netatmo_all_ngbrs].dropna().values

            # edf netatmo vals
            edf_netatmo_vals = netatmo_in_vals_df.loc[
                event_date, stn_netatmo_all_ngbrs].dropna().values

            x_netatmo_all_ngbrs = netatmo_in_coords_df.loc[
                stn_netatmo_all_ngbrs, 'X'].values
            y_netatmo_all_ngbrs = netatmo_in_coords_df.loc[
                stn_netatmo_all_ngbrs, 'Y'].values
            
#             # find the closest neighbors, 25 first
#             netatmo_xy_coords = np.array([(x, y) for x, y in zip(
#                 x_netatmo_all_ngbrs, y_netatmo_all_ngbrs)])
#             
#             ngbrs_dist = distance.cdist([(x_dwd_interpolate[0],
#                                           y_dwd_interpolate[0])],
#                                         netatmo_xy_coords,
#                                         'euclidean')
            
            #===================================================================
#             plt.ioff()           
#             plt.scatter(x_netatmo_all,
#                         y_netatmo_all,
#                         c='g', label='netatmo all')  
#             plt.scatter(x_netatmo_all_ngbrs,y_netatmo_all_ngbrs, c='b',
#                         label='netatmo ngbrs') 
#             plt.scatter(x_dwd_interpolate,y_dwd_interpolate, c='r',
#                         label='dwd interp')    
#             plt.show()
            #===================================================================
                        
            # get vg model for this day
            vgs_model_dwd_ppt = df_vgs.loc[event_date, 'vg_model']

            vg_sill = float(vgs_model_dwd_ppt.split(" ")[0])
            dwd_vals_var = np.var(ppt_dwd_vals_nona)
            vg_scaling_ratio = dwd_vals_var / vg_sill

            if vg_scaling_ratio == 0:
                vg_scaling_ratio = 1
            # rescale variogram
            vgs_model_dwd_ppt = str(
                np.round(vg_scaling_ratio, 4)
                ) + ' ' + vgs_model_dwd_ppt.split(" ")[1]
            #print(vgs_model_dwd_ppt)
            
            dwd_xcoords = np.array(x_dwd_all_ngbrs)
            dwd_ycoords = np.array(y_dwd_all_ngbrs)
            ppt_dwd_vals = np.array(ppt_dwd_vals_nona)

            #==========================================================
            # NO FILTER USED
            #==========================================================

            #print('\n**NETATMO NOT FILTERED**')
            netatmo_ppt_vals_fr_dwd_interp = np.array(
                ppt_netatmo_vals_nona)
 
            x_netatmo_ppt_vals_fr_dwd_interp = np.array(
                x_netatmo_all_ngbrs)
            y_netatmo_ppt_vals_fr_dwd_interp = np.array(
                y_netatmo_all_ngbrs)
            #======================================================
            # # Krigging PPT
            #======================================================
             
            # using DWD data
            ordinary_kriging_dwd_ppt = OrdinaryKriging(
                xi=dwd_xcoords,
                yi=dwd_ycoords,
                zi=ppt_dwd_vals,
                xk=x_dwd_interpolate,
                yk=y_dwd_interpolate,
                model=vgs_model_dwd_ppt)
 
            # using Netatmo data
            ordinary_kriging_netatmo_ppt = OrdinaryKriging(
                xi=x_netatmo_ppt_vals_fr_dwd_interp,
                yi=y_netatmo_ppt_vals_fr_dwd_interp,
                zi=netatmo_ppt_vals_fr_dwd_interp,
                xk=x_dwd_interpolate,
                yk=y_dwd_interpolate,
                model=vgs_model_dwd_ppt)
 
            # print('\nOK using DWD')
            ordinary_kriging_dwd_ppt.krige()
 
            # print('\nOK using Netatmo')
            ordinary_kriging_netatmo_ppt.krige()
 
            interpolated_vals_dwd_only = ordinary_kriging_dwd_ppt.zk.copy()
            interpolated_vals_netatmo_only = ordinary_kriging_netatmo_ppt.zk.copy()
 
            interpolated_vals_dwd_only[
                interpolated_vals_dwd_only < 0] = 0
 
            interpolated_vals_netatmo_only[
                interpolated_vals_netatmo_only < 0] = 0

            #==========================================================
            # FIRST AND SECOND FILTER
            #==========================================================

            #print('\n**using Netatmo gd stns**')
            # netatmo neighbors gd
            netatmo_gd_stns_evt = netatmo_in_ppt_vals_df_gd.columns.intersection(
                stn_netatmo_all_ngbrs)
            
            netatmo_df_gd = netatmo_in_ppt_vals_df_gd.loc[
                event_date, netatmo_gd_stns_evt].dropna(how='all')            

            netatmo_xcoords = netatmo_in_coords_df.loc[
                netatmo_df_gd.index, 'X'].values.ravel()
            netatmo_ycoords = netatmo_in_coords_df.loc[
                netatmo_df_gd.index, 'Y'].values.ravel()

            ppt_netatmo_vals_gd = np.round(np.array(
                netatmo_df_gd.values).ravel(), 2)

            netatmo_dwd_x_coords = np.concatenate([netatmo_xcoords,
                                                   dwd_xcoords])
            netatmo_dwd_y_coords = np.concatenate([netatmo_ycoords,
                                                   dwd_ycoords])
            netatmo_dwd_ppt_vals_gd = np.round(np.hstack(
                (ppt_netatmo_vals_gd,
                 ppt_dwd_vals)), 2).ravel()

            #print('\n+*-KRIGING WITH 1st and 2nd Filter-*+')
            ordinary_kriging_dwd_netatmo_ppt = OrdinaryKriging(
                xi=netatmo_dwd_x_coords,
                yi=netatmo_dwd_y_coords,
                zi=netatmo_dwd_ppt_vals_gd,
                xk=x_dwd_interpolate,
                yk=y_dwd_interpolate,
                model=vgs_model_dwd_ppt)

            try:
                ordinary_kriging_dwd_netatmo_ppt.krige()
            except Exception as msg:
                print('ERROR BEI OK DWD-NETATMO', msg)
            interpolated_vals_dwd_netatmo = (
                ordinary_kriging_dwd_netatmo_ppt.zk.copy())

            # put negative values to 0
            interpolated_vals_dwd_netatmo[
                interpolated_vals_dwd_netatmo < 0] = 0

            #======================================================
            # # SAVING PPT
            #======================================================
            df_interpolated_dwd_only.loc[
                event_date, stn_comb] = np.round(
                    interpolated_vals_dwd_only, 3)
            df_interpolated_netatmo_only.loc[
                event_date, stn_comb] = np.round(
                    interpolated_vals_netatmo_only, 3)
            df_interpolated_dwd_netatmos_comb.loc[
                event_date, stn_comb] = np.round(
                    interpolated_vals_dwd_netatmo, 3)
                
    df_interpolated_dwd_only.dropna(how='all', inplace=True)
    df_interpolated_netatmo_only.dropna(how='all', inplace=True)
    df_interpolated_dwd_netatmos_comb.dropna(how='all', inplace=True)
    
df_interpolated_dwd_only.to_csv(out_plots_path / (
        'df_interpolated_dwd_only_%s_data_%s.csv'
        % (temp_agg, used_data_acc)),
        sep=';', float_format='%0.3f')
df_interpolated_netatmo_only.to_csv(out_plots_path / (
        'df_interpolated_netatmo_only_%s_data_%s.csv'
        % (temp_agg, used_data_acc)),
        sep=';', float_format='%0.3f')
df_interpolated_dwd_netatmos_comb.to_csv(out_plots_path / (
        'df_interpolated_dwd_netatmos_comb_%s_data_%s.csv'
        % (temp_agg, used_data_acc)),
        sep=';', float_format='%0.3f')


stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s \a\a\a' %
      (time.asctime()))
