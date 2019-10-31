'''
Nov 25, 2018
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import pandas as pd

from spinterps_new import SpInterpMain


def main():

    main_dir = Path(
        r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo')
    os.chdir(main_dir)
    #==========================================================================
    in_data_file = os.path.join(
        r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
        r'all_dwd_ppt_data_monthly_.csv')
    # r'all_netatmo_ppt_data_monthly_.csv')

#     in_vals_df_loc = (
#         r'F:\DWD_download_Temperature_data'
#         r'\all_dwd_daily_temp_data_combined_2014_2019.csv')
#
#     in_stn_coords_df_loc = (
#         r"F:\DWD_download_Temperature_data"
#         r"\Dwd_temperature_stations_coords_in_BW_utm32.csv")
#
#     out_dir = r'F:\DWD_temperature_kriging'

#     in_data_file = os.path.join(
#         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
#         r'all_dwd_daily_ppt_data_combined_2014_2019_.csv')

#     in_data_file = os.path.join(
#         r'F:\download_DWD_data_recent',
#         r'all_dwd_daily_ppt_data_combined_2014_2019_.csv')
    #==========================================================================
    # added by abbas
    path_to_netatmo_gd_stns_file = (
        r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes"
        r"\plots_NetAtmo_ppt_DWD_ppt_correlation_"
        r"\keep_stns_all_neighbor_99_0_per_60min_s0_comb.csv")
    #==========================================================================
#     in_data_file = os.path.join(
#         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
#         r'all_dwd_ppt_data_monthly_.csv')
#     in_data_file = os.path.join(
#         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
#         r'edf_ppt_all_dwd_monthly_stns_combined_.csv')
    #==========================================================================
#     in_vgs_file = os.path.join(
#         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo',
#         r'vg_strs_netatmo_edf_high_range_gd_stns.csv')

    in_vgs_file = os.path.join(
        r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo',
        r'vg_strs_dwd_monthly_ppt.csv')

    #==========================================================================
#
    in_stns_coords_file = os.path.join(
        os.path.dirname(in_data_file),
        r'station_coordinates_names_hourly_only_in_BW_utm32.csv')
#
#     in_stns_coords_file = os.path.join(
#         os.path.dirname(in_data_file),
#         r'netatmo_bw_1hour_coords_utm32.csv')
    #==========================================================================
    index_type = 'date'

    out_dir = r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo'

    var_units = 'mm'  # 'mm'  # u'\u2103'  # 'centigrade'
    var_name = 'precipitation'  # 'precipitation'

    out_krig_net_cdf_file = r'Dwd_dwd_daily_precipitation_kriging_%s_to_%s_1km_mid_rg_.nc'
    freq = 'M'
    strt_date = r'2015-01-01'
    end_date = r'2019-08-01'

    out_krig_net_cdf_file = out_krig_net_cdf_file % (strt_date, end_date)

    in_drift_rasters_list = (
        [r'X:\hiwi\ElHachem\Peru_Project\ancahs_dem\srtm_mosacis_deu_clip_1km_utm32.tif'])

#     in_bounds_shp_file = (
#         r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo\shapefile_BW\shapefile_BW_boundaries.shp")
    in_bounds_shp_file = (
        r"F:\data_from_exchange\Netatmo\Landesgrenze_ETRS89\Landesgrenze_10000_ETRS89.shp")
    align_ras_file = in_drift_rasters_list[0]

    nc_time_units = 'days since 1900-01-01 00:00:00.0'
    nc_calendar = 'gregorian'

    min_ppt_thresh = 1  # -float('inf') # 1

    min_var_val = 0
    max_var_val = None

    min_nebor_dist_thresh = 0

    idw_exps = [1, 3, 5]
    n_cpus = 2
    buffer_dist = 2e3
    sec_buffer_dist = 2e3

    neighbor_selection_method = 'all'
    n_neighbors = 5
    n_pies = 8

    in_sep = ';'
    in_date_fmt = '%Y-%m-%d'

    ord_krige_flag = True
    sim_krige_flag = False
    edk_krige_flag = False
    idw_flag = False
    plot_figs_flag = True
    verbose = True
    interp_around_polys_flag = True

    DWD_stations = True

#     ord_krige_flag = False
#     sim_krige_flag = False
#     edk_krige_flag = False
#     idw_flag = False
#     plot_figs_flag = False
#     verbose = False
#     interp_around_polys_flag = False

    in_df_stns = pd.read_csv(path_to_netatmo_gd_stns_file, index_col=0,
                             sep=';')
    good_stns = list(in_df_stns.values.ravel())

    in_data_df = pd.read_csv(
        in_data_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8')

    in_vgs_df = pd.read_csv(
        in_vgs_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8',
        dtype=str).dropna(how='all')

    in_stns_coords_df = pd.read_csv(
        in_stns_coords_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8')

    in_stns_coords_df.drop_duplicates(['X', 'Y'], keep='first', inplace=True)

#     in_data_df = in_data_df.loc[in_data_df.index.intersection(
#         in_vgs_df.index), :].dropna(how='all')

    if DWD_stations:
        # added by Abbas, for DWD stations
        stndwd_ix = ['0' * (5 - len(str(stn_id))) + str(stn_id)
                     if len(str(stn_id)) < 5 else str(stn_id)
                     for stn_id in in_stns_coords_df.index]

        in_stns_coords_df.index = stndwd_ix

    # added by Abbas
#     in_data_df = in_data_df.loc[:, good_stns]

    in_data_df.dropna(inplace=True, how='all')

    if index_type == 'date':
        in_data_df.index = pd.to_datetime(in_data_df.index, format=in_date_fmt)
        in_vgs_df.index = pd.to_datetime(in_vgs_df.index, format=in_date_fmt)

    elif index_type == 'obj':
        in_data_df.index = pd.Index(in_data_df.index, dtype=object)
        in_vgs_df.index = pd.Index(in_vgs_df.index, dtype=object)

    else:
        raise ValueError(f'Incorrect index_type: {index_type}!')
    # added by Abbas

    cmn_idx = in_data_df.index.intersection(in_vgs_df.index)
    in_data_df = in_data_df.loc[cmn_idx, :]
    in_vgs_df = in_vgs_df.loc[cmn_idx, :]

    spinterp_cls = SpInterpMain(verbose)

    spinterp_cls.set_data(in_data_df, in_stns_coords_df, index_type,
                          min_nebor_dist_thresh)
    spinterp_cls.set_vgs_ser(in_vgs_df.iloc[:, 0], index_type=index_type)
    spinterp_cls.set_out_dir(out_dir)

    spinterp_cls.set_netcdf4_parameters(
        out_krig_net_cdf_file,
        var_units,
        var_name,
        nc_time_units,
        nc_calendar)

    spinterp_cls.set_interp_time_parameters(
        strt_date, end_date, freq, in_date_fmt)
    spinterp_cls.set_cell_selection_parameters(
        in_bounds_shp_file,
        buffer_dist,
        interp_around_polys_flag,
        sec_buffer_dist)
    spinterp_cls.set_alignment_raster(align_ras_file)

    spinterp_cls.set_neighbor_selection_method(
        neighbor_selection_method, n_neighbors, n_pies)

    spinterp_cls.set_misc_settings(
        n_cpus,
        plot_figs_flag,
        None,
        min_ppt_thresh,
        min_var_val,
        max_var_val)

    if ord_krige_flag:
        spinterp_cls.turn_ordinary_kriging_on()

    if sim_krige_flag:
        spinterp_cls.turn_simple_kriging_on()

    if edk_krige_flag:
        spinterp_cls.turn_external_drift_kriging_on(in_drift_rasters_list)

    if idw_flag:
        spinterp_cls.turn_inverse_distance_weighting_on(idw_exps)

    spinterp_cls.verify()
    try:
        spinterp_cls.interpolate()
    except Exception as msg:
        print(msg)
    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo\\%s_log_%s.log'
            % (os.path.basename(__file__),
                datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
