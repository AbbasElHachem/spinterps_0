'''
Nov 25, 2018
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import pandas as pd

from spinterps import FitVariograms


def get_mean_temp_paths():

    in_vals_df_loc = (
        r'F:\DWD_download_Temperature_data'
        r'\all_dwd_daily_temp_data_combined_2014_2019.csv')

    in_stn_coords_df_loc = (
        r"F:\DWD_download_Temperature_data"
        r"\Dwd_temperature_stations_coords_in_BW_utm32.csv")

    out_dir = r'F:\DWD_temperature_kriging'

    return in_vals_df_loc, in_stn_coords_df_loc, out_dir


def get_min_temp_paths():

    in_vals_df_loc = os.path.join(
        r'Mulde_temperature_min_norm_cop_infill_1950_to_2015_20190417',
        r'02_combined_station_outputs',
        r'infilled_var_df_infill_stns.csv')

    in_stn_coords_df_loc = os.path.join(
        r'Mulde_temperature_min_norm_cop_infill_1950_to_2015_20190417',
        r'02_combined_station_outputs',
        r'infilled_var_df_infill_stns_coords.csv')

    out_dir = r'Mulde_temperature_min_kriging_20190417'

    return in_vals_df_loc, in_stn_coords_df_loc, out_dir


def get_max_temp_paths():

    in_vals_df_loc = os.path.join(
        r'Mulde_temperature_max_norm_cop_infill_1950_to_2015_20190417',
        r'02_combined_station_outputs',
        r'infilled_var_df_infill_stns.csv')

    in_stn_coords_df_loc = os.path.join(
        r'Mulde_temperature_max_norm_cop_infill_1950_to_2015_20190417',
        r'02_combined_station_outputs',
        r'infilled_var_df_infill_stns_coords.csv')

    out_dir = r'Mulde_temperature_max_kriging_20190417'

    return in_vals_df_loc, in_stn_coords_df_loc, out_dir


def get_ppt_paths():

    # MONTHLY
    #     in_vals_df_loc = os.path.join(
    #         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
    #         r'all_dwd_ppt_data_daily_.csv')
    #         in_vals_df_loc = os.path.join(
    #             r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
    #             r'edf_ppt_all_netatmo_monthly_gd_stns_combined_.csv')

    # DAILY
    #     in_vals_df_loc = os.path.join(
    #         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
    #         r'all_netatmo_ppt_data_daily_.csv')
    #     in_vals_df_loc = os.path.join(
    #         r'F:\download_DWD_data_recent',
    #         r'all_dwd_daily_ppt_data_combined_2014_2019_.csv')

    # Hourly
    #     in_vals_df_loc = os.path.join(
    #         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
    #         r'all_netatmo_ppt_data_daily_.csv')
    #     in_vals_df_loc = os.path.join(
    #         r'F:\download_DWD_data_recent',
    #         r'all_dwd_hourly_ppt_data_combined_2014_2019_.csv')

    # CDF VALUES
    #     in_vals_df_loc = os.path.join(
    #         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
    #         r'edf_ppt_all_netamo_daily_gd_stns_combined_.csv')

    #     in_vals_df_loc = os.path.join(
    #         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
    #         r'edf_ppt_all_dwd_daily_.csv')

    in_vals_df_loc = os.path.join(
        r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
        r'ppt_all_dwd_720min_.csv')

    # Cold - Warm season distributions DWD
#     in_vals_df_loc = os.path.join(
#         r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW',
#         r'df_dwd_distributions_cold_season_hourly.csv')

    # COORDS
    in_stn_coords_df_loc = os.path.join(
        r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW\station_coordinates_names_hourly_only_in_BW_utm32.csv")

#     in_stn_coords_df_loc = os.path.join(
#         r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes\NetAtmo_BW\netatmo_bw_1hour_coords_utm32.csv")

    # added by Abbas

    path_to_netatmo_gd_stns_file = (
        r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes"
        r"\plots_NetAtmo_ppt_DWD_ppt_correlation_"
        r"\keep_stns_all_neighbor_99_0_per_60min_.csv")

    # Netatmo extremes
    path_to_netatmo_ppt_extreme = (
        r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes"
        r"\NetAtmo_BW"
        r"\netatmo_daily_maximum_100_days.csv")
    # DWD extremes
#     path_to_dwd_ppt_extreme = (
#         r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes"
#         r"\NetAtmo_BW"
#         r"\dwd_daily_maximum_100_days.csv")
#     path_to_dwd_ppt_extreme = (
#         r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes"
#         r"\NetAtmo_BW"
#         r"\dwd_daily_maximum_100_days.csv")

    path_to_dwd_ppt_extreme = (
        r"X:\hiwi\ElHachem\Prof_Bardossy\Extremes"
        r"\NetAtmo_BW"
        r"\dwd_720min_maximum_100_event.csv")
    out_dir = r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo'

    return (in_vals_df_loc, in_stn_coords_df_loc,
            out_dir,
            path_to_netatmo_gd_stns_file,
            path_to_netatmo_ppt_extreme,
            path_to_dwd_ppt_extreme)


def main():

    main_dir = Path(
        r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes\kriging_ppt_netatmo')
    os.chdir(main_dir)

    vg_vars = ['ppt']  # ['ppt']

    strt_date = '2015-01-01'
    end_date = '2019-09-01'
    min_valid_stns = 10

    drop_stns = []
    mdr = 0.8
    perm_r_list = [1, 2]
    fit_vgs = ['Sph', 'Exp']
    fil_nug_vg = 'Nug'  #
    n_best = 4
    ngp = 5
    figs_flag = True

    fit_for_extreme_events = True

    use_netatmo_good_stns = False

    DWD_stations = True

    n_cpus = 4

    sep = ';'

    for vg_var in vg_vars:
        if vg_var == 'mean_temp':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir) = get_mean_temp_paths()

        elif vg_var == 'min_temp':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir) = get_min_temp_paths()

        elif vg_var == 'max_temp':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir) = get_max_temp_paths()

        elif vg_var == 'ppt':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir,
             path_to_netatmo_gd_stns_file,
             path_to_netatmo_ppt_extreme,
             path_to_dwd_ppt_extreme) = get_ppt_paths()

        else:
            raise RuntimeError(f'Unknown vg_var: {vg_var}!')

        # added by Abbas

        if use_netatmo_good_stns:
            in_df_stns = pd.read_csv(path_to_netatmo_gd_stns_file,
                                     index_col=0,
                                     sep=';')
            good_netatmo_stns = list(in_df_stns.values.ravel())

        in_vals_df = pd.read_csv(
            in_vals_df_loc, sep=sep, index_col=0, encoding='utf-8')

        in_vals_df.index = pd.to_datetime(in_vals_df.index,
                                          format='%Y-%m-%d')
        in_vals_df = in_vals_df.loc[strt_date:end_date, :]

        if use_netatmo_good_stns:
            in_vals_df = in_vals_df.loc[:, good_netatmo_stns]

        if drop_stns:
            in_vals_df.drop(labels=drop_stns, axis=1, inplace=True)

        in_vals_df.dropna(how='all', axis=0, inplace=True)

        # added by Abbas, for edf
        #in_vals_df = in_vals_df[in_vals_df >= 0]

        in_coords_df = pd.read_csv(
            in_stn_coords_df_loc, sep=sep, index_col=0, encoding='utf-8')

        if fit_for_extreme_events:
            df_extremes = pd.read_csv(path_to_dwd_ppt_extreme,
                                      sep=';', index_col=0, parse_dates=True,
                                      infer_datetime_format=True,
                                      header=None)

            in_vals_df = in_vals_df.loc[df_extremes.index, :]

        if DWD_stations:
            # added by Abbas, for DWD stations

            stndwd_ix = ['0' * (5 - len(str(stn_id))) + str(stn_id)
                         if len(str(stn_id)) < 5 else str(stn_id)
                         for stn_id in in_coords_df.index]
            #stndwd_ix = [stn for stn in stndwd_ix if stn in in_vals_df.columns]

            in_coords_df.index = stndwd_ix

        in_coords_df.index = list(map(str, in_coords_df.index))

        if drop_stns:
            in_coords_df.drop(labels=drop_stns, axis=0, inplace=True)

        if in_coords_df.shape[0] > in_vals_df.shape[1]:

            in_vals_df = in_vals_df.loc[:, in_coords_df.index]
        else:
            in_coords_df = in_coords_df.loc[in_vals_df.columns, :]
        fit_vg_cls = FitVariograms()

        fit_vg_cls.set_data(in_vals_df, in_coords_df)

        fit_vg_cls.set_vg_fitting_parameters(
            mdr,
            perm_r_list,
            fil_nug_vg,
            ngp,
            fit_vgs,
            n_best)

        fit_vg_cls.set_misc_settings(n_cpus, min_valid_stns)

        fit_vg_cls.set_output_settings(out_dir, figs_flag)

        fit_vg_cls.verify()

        fit_vg_cls.fit_vgs()

        fit_vg_cls.save_fin_vgs_df()

        fit_vg_cls = None

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'C:\Users\hachem\Desktop\fd\\%s_log_%s.log' % (
                # r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
                os.path.basename(__file__),
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
