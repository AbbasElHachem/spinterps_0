#-------------------------------------------------------------------------------
# Name:        inter_monthly_precip.py
# Purpose:     interpolation of monthly and yearly precipitation values for a
#              5 km x 5 km grid of Germany with OK and EDK
#
# Author:      mosthaf
#
# Created:     14.07.2017
# Copyright:   (c) mosthaf 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys, os
import time
import datetime
import matplotlib as mpl
#if Xming is making troubles starting the script more than once add this line
#but plt.show() is not working anymore!
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tables
import pandas as pd

basepath = os.getcwd()
while os.path.basename(basepath) != '2017_SYNOPSE_II':
    basepath = os.path.dirname(basepath)
# lhg modules
sys.path.append(os.path.join(basepath, '04_Research', '00_Interpolation',
                             '00_code', 'functions'))
sys.path.append(os.path.join(basepath, '03_Data', '00_General_Functions'))
import idw_func
import z_functions
import variogram_func as vf
import kriging_func as kf


def main():
    # Input variables
    # ##########################################################################
    # Start and end year
    start_year = 1997
    end_year   = 2012

    # Aggregations ('A' : annual, 'M' : monthly)
    agg     = ['A', 'M']
    # Number of nan values in daily resolution for each aggregation
    nan_max = [25, 2]

    # plot variogram : True or False
    save_plots = True

    # Input and output files
    # ##########################################################################
    # daily precipitation data
    fn_precip = os.path.join(basepath,'03_Data','1781_2016_daily.h5')
##    fn_precip = os.path.join(basepath, '03_Data','02_data_cross_validation',
##                             'set_a', '1781_2016_daily.h5')
    # dem germany
    fn_dem = os.path.join(basepath, '02_Import', '03_QGIS_project_germany',
                          'xyz_germany_utm.dat')
    # smoothed elevation of gauges
    fn_sme_gauges =os.path.join(basepath,'05_Smoothed_elevation',
                               'smoothed_elevation_gauges.h5')
    # smoothed elevation grid
    fn_sme_grid = os.path.join(basepath, '05_Smoothed_elevation',
                               'smoothed_elevation_ger.h5')

    outpath = os.path.join(basepath, '04_Research', '00_Interpolation',
                          '01_results', '01_data', 'set_orig')
    outpath_misc = os.path.join(outpath, 'misc')
    if not os.path.exists(outpath_misc): os.makedirs(outpath_misc)


    # output file name (with EDK interpolated precipitation values)
    fn_month =  os.path.join(outpath,
                  r'monthly_precip_interpolated_{}_{}.h5'.format(start_year,
                                                                  end_year))
    fn_year = os.path.join(outpath,
               r'yearly_precip_interpolated_{}_{}.h5'.format(start_year,
                                                              end_year))
    # Load data
    # ##########################################################################
    print('reading data')
    # dem germany
    dem = np.loadtxt(fn_dem, delimiter=';').astype(int)
##    ids =['00001', '00002', '00003', '00004', '00006', '00007', '00008',
##       '00009', '00010', '00012', '00013', '00014', '00015', '00016',
####       '00017', '00018', '00019', '00020', '00021', '00022', '00023',
####       '00024', '00025', '00026', '00027', '00028', '00029', '00030',
####       '00031', '00032', '00033', '00034', '00035', '00036', '00037',
####       '00038', '00039', '00040', '00041', '00042', '00043', '00044',
##       '00045', '00046', '00047', '00048', '00050', '00051', '00052',
##       '00053']
    # get data with the SYNOPSE II functions ;-)
    data_daily = z_functions.Get_Daily_Data(fn_precip, start_year, end_year##, ids = ids
    )

    # meta data
    ts_iso    = data_daily.df.index.map(lambda x: datetime.datetime.strftime(
                                                        x, '%Y-%m-%dT%H:%M:%S'))
    ts_year = data_daily.df.index.year

    # Preprocessing
    # ##########################################################################
    print 'aggregating data'
    for iagg, inan_max in zip(agg, nan_max):
        data_daily.get_filtered_timeseries(iagg, inan_max)

    # Isoformat transformation
    ts_month = data_daily.data['M'].index.map(lambda x:
                             datetime.datetime.strftime(x, '%Y-%m-%dT%H:%M:%S'))
    ts_year = data_daily.data['A'].index.map(lambda x:
                             datetime.datetime.strftime(x, '%Y-%m-%dT%H:%M:%S'))

    # Create output hdfs
    # ##########################################################################
    print 'create hdf output files'

    # Monthly
    # --------------------------------------------------------------------------
    hf = tables.open_file(fn_month, 'w', filters=tables.Filters(complevel=6))

    # Write kriged data to file
    hf.create_carray(where=hf.root,
                    name='monthly_inter',
                    atom=tables.FloatAtom(dflt=np.nan),
                    shape=(dem.shape[0],
                           ts_month.shape[0]),
                    chunkshape =(dem.shape[0],
                                 1),
                    title=('Interpolated monthly rainfall values (IDW).'))

    # Write metadata to hdf file
    # Timestamps
    hf.create_carray(where=hf.root,
                    name='ts_iso',
                    atom=tables.StringAtom(itemsize=19, dflt=''),
                    shape=ts_month.shape,
                    chunkshape=ts_month.shape,
                    title='Timestamps in isoformat.')
    hf.root.ts_iso[:] = ts_month

    # xyz values
    hf.create_group(where=hf.root,
                   name='coord_utm',
                   title=('UTM-coordinates for a 5 km grid of Germany.'))
    hf.create_carray(where=hf.root.coord_utm,
                    name='x',
                    atom=tables.Int64Atom(dflt=-99),
                    shape=(dem.shape[0],),
                    chunkshape=(dem.shape[0],),
                    title='UTM-coordinates in x direction.')
    hf.create_carray(where=hf.root.coord_utm,
                    name='y',
                    atom=tables.Int64Atom(dflt=-99),
                    shape=(dem.shape[0],),
                    chunkshape=(dem.shape[0],),
                    title='UTM-coordinates in y direction.')
    hf.root.coord_utm.x[:] = dem[:, 0]
    hf.root.coord_utm.y[:] = dem[:, 1]

    # Close hdf file
    hf.close()

    # Yearly
    # --------------------------------------------------------------------------
    hf = tables.open_file(fn_year, 'w', filters=tables.Filters(complevel=6))

    # Write kriged data to file
    hf.create_carray(where=hf.root,
                    name='yearly_inter',
                    atom=tables.FloatAtom(dflt=np.nan),
                    shape=(dem.shape[0],
                           ts_year.shape[0]),
                    chunkshape =(dem.shape[0],
                                 1),
                    title=('Interpolated yearly rainfall values (IDW).'))

    # Write metadata to hdf file
    # Timestamps
    hf.create_carray(where=hf.root,
                    name='ts_iso',
                    atom=tables.StringAtom(itemsize=19, dflt=''),
                    shape=ts_year.shape,
                    chunkshape=ts_year.shape,
                    title='Timestamps in isoformat.')
    hf.root.ts_iso[:] = ts_year

    # xyz values
    hf.create_group(where=hf.root,
                   name='coord_utm',
                   title=('UTM-coordinates for a 5 km grid of Germany.'))
    hf.create_carray(where=hf.root.coord_utm,
                    name='x',
                    atom=tables.Int64Atom(dflt=-99),
                    shape=(dem.shape[0],),
                    chunkshape=(dem.shape[0],),
                    title='UTM-coordinates in x direction.')
    hf.create_carray(where=hf.root.coord_utm,
                    name='y',
                    atom=tables.Int64Atom(dflt=-99),
                    shape=(dem.shape[0],),
                    chunkshape=(dem.shape[0],),
                    title='UTM-coordinates in y direction.')
    hf.root.coord_utm.x[:] = dem[:, 0]
    hf.root.coord_utm.y[:] = dem[:, 1]

    # Close hdf file
    hf.close()

    # Preprocessing
    # ##########################################################################
    # ##########################################################################

    # smoothed elevation of gauges
    hdf = tables.open_file(fn_sme_gauges, 'r')
    # smoothing radii
    se_radii = hdf.root.sm_radii[:]
    # smoothing vector
    se_vector = hdf.root.sm_vec[:]
    # gauges for smoothed elevation arrays
    se_ids   = hdf.root.id[:]
    bool_ids = np.in1d(se_ids, data_daily.ids)
    # smoothed elevation
    se_sme = hdf.root.sme[:][bool_ids]
    se_ids = hdf.root.id[:][bool_ids]
    hdf.close()

    # Calculate correlations with smoothed elevation
    # ##########################################################################
    print 'find smoothed elevation'
    corr_month = np.full((ts_month.shape[0],
                          se_vector.shape[0],
                          se_radii.shape[0]), fill_value=np.nan)
    corr_year  = np.full((ts_year.shape[0],
                          se_vector.shape[0],
                          se_radii.shape[0]), fill_value=np.nan)


    # monthly precipitation
    # ---------------------
    for ivec, vec in enumerate(se_vector):
        for iradii, radii in enumerate(se_radii):
            for imonth, month in enumerate(ts_month):
                # finite values
                fin_bool = np.isfinite(data_daily.data['M'].values[imonth])
                # calculate correlation coefficient : smoothed elevation and
                # precipitation
                corr_month[imonth, ivec, iradii] = (
                        np.corrcoef(se_sme[fin_bool,
                                           ivec,
                                           iradii],
                                    data_daily.data['M'].values[imonth,
                                                                fin_bool])[0, 1]
                                                    )
    # mean monthly correlations (mean over months)
    meancorr_month = np.mean(corr_month, axis=0)
    # find smoothing vector and radius for maximum correlation
    idxs_maxmonth = np.where(meancorr_month==np.max(meancorr_month))
    print 'monthly: {} / {} km'.format(se_vector[idxs_maxmonth[0]][0],
                                       se_radii[idxs_maxmonth[1]][0])
    # smoothed elevation gauges
    smgauges_month = se_sme[:, idxs_maxmonth[0], idxs_maxmonth[1]]

    # smoothed grid elevation for monthly precipitation
    hdf = tables.open_file(fn_sme_grid, 'r')
    smgrid_month = hdf.root.sme[:, idxs_maxmonth[0], idxs_maxmonth[1]].flatten()
    hdf.close()

    # save information to file
    with open(os.path.join(outpath_misc,'monthly_se_info.txt'), 'w') as f:
        f.write('direction: {}\n'.format(se_vector[idxs_maxmonth[0]][0]))
        f.write('radius: {}'.format(se_radii[idxs_maxmonth[1]][0]))


    # yearly precipitation
    # --------------------
    for ivec, vec in enumerate(se_vector):
        for iradii, radii in enumerate(se_radii):
            for iyear, year in enumerate(ts_year):
                # finite values
                fin_bool = np.isfinite(data_daily.data['A'].values[iyear])
                # calculate correlation coefficient : smoothed elevation and
                # precipitation
                corr_year[iyear, ivec, iradii] = (
                        np.corrcoef(se_sme[fin_bool,
                                           ivec,
                                           iradii],
                                    data_daily.data['A'].values[iyear,
                                                                fin_bool])[0, 1]
                                                    )
    # mean yearly correlations (mean over years)
    meancorr_year = np.mean(corr_year, axis=0)
    # find smoothing vector and radius for maximum correlation
    idxs_maxyear = np.where(meancorr_year==np.max(meancorr_year))
    print 'yearly: {} / {} km'.format(se_vector[idxs_maxyear[0]][0],
                                      se_radii[idxs_maxyear[1]][0])

    # smoothed elevation gauges
    smgauges_year = se_sme[:, idxs_maxyear[0], idxs_maxyear[1]]

    # smoothed grid elevation for yearly precipitation
    hdf = tables.open_file(fn_sme_grid, 'r')
    smgrid_year = hdf.root.sme[:, idxs_maxyear[0], idxs_maxyear[1]].flatten()
    hdf.close()

    # save information to file
    with open(os.path.join(outpath_misc,'yearly_se_info.txt'), 'w') as f:
        f.write('direction: {}\n'.format(se_vector[idxs_maxyear[0]][0]))
        f.write('radius: {}'.format(se_radii[idxs_maxyear[1]][0]))

    # Variograms
    # ##########################################################################
    print 'estimate variograms'

    # Set variogram
    vario = vf.Variogram()
    # width of distance classes of the variogram (5 km)
    vario.setwidthclasses(5000.)

    # empirical variogram array
    distances = np.arange(2500., 102500., 5000.)
    empvarios_month = np.full((distances.shape[0], ts_month.shape[0]),
                              fill_value=np.nan)

    # bounds of the variogram parameters for the variogram fit
    bounds = {'spherical range': [5000., 100000.],
              'spherical sill': [0., 100000.],
              'exponential range':[5000., 100000.] ,
              'exponential sill': [0., 100000.],
              'gauss range':[5000., 100000.] ,
              'gauss sill': [0., 100000.]}

    # monthly precipitation
    # --------------------------------------------------------------------------
    print 'monthly precip'
    # calculate empirical variograms for each month
    for imonth, month in enumerate(ts_month):
        ##print imonth
        # finite values
        fin_bool = np.isfinite(data_daily.data['M'].values[imonth])
        # empirical variograms
        vario.setxcoord(data_daily.x[fin_bool].values)
        vario.setycoord(data_daily.y[fin_bool].values)
        vario.setdata(data_daily.data['M'].values[imonth][fin_bool])
        vario.calc_expervar()
        # for which distances
        bool_dist = np.in1d(vario.expervar[0], distances)
        empvarios_month[bool_dist, imonth] = vario.expervar[1]

    # fit theoretical variogram to every month of the year
    variomod_month = pd.Series().astype(str)
    for year_month in range(1,13):
        print year_month
        # varios months belonging to the year_month
        empvarios = empvarios_month[:,
                                   data_daily.data['M'].index.month==year_month]
        # mean vario
        meanvario = np.nanmean(empvarios, axis=1)
        # fit variogram
        dist_vario = np.vstack((distances, meanvario))
        nb_dec, sillbound = 2, 100000.
        variomod = vf.find_bestvario_expervar(dist_vario, bounds, nb_dec,
                                              sillbound,
                                              variomods=['exponential',
                                                         'gauss',
                                                         'spherical'
                                                         ]
                                              )

        # calculate theoretical variogram values
        plotdistances = np.arange(0., distances[-1]+2500., 100.)
        vario = vf.Variogram()
        vario.name = variomod
        vario.get_params_name()
        vario.get_theovartype_name()
        vario.calc_theovar(plotdistances)
        variomod_month[str(year_month)] = variomod

        # Control plot
        if save_plots:
            plt.title('monthly variogram / month: {}'.format(year_month))
            plt.plot(plotdistances, vario.theovar, 'r-', lw=3)
            plt.plot(distances, meanvario, 'kx')
            plt.tight_layout()
            plt.savefig(os.path.join(outpath_misc,
                                     'monthly_vario_{}.png'.format(year_month)))
            plt.close()

    variomod_month.to_csv(os.path.join(outpath_misc,'monthly_vario_fit.csv'),
                          sep = ';')
    # read the data to have it in a consistent structure compared to other
    # scripts
    variomod_month = pd.read_csv(
                           os.path.join(outpath_misc,'monthly_vario_fit.csv'),
                           sep = ';', header = None, index_col = 0)[1]


    # yearly precipitation
    # --------------------------------------------------------------------------
    # Set variogram
    vario = vf.Variogram()
    # width of distance classes of the variogram (5 km)
    vario.setwidthclasses(5000.)

    empvarios_year = np.full((distances.shape[0], ts_year.shape[0]),
                              fill_value=np.nan)
    print 'yearly precip'

    # calculate empirical variograms for each year
    for iyear, year in enumerate(ts_year):
        print iyear
        # finite values
        fin_bool = np.isfinite(data_daily.data['A'].values[iyear])
        # empirical variograms
        vario.setxcoord(data_daily.x[fin_bool].values)
        vario.setycoord(data_daily.y[fin_bool].values)
        vario.setdata(data_daily.data['A'].values[iyear][fin_bool])
        vario.calc_expervar()
        # for which distances
        bool_dist = np.in1d(vario.expervar[0], distances)
        empvarios_year[bool_dist, iyear] = vario.expervar[1]

    # fit theoretical variogram to every month of the year
    variomod_year = pd.Series().astype(str)
    for iyear, year in enumerate(data_daily.data['A'].index.year):
        # varios for every year
        empvarios = empvarios_year[:, iyear]
        # fit variogram
        dist_vario = np.vstack((distances, empvarios))
        nb_dec, sillbound = 2, 100000.
        variomod = vf.find_bestvario_expervar(dist_vario, bounds, nb_dec,
                                              sillbound,
                                              variomods=['exponential',
                                                         'gauss',
                                                         'spherical']
                                              )

        # calculate theoretical variogram values
        plotdistances = np.arange(0., distances[-1]+2500., 100.)
        vario = vf.Variogram()
        vario.name = variomod
        vario.get_params_name()
        vario.get_theovartype_name()
        vario.calc_theovar(plotdistances)
        variomod_year[str(year)] = variomod

        # Control plot
        if save_plots:
            plt.title('yearly variogram / year: {}'.format(year))
            plt.plot(plotdistances, vario.theovar, 'r-', lw=3)
            plt.plot(distances, empvarios, 'kx')
            plt.tight_layout()
            plt.savefig(os.path.join(outpath_misc,
                        'yearly_vario_{}.png'.format(year)))
            plt.close()

    variomod_year.to_csv(os.path.join(outpath_misc,'yearly_vario_fit.csv'),
                          sep = ';')
    # read the data to have it in a consistent structure compared to other
    # scripts
    variomod_year = pd.read_csv(
                           os.path.join(outpath_misc,'yearly_vario_fit.csv'),
                           sep = ';', header = None, index_col = 0)[1]

    # Interpolation
    # ##########################################################################
    # ##########################################################################

    print 'monthly interpolation'

    # external drift kriging
    kriging = kf.Kriging()
    # loop over timesteps
    for its, ts in enumerate(ts_month):
        print 'its: {} / {}'.format(its+1, ts_month.shape[0])

        # Select only finite values
        mask_precip = np.isfinite(data_daily.data['M'].values[its])
        precip_ts   = data_daily.data['M'].values[its, mask_precip]
        xy          = np.vstack((data_daily.x[mask_precip],
                                 data_daily.y[mask_precip])).T

        # get variogram of the month and set the variogram for kriging
        variomod = variomod_month[data_daily.data['M'].index.month[its]]
        vario = vf.Variogram()
        vario.name = variomod
        vario.get_params_name()
        vario.get_theovartype_name()

        # Get external drift of gauges
        ed_gauges = (smgauges_month.flatten())[mask_precip]

##        # control plot
##            plt.scatter(dem[:, 0], dem[:, 1], c=smgrid_month,
##                        lw=0, s=20, marker='o')
##            plt.scatter(data_daily.x, data_daily.y, c=smgauges_month,
##                        lw=0, s=20, marker='s')
##            plt.show()
##            plt.close()


        # controls
        kriging.setcontrols(xy)
        kriging.setcontrolvalues(precip_ts)
        kriging.setcontrols_ed(ed_gauges)
        # targets
        kriging.settargets(dem[:, :2])
        kriging.settargets_ed(smgrid_month)

        kriging.krige_values(vario, method = 'external drift kriging',
                             min_stations = 10,
                             positive_weights = 'no')

        # do not allow for negative values
        target_values = kriging.kriged_values
        # checking for error values (negative values)
        target_values[target_values<0.] = 0.
        print kriging.kriging_weights[0]
        print target_values[0]
        # control plot
        plt.scatter(dem[:, 0], dem[:, 1], c=target_values,
                    lw=0, s=20, marker='o')
        plt.scatter(xy[:, 0], xy[:, 1], c=precip_ts,
                    lw=0, s=20, marker='s')
        # plt.show()
        plt.savefig('montly_interpolation.png', dpi=600)
        plt.close()

        # Write interpolated data to hdf file
        hf = tables.open_file(fn_month, 'r+')
        hf.root.monthly_inter[:, its] = target_values
        hf.close()

    # ##########################################################################
    print 'yearly interpolation'

     # loop over timesteps
    for its, ts in enumerate(ts_year):
        print 'its: {} / {}'.format(its+1, ts_year.shape[0])

        # Select only finite values
        mask_precip = np.isfinite(data_daily.data['A'].values[its])
        precip_ts   = data_daily.data['A'].values[its, mask_precip]
        xy          = np.vstack((data_daily.x[mask_precip],
                                 data_daily.y[mask_precip])).T

        # get variogram of the year and set the variogram for kriging
        variomod = variomod_year[data_daily.data['A'].index.year[its]]
        vario = vf.Variogram()
        vario.name = variomod
        vario.get_params_name()
        vario.get_theovartype_name()

        # Get external drift of gauges
        ed_gauges = (smgauges_year.flatten())[mask_precip]

        # external drift kriging
        kriging = kf.Kriging()
        # controls
        kriging.setcontrols(xy)
        kriging.setcontrolvalues(precip_ts)
        kriging.setcontrols_ed(ed_gauges)
        # targets
        kriging.settargets(dem[:, :2])
        kriging.settargets_ed(smgrid_year)

        kriging.krige_values(vario, method = 'external drift kriging',
                             min_stations = 10,
                             positive_weights = 'no')
        # do not allow for negative values
        target_values = kriging.kriged_values
        # checking for error values (negative values)
        target_values[target_values<0.] = 0.


        # control plot
        plt.scatter(dem[:, 0], dem[:, 1], c=target_values,
                   lw=0, s=20, marker='o')
        plt.scatter(xy[:, 0], xy[:, 1], c=precip_ts,
                   lw=0, s=20, marker='s')
        plt.savefig('yearly_interpolation.png', dpi=600)
        # plt.show()
        plt.close()


        # Write interpolated data to hdf file
        hf = tables.open_file(fn_year, 'r+')
        hf.root.yearly_inter[:, its] = target_values
        hf.close()

if __name__ == '__main__':
    main()