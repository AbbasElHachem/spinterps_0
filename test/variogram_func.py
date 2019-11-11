import numpy as np
import scipy.special as sps
import scipy as sp
import scipy.spatial as spatial
#import simplex_func
from scipy.optimize import minimize
import itertools
import matplotlib.pyplot as plt

class Variogram:

    # --------------------------------------------------------------------------
    # Experimental variogram
    # --------------------------------------------------------------------------

    def __init__(self):
        # For the experimental variogram
        self.widthclasses = 1000.
        self.maxdistance  = 100000.

        # Default sum of sills which should not be crossed during the
        # optimization
        self.sillbound = 100.
        # Define maximum range
        self.rangebound = 200000.

        # For the theoretical variogramm model starting parameters
        # draw starting parameter set randomly between bounds
        # ----------------------------------------------------------------------
        # For the fitting of a theoretical to the experimental variogram
        self.bounds = {'nugget sill': [0., 100.],
                       'spherical range': [0., 100000.],
                       'spherical sill': [0., 1000.],
                       'exponential range':[0., 100000.] ,
                       'exponential sill': [0., 1000.],
                       'mat v': [0.5, 50.],
                       'mat range': [0., 100000.],
                       'mat sill': [0., 20.],
                       'gauss range':[0., 100000.] ,
                       'gauss sill': [0., 1000.]}

        self.parameters = {}
        for ikey, key in enumerate(self.bounds.keys()):
            param0 = np.random.rand() * 0.6 + 0.2
            param0 = param0 * (self.bounds[key][1] - self.bounds[key][0]
                              ) + self.bounds[key][0]
            self.parameters[key] = param0

        # Calculate the sillsumm
        sumsill = 0.
        for ikey, key in enumerate(self.parameters.keys()):
            if 'sill' in key:
                sumsill += self.parameters[key]
        if sumsill > self.sillbound:
            self.setSillbound(sumsill)

        # Number of decimals
        self.nb_dec = 4

    # -------------------------------------------------------------------------
    def setParambounds(self, bounds):
        """ Must be a dictionnary  ( see __init___ !!!) """
        self.bounds = bounds

        # For the theoretical variogram model starting parameters
        # draw starting parameter set randomly between bounds
        # ----------------------------------------------------------------------
        # random number between 0.2 and 0.8 --> scale to parameter space
        # parameter is then between bounds +- 20%...
        self.parameters = {}
        for ikey, key in enumerate(self.bounds.keys()):
            param0 = np.random.rand() * 0.6 + 0.2
            param0 = param0 * (self.bounds[key][1] - self.bounds[key][0]
                              ) + self.bounds[key][0]
            self.parameters[key] = param0

        # Calculate the sillsumm
        sumsill = 0.
        for ikey, key in enumerate(self.parameters.keys()):
            if 'sill' in key:
                sumsill += self.parameters[key]
        if sumsill > self.sillbound:
            self.setSillbound(sumsill)


    def setSillbound(self, sillbound):
        """ Set the sillbound for the optimization
        """
        self.sillbound = sillbound

        # Define the start parameters according to the sillbound
        sumsill = 0.
        for ikey, key in enumerate(self.parameters.keys()):
            if 'sill' in key:
                sumsill += self.parameters[key]

        # Cut down the values if needed
        if sumsill > sillbound:
            for ikey, key in enumerate(self.parameters.keys()):
                if 'sill' in key:
                    self.parameters[key] = self.parameters[key] * (sillbound/
                                                               sumsill) - 0.0001

    def setdata(self, data):
        self.data = data

    def setxcoord(self, xcoord):
        self.xcoord = xcoord

    def setycoord(self, ycoord):
        self.ycoord = ycoord

    def setwidthclasses(self, widthclasses):
        self.widthclasses = np.float(widthclasses)

    def setmaxdistance(self, maxdistance):
        self.maxdistance = np.float(maxdistance)

    def setparams_tozero(self):
        """ set all parameter values to zero """
        for iparam, param in enumerate(self.parameters.keys()):
            self.parameters[param] = 0.

    def setexpervar(self, expervar):
        """ Set experimental variogram.
        """
        self.expervar = np.array(expervar)

    def calc_expervar(self):
        """ Calculates the experimental variogram. The input coordinates must be
        in a equidistant coordinate system (like the Gauss-Krueger system),
        otherwise the input coordinates must be transformed firstly.

        Parameters
        ----------
        data : array_like
            1D or 2D array which contains the data. If a temporal dimension
            exists, the temporal dimension must be the first dimension and the
            second the spatial (station) dimension.
        xcoord : array_like
            1D array which contains the x coordinates of the controls (e.g. of
            precipitation gauges).
        ycoord : array_like
            1D array which contains the y coordinates. Must have the same shape
            as xcoord.
        widthclasses : float
            Width of the distance classes for which the variogram will be
            calculated.
        maxdistance : float
            Maximum value of the upper bound of the distance classes, must be
            divisible by widthclasses.
        timedimension : boolean (True or False)
            True if the data array has a temporal dimension.

        Returns
        -------
        variogram : array_like
            2D array, which contains the values of the experimental variogram
            and the corresponding distance values (middle of the classes).

        """
        # 0. Reshape the data array
        if len(self.data.shape) < 2:
            data_reshape = self.data.reshape(1, self.data.shape[0])
        else:
            data_reshape = self.data

        # 1. Calculate the distance matrix
        # --------------------------------
        # Calculate the pairwise distances from every station to all other
        # stations the shape therfore is : number of stations, number of
        # stations (with the value 0. included for the distance to the station
        # itself)

        coord_stations     = np.vstack((self.xcoord, self.ycoord)).T
        distance_matrixall = spatial.distance.cdist(coord_stations,
                                                    coord_stations,
                                                    'euclidean')

        # 2. Define the distance classes for calculating the experimental
        # variogram
        # ----------------------------------------------------------------------
        # The distance classes to calculate the experimental variogram are
        # represented by classes_bounds which contains the lower bounds of the
        # classes.
        if np.mod(self.maxdistance, self.widthclasses) != 0:
            raise ValueError('widthclasses is not a divisor of maxdistance !')
        # Define the lower bounds of the distance classes
        classes_bounds = np.arange(0., self.maxdistance, self.widthclasses)

        # 3. Loop to calculate the experimental variogram
        # -----------------------------------------------
        # Initialise the experimental variogram
        self.expervar    = np.ones((2, classes_bounds.shape[0]))
        self.expervar[:] = np.nan
        # For loop over the distance classes
        for idx_classes in range(classes_bounds.shape[0]):
            # Find stations with the pairwise distance in the regarded class
            distance   = classes_bounds[idx_classes]
            idx1_d, idx2_d = np.where((distance_matrixall > distance) &
                             (distance_matrixall <= distance+self.widthclasses))
            # release double pairs
            idx1 = idx1_d[np.argsort(distance_matrixall[idx1_d, idx2_d])[::2]]
            idx2 = idx2_d[np.argsort(distance_matrixall[idx1_d, idx2_d])[::2]]
            # Calculate the value of the experimental variogram for the regarded
            # distance class
            values_class    = np.ones(idx1.shape[0])
            values_class[:] = np.nan
            nb_values       = np.ones(idx1.shape[0])
            nb_values[:]    = np.nan
            # Loop over pairwise stations with the correct distance
            for idx_pair in range(idx1.shape[0]):
                # Assign values of the pairwise stations
                data1raw = data_reshape[:, idx1[idx_pair]]
                data2raw = data_reshape[:, idx2[idx_pair]]
                # Only use values, for which pairwise values exist
                pairwiselocs = np.where((data1raw>= 0.) & (data2raw>= 0.))[0]
                data1 = data1raw[pairwiselocs]
                data2 = data2raw[pairwiselocs]
                # Calculate the (z(s)-z(s+h))**2 values and the number of values
                values_class[idx_pair] = np.sum((data1 - data2)**2.)
                nb_values[idx_pair]    = data1.shape[0]
            # Calculate the variogram value for the regared distance class
            # If there are now values the sum is 0.
            if np.sum(nb_values) == 0. :
                self.expervar[1, idx_classes] = np.nan
                self.expervar[0, idx_classes] = np.nan
            else:
                self.expervar[1, idx_classes] = (1./(2.*np.sum(nb_values)))*(
                                                           np.sum(values_class))
                self.expervar[0, idx_classes] = distance + (
                                                           self.widthclasses/2.)

##            if idx_classes%10==0: print 'class',idx_classes,'finished'
        # Only no nan values are valid for the variogram
        nonan     = np.where(np.isfinite(self.expervar).any(0))[0]
        self.expervar = self.expervar[:, nonan]

    # --------------------------------------------------------------------------
    # Theoretical variograms
    # --------------------------------------------------------------------------
    def setParameters(self, parameters):
        """ Must be a dictionnary """
        self.parameters = parameters

    def setTheovar(self, theovartype):
        """ Define, which theoretical variogram should be used
            composed: all theoretical models combined
            single models: 'nugget', 'matern', 'gauss', 'spherical',
                           'exponential'
            the single models can also be combined in the following way:
            'nugget+matern'
        """
        self.theovartype = theovartype

        if 'composed' not in self.theovartype:
            # Kick out the parameters
            if 'nugget' not in self.theovartype:
                if 'nugget sill' in self.parameters.keys():
                    self.parameters.pop('nugget sill')
            if 'matern' not in self.theovartype:
                if 'mat range' in self.parameters.keys():
                    self.parameters.pop('mat range')
                    self.parameters.pop('mat sill')
                    self.parameters.pop('mat v')
            if 'gauss' not in self.theovartype:
                if 'gauss range' in self.parameters.keys():
                    self.parameters.pop('gauss range')
                    self.parameters.pop('gauss sill')
            if 'spherical' not in self.theovartype:
                if 'spherical range' in self.parameters.keys():
                    self.parameters.pop('spherical range')
                    self.parameters.pop('spherical sill')
            if 'exponential' not in self.theovartype:
                if 'exponential range' in self.parameters.keys():
                    self.parameters.pop('exponential range')
                    self.parameters.pop('exponential sill')

        # get name
        self.create_varname()
        # get the theovartyp from the name
        self.get_theovartype_name()
        types = self.theovartype.split('+')
        for itype, type in enumerate(types):
            if type not in self.theovartype:
                raise Exception('Variogram type not setted correctly!')


    def calc_theovar(self, xvalues):
        """ Calculate the theoretical variogram.
        If composed variogram:
        - parameter of the nugget --> sill nugget
        - spherical --> sill spherical, range spherical
        - exponential --> sill exponential, range exponential
        - matern --> v mat, sill mat, range mat
        - gauss --> sill gauss, range gauss
        """
        # variogram composed of all theoretical variograms
        if self.theovartype == 'composed':
            self.theovar = calc_nuggetvar(self.parameters['nugget sill'],
                                          xvalues)
            self.theovar = self.theovar + calc_sphericalvar(
                                             self.parameters['spherical range'],
                                             self.parameters['spherical sill'],
                                             xvalues)
            self.theovar = self.theovar + calc_expvar(
                                           self.parameters['exponential range'],
                                           self.parameters['exponential sill'],
                                           xvalues)
            self.theovar = self.theovar + calc_matvar(
                                                   self.parameters['mat v'],
                                                   self.parameters['mat range'],
                                                   self.parameters['mat sill'],
                                                   xvalues)
            self.theovar = self.theovar + calc_gaussvar(
                                                 self.parameters['gauss range'],
                                                 self.parameters['gauss sill'],
                                                 xvalues)
        # variogram composed of certain variograms
        else:
            # initialize the variogram
            self.theovar = np.zeros(xvalues.shape[0])
            # calculate its values
            if 'spherical' in self.theovartype:
                self.theovar += calc_sphericalvar(
                                             self.parameters['spherical range'],
                                             self.parameters['spherical sill'],
                                             xvalues)
            if 'exponential' in self.theovartype:
                self.theovar += calc_expvar(
                                           self.parameters['exponential range'],
                                           self.parameters['exponential sill'],
                                           xvalues)
            if 'nugget' in self.theovartype:
                self.theovar = calc_nuggetvar(self.parameters['nugget sill'],
                                          xvalues)
            if 'matern' in self.theovartype:
                self.theovar += calc_matvar(self.parameters['mat v'],
                                            self.parameters['mat range'],
                                            self.parameters['mat sill'],
                                            xvalues)

            if 'gauss' in self.theovartype:
                self.theovar += calc_gaussvar(self.parameters['gauss range'],
                                              self.parameters['gauss sill'],
                                              xvalues)

    # --------------------------------------------------------------------------
    # Fit theoretical variogram to the experimental
    # Only works with the composed variogram until now
    # --------------------------------------------------------------------------
    def fit_theovar(self):
        """ Fit a theoretical variogram to the experimental one.
        """
        # Fit the theoretical variogram

        # Optimization algorithm parameters
        # Set the start parameters
        x0          = np.ones(len(self.parameters))
        keys_sorted = np.sort(self.parameters.keys())
        for iparam in range(len(self.parameters)):
            x0[iparam] = self.parameters[keys_sorted[iparam]]

        # Optimization bounds of the parameters
        bounds = np.ones((len(self.parameters) ,2))
        for ibound in range(len(self.parameters)):
            bounds[ibound, :] = self.bounds[keys_sorted[ibound]]

        # Arguments
        args1 = np.copy(self.expervar[0, :])
        args2 = np.copy(self.expervar[1, :])
        # Sum of sills
        args3 = self.sillbound
        args4 = self.nb_dec
        args5 = self.rangebound
        args6 = self.theovartype
        args = (args1, args2, args3, args4, args5, args6)

        # Auxiliary variables to get also a variogram if there is not really a
        # "good" spatial dependence structure
        vario_ok = False
        n_iter   = 0
        max_iter = 10

        # Try to optimize the variogram and check if it has failed (horizontal
        # line)
        while ((vario_ok == False) & (n_iter<max_iter)):
            # Simplex Algorithm (by Philip Guthke)
            opt_result = minimize(opt_theovar,
                                   x0,
                                   args=args,
                                   bounds=bounds,
                                   method='trust-constr')
            
#            opt_result = simplex_func.minimize_bounded(opt_theovar,
#                                                       x0,
#                                                       args=args,
#                                                       bounds=bounds,
#                                                       talk_to_me=False)
            # Read optimized values
            self.ls = opt_result[1]
            params  = opt_result[0]
            params[params<1*10**(-self.nb_dec)] = 0.

            # assign parameters
            # index counter for the parameters
            param_index = 0
            # Optimization parameters
            opt_params = {}
            # alphabetical order !
            if 'exponential' in self.theovartype:
                opt_params['exponential range'] = params[0]
                opt_params['exponential sill'] = params[1]
                param_index += 2
            if 'gauss' in self.theovartype:
                opt_params['gauss range'] = params[param_index]
                opt_params['gauss sill']  = params[param_index+1]
                param_index += 2
            if 'matern' in self.theovartype:
                opt_params['mat range'] = params[param_index]
                opt_params['mat sill']  = params[param_index+1]
                opt_params['mat v']  = params[param_index+2]
                param_index += 3
            if 'nugget' in self.theovartype:
                opt_params['nugget sill']  = params[param_index]
                param_index += 1
            if 'spherical' in self.theovartype:
                opt_params['spherical range'] = params[param_index]
                opt_params['spherical sill']  = params[param_index+1]
                param_index += 2.
            # loop over parameters
            for ikey, key in enumerate(opt_params.keys()):
                opt_params[key] = np.round(opt_params[key], self.nb_dec)
            # if sill is 0 range is also zero and the other way round
            for ikey, key in enumerate(opt_params.keys()):
                if 'sill' in key:
                    model = key.split()[0]
                    if opt_params[key] == 0.:
                        opt_params[model + ' range'] = 0.
                if 'range' in key:
                    model = key.split()[0]
                    if opt_params[key] == 0.:
                        opt_params[model + ' sill'] = 0.
            self.setParameters(opt_params)

            # Calculate the variogram and check if the opimization failed
            self.calc_theovar(self.expervar[0,:])

            # variogram optimizatioin fails, if the varigoram is an horizontal
            # line -> if the first and last value are almost the same retry the
            # optimization 10 times
            if (self.theovar[0] / self.theovar[-1])<0.99:
                vario_ok = True
            else:
                n_iter +=1
                #print 'Retry the optimization', n_iter

        # if the varigram couldn't be optimized after 10 tries
        # (sills are almost 0) the sills of both
        # variograms are (slightly) stepwise increased to avoid an horizontal
        # variogram, which cause issues in the krigging
        # If the first an the last value of the variogram is significantly
        # different the new "optimization" is finished
        if vario_ok == False:
            #print 'Variogram could not be optimized: Pure nugget variogram ! '
            # Pure nugget variogram
            for ikey, key in enumerate(opt_params.keys()):
                opt_params[key] = self.expervar[1,-1]
            self.setParameters(opt_params)
            self.setTheovar('nugget')
            self.calc_theovar(self.expervar[0,:])
            self.ls = np.nan

        return opt_params

    # --------------------------------------------------------------------------
    # Create name for the variogram
    # Only works with the composed variogram until now
    # --------------------------------------------------------------------------
    def create_varname(self):
        """ Create variogram name, according to the covcariance names of PG and
        SH.
        """
        # if sill is 0 range is also zero and the other way round
        for ikey, key in enumerate(self.parameters.keys()):
            if 'sill' in key:
                model = key.split()[0]
                if self.parameters[key] == 0.:
                    if model != 'nugget':
                        self.parameters[model + ' range'] = 0.
                        if model == 'mat':
                            self.parameters[model + ' v'] = 0.
            if 'range' in key:
                model = key.split()[0]
                if self.parameters[key] == 0.:
                    self.parameters[model + ' sill'] = 0.
                    if model == 'mat':
                        self.parameters[model + ' v'] = 0.
        self.setParameters(self.parameters)

        # Initialise the name
        name = ''

        # Find variogram parameters which have "real" values
        idx_param   = np.where(np.array(self.parameters.values())>0.)[0]
        vario_names = np.array(self.parameters.keys())[idx_param]

        if 'nugget sill' in vario_names:
            name += '{0}Nug+'.format(np.round(
                                   self.parameters['nugget sill'], self.nb_dec))
        if 'exponential sill' in vario_names:
            name += '{0}Exp({1})+\n'.format(
                    np.round(self.parameters['exponential sill'], self.nb_dec),
                    np.round(self.parameters['exponential range'], self.nb_dec))
        if 'gauss sill' in vario_names:
            name += '{0}Gau({1})+\n'.format(
                          np.round(self.parameters['gauss sill'], self.nb_dec),
                          np.round(self.parameters['gauss range'], self.nb_dec))
        if 'spherical sill' in vario_names:
            name += '{0}Sph({1})+\n'.format(
                      np.round(self.parameters['spherical sill'], self.nb_dec),
                      np.round(self.parameters['spherical range'], self.nb_dec))
        if 'mat sill' in vario_names:
            name += '{0}Mat({1})^{2}\n'.format(
                            np.round(self.parameters['mat sill'], self.nb_dec),
                            np.round(self.parameters['mat range'], self.nb_dec),
                            np.round(self.parameters['mat v'], self.nb_dec))
        # Final output
        self.name = name

    # --------------------------------------------------------------------------
    # Get parameter values from the variogram name
    # --------------------------------------------------------------------------
    def get_params_name(self):
        """ Get the values of the variogram parameters from the name
        """
        vario_models = self.name.split('+')
        # Theoretical models which are not part of the regarded model
        if 'Nug' not in self.name:
            self.parameters.pop('nugget sill')
        if 'Exp' not in self.name:
            self.parameters.pop('exponential range')
            self.parameters.pop('exponential sill')
        if 'Gau' not in self.name:
                self.parameters.pop('gauss range')
                self.parameters.pop('gauss sill')
        if 'Sph' not in self.name:
                self.parameters.pop('spherical range')
                self.parameters.pop('spherical sill')
        if 'Mat' not in self.name:
                self.parameters.pop('mat range')
                self.parameters.pop('mat sill')
                self.parameters.pop('mat v')

        # Loop over theoretical variogram models to get the parameter values
        for imodel, model in enumerate(vario_models):
            # Nugget model
            if 'Nug' in model:
                Sill  = np.float(model.split('Nug')[0])
                self.parameters['nugget sill']  = Sill
            # Exponential model
            if 'Exp' in model:
                Range = np.float(model.split('Exp')[1].split('(')[1].split(
                                                                       ')')[0])
                Sill  = np.float(model.split('Exp')[0])
                self.parameters['exponential range'] = Range
                self.parameters['exponential sill']  = Sill
            # Gaussian model
            if 'Gau' in model:
                Range = np.float(model.split('Gau')[1].split('(')[1].split(
                                                                       ')')[0])
                Sill  = np.float(model.split('Gau')[0])
                self.parameters['gauss range'] = Range
                self.parameters['gauss sill']  = Sill
            # Sperical model
            if 'Sph' in model:
                Range = np.float(model.split('Sph')[1].split('(')[1].split(
                                                                       ')')[0])
                Sill  = np.float(model.split('Sph')[0])
                self.parameters['spherical range'] = Range
                self.parameters['spherical sill']  = Sill
            # Matern model
            if 'Mat' in model:
                Range = np.float(model.split('Mat')[1].split('(')[1].split(
                                                                       ')')[0])
                Sill  = np.float(model.split('Mat')[0])
                V     = np.float(model.split('Mat')[1].split('^')[1])
                self.parameters['mat range'] = Range
                self.parameters['mat sill']  = Sill
                self.parameters['mat v']     = V


    # --------------------------------------------------------------------------
    # Get theoretical variorgram typefrom the variogram name
    # --------------------------------------------------------------------------
    def get_theovartype_name(self):
        """ Get the theoretical variorgram type from the name
        """
        # Theoretical models which are not part of the regarded model
        self.theovartype = ''
        if 'Nug' in self.name:
            self.theovartype += 'nugget'
        if 'Exp' in self.name:
            if len(self.theovartype) == 0:
                self.theovartype += 'exponential'
            else:
                self.theovartype += '+exponential'
        if 'Gau'  in self.name:
            if len(self.theovartype) == 0:
                self.theovartype += 'gauss'
            else:
                self.theovartype += '+gauss'
        if 'Sph' in self.name:
            if len(self.theovartype) == 0:
                self.theovartype += 'spherical'
            else:
                self.theovartype += '+spherical'
        if 'Mat' in self.name:
            if len(self.theovartype) == 0:
                self.theovartype += 'matern'
            else:
                self.theovartype += '+matern'

# ------------------------------------------------------------------------------
# Optimization algorithm
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def opt_theovar(params, distances, expervariogram, sillbound, nb_dec,
                rangebound, theovartype):
    """ Objective function of the optimization of the NiedSim variogram.

    Parameters
    ----------
    params : array
        optimization parameters
    distances : array
        contains the distances for which the variogram should be fitted
    expervariogram : array
        contains the values of the experimental variogram for the distances
        array
    """
    # index counter for the parameters
    param_index = 0
    # initialise variogram
    variogram = np.zeros(distances.shape[0])
    # initialise sumsill
    sumsill = 0

    # Optimization parameters
    if 'exponential' in theovartype:
        variogram   += calc_expvar(params[0], params[1], distances)
        sumsill     += params[1]
        if params[param_index] > rangebound: obj = 1000000.
        param_index += 2
    if 'gauss' in theovartype:
        variogram   += calc_gaussvar(params[param_index],
                                     params[param_index+1],
                                     distances)
        sumsill     += params[param_index+1]
        if params[param_index] > rangebound: obj = 1000000.
        param_index += 2
    if 'matern' in theovartype:
        variogram   += calc_matvar(params[param_index+2], params[param_index],
                                   params[param_index+1], distances)
        sumsill     += params[param_index+1]
        if params[param_index] > rangebound: obj = 1000000.
        param_index += 3
    if 'nugget' in theovartype:
        variogram   += calc_nuggetvar(params[param_index], distances)
        sumsill     += params[param_index]
        param_index += 1
    if 'spherical' in theovartype:
        variogram   += calc_sphericalvar(params[param_index],
                                         params[param_index+1], distances)
        sumsill     += params[param_index+1]
        if params[param_index] > rangebound: obj = 1000000.

    # Least-squares optimization
    if sumsill > sillbound:
        obj = 1000000.
    else:
        obj = np.sum(((variogram-expervariogram))**2.)
    return obj

# ------------------------------------------------------------------------------
# Fitting over all variograms (with data as input)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def find_bestvario(data, xcoord, ycoord, bounds, nb_dec, sillbound,
                   variomods=['exponential','gauss','matern', 'spherical',
                              'nugget'],
                   len_comb='default',
                   widthclasses=1000.,
                   maxdistance=100000.):
    """ Find the best combination of all variogram models.
    """

    # Get all corresponding combinations of the single variogram models
    # ------------------------------------------------------------------
    variomods_comb = []
    if len_comb=='default':
        len_comb = 1 + np.array(range(len(variomods)))

    # loop over the different number of used for the combinations
    for size in (len_comb):
        # Append the different models for this number of used models
        for comb in itertools.combinations(variomods, size):
            name = ''
            for i in np.sort(comb):
                name = name + '+' + i
            variomods_comb.append(name[1:])
    variomods_comb = np.sort(np.array(variomods_comb))

    # exclude the pure nugget variogram model (is not possible to fit ->
    # singular matrix problem for the covariance matrix)
    variomods_comb = variomods_comb[variomods_comb!='nugget']

    # Fit the different models
    # ------------------------
    # Initialise the least squares
    lsq = []
    # Initialise the corresponding list of the models
    fitted_models = []
    # Loop over the variogram models
    for variomod in variomods_comb:
        # Set up the variogram
        vario = Variogram()
        vario.setdata(data)
        vario.setxcoord(xcoord)
        vario.setycoord(ycoord)
        vario.setwidthclasses(widthclasses)
        vario.setmaxdistance(maxdistance)
        vario.setParambounds(bounds)
        vario.nb_dec    = nb_dec
        vario.sillbound = sillbound
        # Calculate experimental variogram
        vario.calc_expervar()
        # Define the theoretical variogram
        vario.setTheovar(variomod)
        # Fit the theoretical variogram
        vario.fit_theovar()
        # Get name of the variogram
        vario.create_varname()
        # appending least squares and optimized variogram model
        if np.isfinite(vario.ls):
            lsq.append(np.sum(vario.expervar[1] - vario.theovar)**2)
            fitted_models.append(vario.name)

            #plt.title(vario.name)
            #plt.plot(vario.expervar[0], vario.expervar[1], 'k+')
            #plt.plot(vario.expervar[0], vario.theovar, 'r--')
            #plt.tight_layout()
            #plt.show()
            #plt.clf()
            #plt.close()

    # Array conversion
    if len(fitted_models) == 0:
        print('Attention!\n No fitting model!')
        bestmod = None

    else:
        fitted_models = np.array(fitted_models)
        lsq           = np.array(lsq)
        # Get best model
        idx_best = np.argmin(lsq)
        bestmod  = fitted_models[idx_best]
        # Plot best model
        #varbest = Variogram()
        #varbest.name = bestmod
        #varbest.get_params_name()
        #varbest.get_theovartype_name()
        #varbest.calc_theovar(vario.expervar[0])
        # Plotting
        #plt.title(varbest.name)
        #plt.plot(vario.expervar[0], vario.expervar[1], 'k+')
        #plt.plot(vario.expervar[0], varbest.theovar, 'r--')
        #plt.tight_layout()
        #plt.show()
        #plt.clf()
        #plt.close()
    return bestmod, vario.expervar

# ------------------------------------------------------------------------------
# Fitting over all variograms (with Expervar as input)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def find_bestvario_expervar(expervar, bounds, nb_dec, sillbound,
                            variomods = ['exponential','gauss','matern',
                                         'spherical', 'nugget'],
                            len_comb='default'):
    """ Find the best combination of all variogram models.
    """

    # Get all corresponding combinations of the single variogram models
    # ------------------------------------------------------------------
    variomods_comb = []
    if len_comb=='default':
        len_comb = 1 + np.array(range(len(variomods)))

    # loop over the different number of used for the combinations
    for size in (len_comb):
        # Append the different models for this number of used models
        for comb in itertools.combinations(variomods, size):
            name = ''
            for i in np.sort(comb):
                name = name + '+' + i
            variomods_comb.append(name[1:])
    variomods_comb = np.sort(np.array(variomods_comb))

    # exclude the pure nugget variogram model (is not possible to fit ->
    # singular matrix problem for the covariance matrix)
    variomods_comb = variomods_comb[variomods_comb!='nugget']

    # Fit the different models
    # ------------------------
    # Initialise the least squares
    lsq = []
    # Initialise the corresponding list of the models
    fitted_models = []
    # Loop over the variogram models
    for variomod in variomods_comb:
        # Set up the variogram
        vario = Variogram()
        vario.setexpervar(expervar)
        vario.setParambounds(bounds)
        vario.nb_dec    = nb_dec
        vario.sillbound = sillbound
        # Define the theoretical variogram
        vario.setTheovar(variomod)
        # Fit the theoretical variogram
        vario.fit_theovar()
        # Get name of the variogram
        vario.create_varname()
        # appending least squares and optimized variogram model
        if np.isfinite(vario.ls):
            lsq.append(np.sum(vario.expervar[1] - vario.theovar)**2)
            fitted_models.append(vario.name)

    # Array conversion
    fitted_models = np.array(fitted_models)
    lsq           = np.array(lsq)
    # Get best model
    idx_best = np.argmin(lsq)
    bestmod  = fitted_models[idx_best]

    return bestmod

# ------------------------------------------------------------------------------
# Functions for different theoretical variograms
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def calc_sphericalvar(range, sill, distances):
    """
    Calculate the spherical variogram
    """
    # Initialise the variogram with NAN values
    theovar    = np.ones(distances.shape[0])
    theovar[:] = np.nan

    # Calculate the variogram values
    # 1. For distances smaller than or equal the range
    smlt_range = (distances <= range)
    if range > 1.e-100:
        dist1               = distances[smlt_range]
        theovar[smlt_range] = sill*(1.5*(dist1/range)-0.5*(dist1**3./range**3.))
    else:
        theovar[smlt_range] = 0.

    # 2. For distances greater than the range
    gtt_range          = (distances > range)
    theovar[gtt_range] = sill
    return theovar


def calc_expvar(range, sill, distances):
    """ Calculates the exponential variogram.
    """
    if range != 0.:
        theovar = sill*(1.-np.exp(-distances/range))
    else:
        theovar = np.zeros(distances.shape[0])
    return theovar


def calc_matvar(v, range, sill, distances):
    """
    Matern variogram family:
    v = 0.5 --> Exponential Model
    v = inf --> Gaussian Model
    """
    # for v > 100 shit happens --> use Gaussian model
    if v > 100:
        theovar = calc_gaussvar(range, sill, distances)
    elif v == 0.5:
        theovar = calc_expvar(range, sill, distances)
    else:
        bessel = sps.kv     # modified bessel function of second kind of order v
        gamma  = sps.gamma  # Gamma function

        factor1 = 1./((2.**(v-1.))*gamma(v))
        factor2 = (distances/range)**v
        factor3 = bessel(v, distances/range)
        theovar = sill*(1.-(factor1*factor2*factor3))

        # set nan-values at distances=0 to sill
        # theovar[distances>range]=sill
        theovar[distances<0.000001]=0.
    return theovar


def calc_gaussvar(range, sill, distances):
    """ Calculates the gaussian variogram.
    """
    if range != 0.:
        theovar = sill*(1.-np.exp(-(distances**2.)/(range**2)))
    else:
        theovar = np.zeros(distances.shape[0])
    return theovar


def calc_nuggetvar(sill, distances):
    """ Calculates the nugget variogram.
    """
    theovar    = np.ones(distances.shape[0])
    theovar[:] = np.nan
    theovar[:] = sill
    return theovar