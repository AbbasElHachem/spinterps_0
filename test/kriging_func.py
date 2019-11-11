import numpy as np
import scipy as sp

import variogram_func as vf

class Kriging:

    def setcontrols(self, controls):
        """ Set the controls for the kriging procedure """
        # Controls are the coordinates of the gauges (places where data is
        # available)
        self.controls = controls

    def setcontrolvalues(self, controlvalues):
        """ Set the values for the controls for the kriging procedure """
        self.controlvalues = np.array([controlvalues]).flatten()

    def setcontrols_ed(self, controls_ed):
        """ Set the values for the controls for the kriging procedure """
        self.controls_ed = controls_ed

    def settargets(self, targets):
        """ Set the targets (coordinates) for the kriging procedure """
        self.targets = targets

    def settargets_ed(self, targets_ed):
        """ Set the targets external drift for the kriging procedure """
        self.targets_ed = np.array([targets_ed]).flatten()

    def krige_values(self, variogram, method='ordinary kriging',
                     min_stations=10, positive_weights='yes'):
        """
        """
        self.calc_weights(variogram, method, min_stations,
                          positive_weights)

        # Calculate the kriged values
        self.kriged_values = np.ones(self.targets.shape[0])
        self.kriged_values[:] = np.nan

        # Loop over the target values to calculate them
        for target_idx in range(self.targets.shape[0]):
            self.kriged_values[target_idx] = np.sum(
               self.kriging_weights[target_idx]*self.kriging_values[target_idx])


    def calc_weights(self, variogram, method='ordinary kriging',
                     min_stations=10, positive_weights='no'):
        """
        method - 'ordinary kriging'
               - 'external drift kriging'
        min_stations - number of stations, which will be included in the kriging
                       --> at least this number of stations should be available
        positive_weights='yes' --> only positive weights allowed
        """
        # Ordinary Kriging and External Drift Kriging
        # -------------------------------------------

        # 1. Check, if there are any double stations in the controls, if so,
        #    delete one of them
        # ---------------------
        # Create a tree for all controls
        controltree = sp.spatial.cKDTree(self.controls)
        qtree       = controltree.query(self.targets[0],
                                        self.controls.shape[0])

        # Check, if there are equal distances, if so theres a double control
        doubles_oneside = np.where((qtree[0][:-1] == qtree[0][1:])==True)[0]

        if doubles_oneside.shape[0] > 0:
            # Loop over double pairs
            for idx_double in range(doubles_oneside.shape[0]):
                # Replace one of the double control pairs with -9999999, to have
                # it always out of the kriging range
                self.controls[qtree[1][doubles_oneside[idx_double]]] = -9999999


        # 2. Initialise an array which will contain the temporal control values
        # ----------------------------------------------------------------------
        controls_temp    = np.ones((self.controls.shape[0], 2))
        controls_temp[:] = np.nan

        # 3. Perform the kriging
        # ----------------------
        # Spatial indices (of stations) for which a control value exists
        spatial_idx = np.where(np.isnan(self.controlvalues) == False)[0]

        # If there are at least "min_stations" values, proceed with the kriging,
        # otherwise all kriged values will be nan

        if spatial_idx.shape[0] >= min_stations:
            # Control values and coordinates, which are available
            controls_temp      = self.controls[spatial_idx]
            controlvalues_temp = self.controlvalues[spatial_idx]

            if method == 'external drift kriging':
                # Controls external drift
                controls_ed_temp = self.controls_ed[spatial_idx]

            # Set up the KDTree for the coordinates of the stations, to get the
            # relation of the distances between the controls
            controltree = sp.spatial.cKDTree(controls_temp)

            # Indices of the stations used for the kriging
            self.idx_krigingstats = np.ones((self.targets.shape[0],
                                             min_stations), dtype=np.int)
            self.idx_krigingstats[:] = -99
            # Kriging weights
            self.kriging_weights = np.ones((self.targets.shape[0],
                                            min_stations))
            self.kriging_weights[:] = np.nan
            # Kriging values
            self.kriging_values = np.ones((self.targets.shape[0],
                                           min_stations))
            self.kriging_values[:] = np.nan

            # Loop over the target values to calculate them
            for target_idx in range(self.targets.shape[0]):
                target = self.targets[target_idx]

                if method == 'external drift kriging':
                    # Targets external drift
                    target_ed = self.targets_ed[target_idx]

                # Query the controls-tree for the "min_stations" nearest points
                # to the target qtree contains the distances and the indices of
                # the nearest stations beginning with the nearest station
                qtree = controltree.query(target, min_stations)

                # Initialize temporary controls and external drift lists by
                # selecting the points returned by the tree query --> for the
                # regarded target
                controls_temptar = controls_temp[qtree[1]]

                if method == 'external drift kriging':
                    controls_ed_temptar   = controls_ed_temp[qtree[1]]

                # Also initialise the according control values
                controlvalues_temptar = controlvalues_temp[qtree[1]]


                # Calculate the kriging matrix
                #---------------------------------------------------------------
                # Calculate the distance matrix (euclidian distances)
                distance_matrix = sp.spatial.distance.cdist(controls_temptar,
                                                            controls_temptar,
                                                            'euclidean')
                # Distance to the closest station (not important for the kriging
                # procedure, only nice to know)
                self.dist_closeststation = distance_matrix[1, 0]

                # Variogram values of the distance matrix values
                variogram.calc_theovar(distance_matrix.flatten())
                var_distmatrix = variogram.theovar

                # Reshape the variogram values of the distance matrix
                var_distmatrix = var_distmatrix.reshape(
                             distance_matrix.shape[0], distance_matrix.shape[1])

                # Get the ordinary kriging matrix
                # 1. Add row of ones after the last row of var_distmatrix
                ok_matrix  = np.vstack((var_distmatrix,
                                        np.ones(distance_matrix.shape[0])))
                # 2. Add column
                add_column = np.append(np.ones(distance_matrix.shape[0]), 0.)
                add_column = add_column.reshape(1, add_column.shape[0]).T
                ok_matrix  = np.concatenate((ok_matrix, add_column), axis=1)

                if method == 'external drift kriging':
                    # Calculate the external drift kriging matrix
                    addrow_edk  = np.append(controls_ed_temptar, 0.)
                    # 1. Add row of controls_ed+0 after the last row of
                    #    ok_matrix
                    edk_matrix  = np.vstack((ok_matrix, addrow_edk))
                    # 2. Add column of controls_ed+0+0 after the last column of
                    #    edk_matrix
                    addcol_edk  = np.append(addrow_edk, 0.)
                    addcol_edk  = addcol_edk.reshape(1, addcol_edk.shape[0]).T
                    edk_matrix  = np.concatenate((edk_matrix, addcol_edk),
                                                 axis=1)

                # Calculate the right hand side of the kriging system of
                # equation
                # 1. Calculate distances from the control values to the target
                #    value --> vector
                dist_target = sp.spatial.distance.cdist(controls_temptar,
                                                        target.reshape(1,2),
                                                        'euclidean')
                # 2. Calculate the variogram values of the distances to the
                #    target
                variogram.calc_theovar(dist_target.flatten())
                var_target = variogram.theovar

                # 3. Get the vector of the right hand side of the kriging system
                #    Add the value 1
                rhs = np.append(var_target, np.array([1.]))

                if method == 'external drift kriging':
                    # Add the external drift of the target to the vector
                    rhs  = np.append(rhs, np.array([target_ed]))
                rhs  = rhs.reshape(rhs.shape[0], 1)

                # If there are any nan values in the ok matrix raise Exception
                if np.where(np.isnan(ok_matrix)==True)[0].shape[0] > 0:
                    raise Exception('nan values in ok_matrix !')

                # Solve the system of equations to calculate the weights for the
                # target value calculation
                # 1. Ordinary Kriging
                if method == 'ordinary kriging':
                    weights = np.linalg.solve(ok_matrix, rhs)
                    weights = weights[:-1]

                    # Only positive weights are allowed ?
                    if positive_weights=='yes' and np.sum(
                                             [weights > 0]) != weights.shape[0]:
                        weights = sp.optimize.nnls(ok_matrix, rhs.flatten())[0]
                        weights = weights.reshape(weights.shape[0], 1)
                        weights = weights[:-1]

                # 2. External drift kriging
                if method == 'external drift kriging':
                    weights = np.linalg.solve(edk_matrix, rhs)
                    weights = weights[:-2]

                    # Only positive weights are allowed ?
                    if positive_weights=='yes' and np.sum(
                                             [weights > 0]) != weights.shape[0]:
                        weights = sp.optimize.nnls(edk_matrix, rhs.flatten())[0]
                        weights = weights.reshape(weights.shape[0], 1)
                        weights = weights[:-2]

                # Indices of the stations used for the kriging
                self.idx_krigingstats[target_idx] = qtree[1]
                # Kriging weights
                self.kriging_weights[target_idx] = weights.flatten()
                # Kriging values
                self.kriging_values[target_idx] = (
                                                controlvalues_temptar.flatten())

        # No values available
        else:
            raise Exception('Not enough control values available')


