import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from utils import *

class MaskToContour():

    def __init__(self, debug=False, dPhi=0.01, dR=0.5, contourDensity=100):
        """
        Construct a contour generator object
        :param debug: (bool) whether to generate figures showing the contour overlayed on mask and image
        :param dPhi: (float) increment of the angle phi when casting rays in polar coordinates
        :param dR: (float) increment of the radius when casting rays in polar coordinates
        :param contourDensity: (int) number of points contained in both contours
        """
        self.dPhi = dPhi
        self.dR = dR
        self.debug = debug
        self.contourDensity = contourDensity


    def __call__(self, solid_mask, myo_mask, imgOverlay=None):
        """
        Obtain the contours of epicaridum, endocardium, and the location of the apex, given the binary masks
        :param solid_mask: (ndarray), shape=(N, M), dtype=
                           1 assigned to every pixel within the ventricle, including the lining and its volume, 0 elsewhere
        :param myo_mask: (ndarray), shape=(N, M), dtype=
                           1 assigned to only pixels on the lining of the ventricle, 0 elsewhere
        :param imgOverlay: (ndarray), shape=(N, M), dtype=uint16, MRI derived initial image before segmentation


        :return -
            endoContour:  (ndarray), shape=(self.pointCloudDensity, 2)  ordered, each row is a coordinate of the equidistant
                          endocardium contour
            epiContour:  (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                          epicardium contour
            apex: (ndarray), shape=(2,) the coordinate of the apex, lying on the epicardium contour
        """

        # Get center of ventricle
        centroid = np.argwhere(solid_mask == 1).sum(0) / np.count_nonzero(solid_mask)
        centroid = np.flip(centroid)

        # Cast rays counter clockwise and return the phi of first all zero ray after finding ventricle
        start_endo_phi = self.getEdgePhi(centroid, myo_mask)

        # Go clockwise and populate the blocked contour of both endo and epi
        endoBlocked, epiBlocked = self.acquireEndoEpiBlocks(start_endo_phi, centroid, myo_mask)

        # Get equi-distant point clouds from the blocked point sets
        endoContour, epiContour = self.populatePointCloudsByInterp(endoBlocked, epiBlocked)

        # Fix the distances of endo and epi so that there is at least a 1 unit gap between them, resampling on each fix
        loopsLeft = 10
        while not self.moveEpi(endoContour, epiContour, centroid) and loopsLeft > 0:
            endoContour, epiContour = self.populatePointCloudsByInterp(endoContour, epiContour)
            loopsLeft -= 1

        endoContour, epiContour = self.populatePointCloudsByInterp(endoContour, epiContour)
        endoContour, epiContour = self.populatePointCloudsByInterp(endoContour, epiContour)

        # Get apex
        apex, ref = self.getApex(epiContour)

        if self.debug:
            # Print the standard deviation of all point distances
            dists = np.array([np.linalg.norm(endoContour[i] - endoContour[i + 1]) for i in range(self.contourDensity - 1)])
            print(dists.std())

            # Color the blocked contours
            for j, i in endoBlocked:
                myo_mask[i, j] = 20

            for j, i in epiBlocked:
                myo_mask[i, j] = 40
            # Display the images with contours overlayed
            self.display(imgOverlay, myo_mask[int(centroid[1])-50:int(centroid[1])+50, int(centroid[0])-50:int(centroid[0])+50], endoContour, epiContour, apex, ref)

        return endoContour, epiContour, apex


    def moveEpi(self, endoContour, epiContour, centroid):
        """
        Go through the neighbours of each point in the epiContour, and check if they are at least 1 unit away from all
        endo neighbours.

        If an epiContour point is too close, move it one unit away in the direction of the normal to that endo point.
        A normal to a point is defined by the sum of the normals of its connecting line segments.

        endoContour:  (ndarray), shape=(self.pointCloudDensity, 2)  ordered, each row is a coordinate of the equidistant
                      endocardium contour
        epiContour:  (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                      epicardium contour
        :return:  (bool) whether a point in the epiContour was too close and had to be moved
        """

        radius = 3
        all_far = True
        for i in range(self.contourDensity):
            # Using the radius get the indices to check surrounding
            check = [j for j in range(i-radius, i+radius+1) if 0 <= j < self.contourDensity]
            for check_i in check:
                nrm = np.linalg.norm(endoContour[check_i] - epiContour[i])

                if nrm < 1:
                    if all_far: all_far = False

                    # Get direction to move by adding the two normals of the neighbouring line segments
                    # If this is a left edge point, its left normal is the zero vector
                    if check_i - 1 < 0:
                        normalL = np.array([0, 0])
                    else:
                        normalL = np.flip(endoContour[check_i - 1] - endoContour[check_i]) * [-1, 1]

                        # Flip the normal if its pointing the wrong way. The dot product with the vector from i to the centroid
                        # should be negative, otherwise negate it
                        if np.dot(normalL, centroid - endoContour[check_i]) > 0:
                            normalL *= -1

                    # Same as above for right size
                    if check_i + 1 >= self.contourDensity:
                        normalR = np.array([0, 0])
                    else:
                        normalR = np.flip(endoContour[check_i + 1] - endoContour[check_i]) * [-1, 1]
                        if np.dot(normalR, centroid - endoContour[check_i]) > 0:
                            normalR *= -1

                    # Assign the new point - adding the normalized vector that is the sum of the two normals.
                    epiContour[i] = endoContour[check_i] + ((normalR + normalL) / np.linalg.norm(normalR + normalL)) * 1.002
                    # Can we break here?

        return all_far


    def getApex(self, epiContour):
        """
        Acquire the coordinate of the apex lying along the epiContour
        :param epiContour: (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                          epicardium contour
        :return: (ndarray), shape=(2,) the coordinate of the apex, lying on the epicardium contour
                 (ndarray), shape=(2,) the coordinate of the ref, middle point of the base
        """
        # Get the middle point of the base
        ref = (epiContour[0] - epiContour[-1]) / 2
        ref += epiContour[-1]

        # Get all distance from the middle of the base to each epiContour point. The apex is equal to the point holding
        # the longest distance
        apex_ind = np.argmax([np.linalg.norm(epiContour[i] - ref) for i in range(self.contourDensity)])
        return epiContour[apex_ind], ref


    def populatePointCloudsByInterp(self, oldEndo, oldEpi):
        """
        Modify the given contour using an interpolation call, constraining the points in contour to be 100, increasing
        its equi-distance and smoothness

        :param oldEndo: (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the endocardium contour
        :param oldEpi: (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the epicardium contour
        :return:
        """
        # Initialize new containers for the contours
        endoContour = np.zeros(shape=(self.contourDensity, 2))
        epiContour = np.zeros(shape=(self.contourDensity, 2))

        # Cumulative endoBlocked Curve Length
        cumEndoCL = cumulativeCurveLength(oldEndo)
        # Cumulative epiBlocked Curve Length
        cumEpiCL = cumulativeCurveLength(oldEpi)

        # Using np.interp for acquisition of new equidistant points at a given density
        endo_interped_x = np.interp(x=np.linspace(0, cumEndoCL[-1], self.contourDensity),
                                    xp=cumEndoCL,
                                    fp=np.array(oldEndo)[:, 0])

        endo_interped_y = np.interp(x=np.linspace(0, cumEndoCL[-1], self.contourDensity),
                                    xp=cumEndoCL,
                                    fp=np.array(oldEndo)[:, 1])

        epi_interped_x = np.interp(x=np.linspace(0, cumEpiCL[-1], self.contourDensity),
                                   xp=cumEpiCL,
                                   fp=np.array(oldEpi)[:, 0])

        epi_interped_y = np.interp(x=np.linspace(0, cumEpiCL[-1], self.contourDensity),
                                   xp=cumEpiCL,
                                   fp=np.array(oldEpi)[:, 1])

        # Update the contour containers with the newly interpolated values
        epiContour[:,0] = epi_interped_x
        epiContour[:,1] = epi_interped_y
        endoContour[:,0] = endo_interped_x
        endoContour[:,1] = endo_interped_y

        return endoContour, epiContour


    def acquireEndoEpiBlocks(self, start_endo_phi, centroid, mask):
        """
        Starting from a given phi, linearly iterate through 2pi radians, casting rays at dPhi and assigning the
        blocked endo and epi lining.

        :param phi:
        :param centroid: (ndarray) shape=(2,), the coordinate of the ventricle's centroid
        :param mask: (ndarray) shape=(N, M), dtype=  1 assigned to only pixels on the lining of the ventricle, 0 elsewhere
        :return:
            endoBlocked:  (ndarray) shape=(unknown, 2)  ordered, each row is an INTEGER coordinate of the
                      endocardium contour
            epiBlocked:  (ndarray) shape=(unknown, 2) ordered, each row is an INTEGER coordinate of the
                      epicardium contour
        """
        started = False
        # Using dictionaries for uniqueness and constant lookup, order also preserved.
        # Keys: (i,j) tupled integers (int = 'Blocked') of the endocardium & epicardium
        endo = {}
        epi = {}
        phis = np.linspace(start_endo_phi, start_endo_phi - 2*np.pi, int(2*np.pi / self.dPhi) - 1)
        for i,phi in enumerate((phis)):
            ray, ray_indices = self.getRay(phi, centroid, mask)
            if np.count_nonzero(ray) == 0:
                if not started: continue
                break

            if not started: started = True

            argwhere = np.argwhere(ray == 1).flatten()

            ij_endo = ray_indices[argwhere[0]]
            if ij_endo not in endo:
                endo[ij_endo] = 1

            # Don't assign endo and epi the same point
            if ray_indices[argwhere[-1]] == ray_indices[argwhere[0]]:
                continue

            ij_epi = ray_indices[argwhere[-1]]
            if ij_epi not in epi:
                epi[ij_epi] = 1

        return np.array(list(endo.keys())), np.array(list(epi.keys()))


    def getRay(self, phi, centroid, mask):
        """
        Generic raycast - given an angle and a start pixel coordinate, obtain a vector of the ray to the border of image

        :param phi: (float) the angle for the polar coordinate to cast this ray at
        :param centroid: (ndarray) shape=(2,) the coordinate of the ventricle's centroid
        :param mask: (ndarray) shape=(N, M), dtype=  1 assigned to only pixels on the lining of the ventricle, 0 elsewhere
        :return:
            (ndarray) shape=(N,) vector of the ray,
            (list) of tuples of the ray as (j,i) euclidean space (origin at bottom left of Q1), or (i,j) in image space (origin at top left)
        """
        # Hold the image values of the ray
        ray_vals = []
        # Hash map. Keys: (j,i) indices, Values are irrelevant. Using this structure for constant lookup and uniqueness
        inds = {}
        imshape = mask.shape

        # Maximum possible radius is the diagonal of the image
        maxR = np.sqrt(imshape[0]**2 + imshape[1]**2)

        for r in np.linspace(0, maxR, int(maxR / self.dR)):
            # Polar coords
            x = r * np.cos(phi)
            y = r * np.sin(phi)

            # Get index of the polar coords by basting to ints
            j, i = round(centroid[0] + x), round(centroid[1] + y)

            # Is this outside the image? Stop, done the ray if so
            if not bounds(i, j, imshape):
                break
            if (j, i) in inds:
                continue
            inds[(j,i)] = 1
            ray_vals.append(mask[i, j])

        # In Python 3.7+ , dictionaries are ordered, hence the indices here line up,
        # that is, np.array(ray_vals)[i] will be the image value of the coordinate list(inds.keys())[i]
        return np.array(ray_vals), list(inds.keys())


    def getEdgePhi(self, centroid, mask):
        """
        Cast rays in a counter clockwise direction to obtain the angle from the centroid to start separating endo and epi
        :param centroid: (ndarray) shape=(2,) the coordinate of the ventricle's centroid
        :return: (float) phi
        """
        hitWall = False
        startPhi = np.pi / 2
        for phi in np.linspace(startPhi, startPhi + 2*np.pi, int(2*np.pi / self.dPhi)):
            ray, ray_indices = self.getRay(phi, centroid, mask)
            if np.count_nonzero(ray) == 0:
                if not hitWall: continue
                return phi
            hitWall = True
        raise Exception("Ventricle is enclosed and could not find starting point for contouring")


    def display(self, img, mask, endoContour=None, epiContour=None, apex=None, ref=None):
        """
        Use plotly to generate figures with the contours overlayed on the mask and image

        :param img: (ndarray), shape=(N, M), dtype=uint16, MRI derived initial image before segmentation, for overlay
        :param myo_mask: (ndarray), shape=(N, M), dtype=  1 assigned to only pixels on the lining of the ventricle, 0 elsewhere
        endoContour:  (ndarray), shape=(self.pointCloudDensity, 2)  ordered, each row is a coordinate of the equidistant
                      endocardium contour
        epiContour:  (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                      epicardium contour
        :param apex: (ndarray) shape=(2,) the coordinate of the ventricle's apex
        :param ref: (ndarray) shape=(2,) the coordinate of the middle ventricle base
        :return: None display the plotly images
        """
        for base_img in (img, mask):
            if base_img is None: continue
            fig = px.imshow(base_img, color_continuous_scale='gray', origin="lower")
            if endoContour is not None:
                fig.add_trace(go.Scatter(x=endoContour[:, 0], y=endoContour[:, 1], mode='markers+lines', marker=dict(color='#f94144', size=8)))
            if epiContour is not None:
                fig.add_trace(go.Scatter(x=epiContour[:, 0], y=epiContour[:, 1], mode='markers+lines', marker=dict(color='#43aa8b', size=8)))
            if apex is not None:
                fig.add_trace(go.Scatter(x=[apex[0], ref[0]], y=[apex[1], ref[1]], marker=dict(color="#277da1", size=20)))

            fig.show()
