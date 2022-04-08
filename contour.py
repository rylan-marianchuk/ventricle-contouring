import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from utils import cumulative_curve_length, bounds, contour_density_distance, first_flood_fill_size

class MaskToContour():

    def __init__(self, debug=False, dPhi=0.01, dR=0.5, contour_density=100, save_to_disk=False):
        """
        Construct a contour generator object
        :param debug: (bool) whether to generate figures showing the contour overlayed on mask and image
        :param dPhi: (float) increment of the angle phi when casting rays in polar coordinates
        :param dR: (float) increment of the radius when casting rays in polar coordinates
        :param contourDensity: (int) number of points contained in both contours
        :param save_to_disk: (bool) whether to save this image to the disk or show it viewable on web
        """
        self.dPhi = dPhi
        self.dR = dR
        self.debug = debug
        self.contour_density = contour_density
        self.save_to_disk = save_to_disk


    def __call__(self, lumen_mask, myo_mask, img_overlay=None, out_name=None):
        """
        Obtain the contours of epicaridum, endocardium, and the location of the apex, and quality parameters given the binary masks

        :param lumen_mask:  (ndarray), shape=(N, M), dtype=int
                            1 assigned to every pixel within the blood volume of the ventricle (lumen), 0 elsewhere
        :param myo_mask:    (ndarray), shape=(N, M), dtype=int
                            1 assigned to only pixels on the lining of the ventricle (myocardium), 0 elsewhere
        :param img_overlay: (ndarray), shape=(N, M), dtype=uint16, MRI derived initial image before segmentation
        :param out_name:    (str) the filename to save the overlayed contour image as

        :return --

        endo_contour:   (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                        endocardium contour
        epi_contour:    (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                        epicardium contour
        apex:           (ndarray), shape=(2,) the coordinate of the apex, lying on the epicardium contour
        quality:        (dict) a collection of statistics populated during contour generation:
            -  endo_equidist     the distance between all points in the endo_contour
            -  epi_equidist      the distance between all points in the epi_contour
            -  loops_left        10 - times needed to move epi_contour due to distance clashes (too close)
            -  prop_first_flood  percentage of 1's filled on the myocardium mask when flood filling from an arbitrary 1 start
            -  start_phi         the angle in radians from initial arm centroid -> (1, 0) to terminal arm centroid -> endo_contour[0]
            -  end_phi           the angle in radians from initial arm centroid -> (1, 0) to terminal arm centroid -> endo_contour[-1]
            -  epi_moved_dist    the total distance (1 unit is 1 pixel) of all points moved automatically in the epi_contour due to distance clashes
            -  epi_moved_count   the total number of all points moved automatically in the epi_contour due to distance clashes (not unique, can include duplicate points over iterations)
        """
        # A Dictionary holding quality parameters to return
        quality = {}

        # Get center of ventricle
        centroid = np.argwhere(lumen_mask == 1).sum(0) / np.count_nonzero(lumen_mask)
        centroid = np.flip(centroid)

        # Cast rays counter clockwise and return the phi of first all zero ray after finding ventricle
        start_endo_phi = self.get_edge_phi(centroid, myo_mask)

        # Go clockwise and populate the blocked contour of both endo and epi
        endo_blocked, epi_blocked = self.acquire_endo_epi_pixels(start_endo_phi, centroid, myo_mask, quality)

        # Get equi-distant point clouds from the blocked point sets
        endo_contour, epi_contour = self.interp_contours(endo_blocked, epi_blocked)

        # Fix the distances of endo and epi so that there is at least a 1 unit gap between them, resampling on each fix
        loops_left = 10
        while not self.move_epi(endo_contour, epi_contour, centroid, quality) and loops_left > 0:
            endo_contour, epi_contour = self.interp_contours(endo_contour, epi_contour)
            loops_left -= 1

        endo_contour, epi_contour = self.interp_contours(endo_contour, epi_contour)
        endo_contour, epi_contour = self.interp_contours(endo_contour, epi_contour)

        # Get apex
        apex, ref = self.get_apex(epi_contour)

        if self.debug:
            # Print the standard deviation of all point distances
            #dists = np.array([np.linalg.norm(endo_contour[i] - endo_contour[i + 1]) for i in range(self.contour_density - 1)])
            #print(dists.std())

            # Color the blocked contours
            #for j, i in endo_blocked:
            #    myo_mask[i, j] = 20

            #for j, i in epi_blocked:
            #    myo_mask[i, j] = 40
            # Display the images with contours overlayed
            self.display(img_overlay, myo_mask, endo_contour, epi_contour, apex, ref, out_name)

        quality["endo_equidist"] = contour_density_distance(endo_contour)
        quality["epi_equidist"] = contour_density_distance(epi_contour)
        quality["loops_left"] = loops_left
        quality["prop_first_flood"] = first_flood_fill_size(myo_mask) / np.count_nonzero(myo_mask)
        return endo_contour, epi_contour, apex, ref, centroid, quality


    def move_epi(self, endoContour, epiContour, centroid, quality):
        """
        Go through the neighbours of each point in the epiContour, and check if they are at least 1 unit away from all
        endo neighbours.

        If an epiContour point is too close, move it one unit away in the direction of the normal to that endo point.
        A normal to a point is defined by the sum of the normals of its connecting line segments.

        :param endoContour:  (ndarray), shape=(self.pointCloudDensity, 2)  ordered, each row is a coordinate of the equidistant
                      endocardium contour
        :param epiContour:  (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                      epicardium contour
        :param centroid: (ndarray) shape=(2,) the coordinate of the ventricle's centroid
        :param quality: (dict) key: (str) description of quality parameter -> val
        :return:  (bool) whether a point in the epiContour was too close and had to be moved
        """

        radius = 3
        all_far = True
        quality["epi_moved_count"] = 0
        quality["epi_moved_dist"] = 0
        for i in range(self.contour_density):
            # Using the radius get the indices to check surrounding
            check = [j for j in range(i-radius, i+radius+1) if 0 <= j < self.contour_density]
            for check_i in check:
                nrm = np.linalg.norm(endoContour[check_i] - epiContour[i])
                # Move if too close or epi is on the inside
                if nrm < 1:
                    quality["epi_moved_count"] += 1
                    if all_far: all_far = False

                    # Get direction to move by adding the two normals of the neighbouring line segments
                    # If this is a left edge point, its left normal is the zero vector
                    if check_i - 1 < 0:
                        normal_l = np.array([0, 0])
                    else:
                        normal_l = np.flip(endoContour[check_i - 1] - endoContour[check_i]) * [-1, 1]

                        # Flip the normal if its pointing the wrong way. The dot product with the vector from i to the centroid
                        # should be negative, otherwise negate it
                        if np.dot(normal_l, centroid - endoContour[check_i]) > 0:
                            normal_l *= -1

                    # Same as above for right size
                    if check_i + 1 >= self.contour_density:
                        normal_r = np.array([0, 0])
                    else:
                        normal_r = np.flip(endoContour[check_i + 1] - endoContour[check_i]) * [-1, 1]
                        if np.dot(normal_r, centroid - endoContour[check_i]) > 0:
                            normal_r *= -1

                    # Assign the new point - adding the normalized vector that is the sum of the two normals.

                    newLoc = endoContour[check_i] + ((normal_r + normal_l) / np.linalg.norm(normal_r + normal_l)) * 1.002
                    quality["epi_moved_dist"] += np.linalg.norm(epiContour[i] - newLoc)
                    epiContour[i] = newLoc
                    # Can we break here?

        return all_far


    def get_apex(self, epi_contour):
        """
        Acquire the coordinate of the apex lying along the epiContour
        :param epi_contour: (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                          epicardium contour
        :return: (ndarray), shape=(2,) the coordinate of the apex, lying on the epicardium contour
                 (ndarray), shape=(2,) the coordinate of the ref, middle point of the base
        """
        # Get the middle point of the base
        ref = (epi_contour[0] - epi_contour[-1]) / 2
        ref += epi_contour[-1]

        # Get all distance from the middle of the base to each epiContour point. The apex is equal to the point holding
        # the longest distance
        apex_ind = np.argmax([np.linalg.norm(epi_contour[i] - ref) for i in range(self.contour_density)])
        return epi_contour[apex_ind], ref


    def interp_contours(self, old_endo, old_epi):
        """
        Modify the given contour using an interpolation call, constraining the points in contour to be 100, increasing
        its equi-distance and smoothness

        :param old_endo: (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the endocardium contour
        :param old_epi: (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the epicardium contour
        :return:
        """
        # Initialize new containers for the contours
        endo_contour = np.zeros(shape=(self.contour_density, 2))
        epi_contour = np.zeros(shape=(self.contour_density, 2))

        # Cumulative endoBlocked Curve Length
        cum_endo_cl = cumulative_curve_length(old_endo)
        # Cumulative epiBlocked Curve Length
        cum_epi_cl = cumulative_curve_length(old_epi)

        # Using np.interp for acquisition of new equidistant points at a given density
        endo_interped_x = np.interp(x=np.linspace(0, cum_endo_cl[-1], self.contour_density),
                                    xp=cum_endo_cl,
                                    fp=np.array(old_endo)[:, 0])

        endo_interped_y = np.interp(x=np.linspace(0, cum_endo_cl[-1], self.contour_density),
                                    xp=cum_endo_cl,
                                    fp=np.array(old_endo)[:, 1])

        epi_interped_x = np.interp(x=np.linspace(0, cum_epi_cl[-1], self.contour_density),
                                   xp=cum_epi_cl,
                                   fp=np.array(old_epi)[:, 0])

        epi_interped_y = np.interp(x=np.linspace(0, cum_epi_cl[-1], self.contour_density),
                                   xp=cum_epi_cl,
                                   fp=np.array(old_epi)[:, 1])

        # Update the contour containers with the newly interpolated values
        epi_contour[:,0] = epi_interped_x
        epi_contour[:,1] = epi_interped_y
        endo_contour[:,0] = endo_interped_x
        endo_contour[:,1] = endo_interped_y

        return endo_contour, epi_contour


    def acquire_endo_epi_pixels(self, start_endo_phi, centroid, mask, quality):
        """
        Starting from a given phi, linearly iterate through 2pi radians, casting rays at dPhi and assigning the
        blocked endo and epi lining.

        :param start_endo_phi: the identified angle from the centroid to start gathering the endo & epicardium
        :param centroid: (ndarray) shape=(2,), the coordinate of the ventricle's centroid
        :param mask: (ndarray) shape=(N, M), dtype=  1 assigned to only pixels on the lining of the ventricle, 0 elsewhere
        :param quality: (dict) key: (str) description of quality parameter -> val
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
            ray, ray_indices = self.get_ray(phi, centroid, mask)
            if np.count_nonzero(ray) == 0:
                if not started: continue
                quality["start_phi"] = start_endo_phi
                quality["end_phi"] = phi
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


    def get_ray(self, phi, centroid, mask):
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


    def get_edge_phi(self, centroid, mask):
        """
        Cast rays in a counter clockwise direction to obtain the angle from the centroid to start separating endo and epi
        :param centroid: (ndarray) shape=(2,) the coordinate of the ventricle's centroid
        :return: (float) phi
        """
        hit_wall = False
        start_phi = np.pi / 2
        for phi in np.linspace(start_phi, start_phi + 2*np.pi, int(2*np.pi / self.dPhi)):
            ray, ray_indices = self.get_ray(phi, centroid, mask)
            if np.count_nonzero(ray) == 0:
                if not hit_wall: continue
                return phi
            hit_wall = True
        raise Exception("Ventricle is enclosed and could not find starting point for contouring")


    def display(self, img, myo_mask, endo_contour=None, epi_contour=None, apex=None, ref=None, out_name=None):
        """
        Use plotly to generate figures with the contours overlayed on the mask and image

        :param img: (ndarray), shape=(N, M), dtype=uint16, MRI derived initial image before segmentation, for overlay
        :param myo_mask: (ndarray), shape=(N, M), dtype=  1 assigned to only pixels on the lining of the ventricle, 0 elsewhere
        :param endo_contour:  (ndarray), shape=(self.pointCloudDensity, 2)  ordered, each row is a coordinate of the equidistant
                      endocardium contour
        :param epi_contour:  (ndarray), shape=(self.pointCloudDensity, 2) ordered, each row is a coordinate of the equidistant
                      epicardium contour
        :param apex: (ndarray) shape=(2,) the coordinate of the ventricle's apex
        :param ref: (ndarray) shape=(2,) the coordinate of the middle ventricle base
        :return: None display the plotly images
        """
        img = img.copy()
        for ij in np.argwhere(myo_mask != 0):
            i,j = ij
            r,g,b = img[i,j]
            new = np.array([min(255.0, r * 1.15), max(0.0, g * 0.85), max(0.0, b * 0.85)])
            img[i,j] = new

        fig = px.imshow(img, origin="lower")
        if endo_contour is not None:
            fig.add_trace(go.Scatter(x=endo_contour[:, 0], y=endo_contour[:, 1], mode='markers+lines',
                                     marker=dict(color='#01497c', size=8), name="endo"))
        if epi_contour is not None:
            fig.add_trace(go.Scatter(x=epi_contour[:, 0], y=epi_contour[:, 1], mode='markers+lines',
                                     marker=dict(color='#89c2d9', size=8), name="epi"))
        if apex is not None:
            fig.add_trace(go.Scatter(x=[apex[0], ref[0]], y=[apex[1], ref[1]], marker=dict(color="#277da1", size=20),
                                     name="apex"))

        if self.save_to_disk:
            fig.write_image(out_name + ".png", width=2500, height=2500)
        else:
            fig.show()

