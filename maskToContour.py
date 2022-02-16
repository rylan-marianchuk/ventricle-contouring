import numpy as np
import plotly.graph_objs as go
import plotly.express as px

class MaskToContour():

    def __init__(self, debug=False, dPhi=0.01, dR=0.5, pointCloudDensity=100):
        self.dPhi = dPhi
        self.dR = dR
        self.debug = debug
        self.pointCloudDensity = pointCloudDensity


    def __call__(self, solid_mask, myo_mask):
        centroid_y, centroid_x = np.argwhere(solid_mask==1).sum(0) / np.count_nonzero(solid_mask)
        start_endo_phi = self.getEdgePhi(centroid_x, centroid_y, myo_mask)
        if self.debug:
            self.overlayContour(myo_mask)
        endo_blocked, epi_blocked, firstRay, lastRay = self.acquireEndoEpiBlocks(start_endo_phi, centroid_x, centroid_y, myo_mask)

        if self.debug:
            for i, j in endo_blocked:
                myo_mask[i, j] = 20

            for i, j in epi_blocked:
                myo_mask[i, j] = 40
            self.overlayContour(myo_mask)



        endo_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))
        epi_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))

        self.populatePointClouds(endo_pointcloud, epi_pointcloud, endo_blocked, epi_blocked)

        apex, ref = self.getApex(endo_pointcloud)

        self.overlayContour(myo_mask, endo_pointcloud, epi_pointcloud, apex, ref)
        return endo_pointcloud, epi_pointcloud, apex


    def bounds(self, i, j, imshape):
        if (0 <= i < imshape[0]) and (0 <= j < imshape[1]):
            return True
        return False


    def curveLength(self, pointCloud):
        """

        :param pointCloud:
        :return:
        """
        cl = 0
        for ind in range(len(pointCloud) - 1):
            base_sq = (pointCloud[ind][0] - pointCloud[ind+1][0])**2
            height_sq = (pointCloud[ind][1] - pointCloud[ind+1][1])**2
            cl += np.sqrt(base_sq + height_sq)
        return cl


    def cumulativeCurveLength(self, blockedPointCloud):
        """

        :param blockedPointCloud:
        :return:
        """
        cumulative = np.zeros(len(blockedPointCloud))

        for i in range(1, cumulative.shape[0]):
            base_sq = (blockedPointCloud[i][0] - blockedPointCloud[i - 1][0]) ** 2
            height_sq = (blockedPointCloud[i][1] - blockedPointCloud[i - 1][1]) ** 2
            dist = np.sqrt(base_sq + height_sq)
            cumulative[i] = cumulative[i-1] + dist
        return cumulative


    def getApex(self, epi_pointcloud):
        """

        :param endo_pointcloud:
        :param epi_pointcloud:
        :return:
        """
        ref = (epi_pointcloud[0] - epi_pointcloud[-1]) / 2
        ref += epi_pointcloud[-1]
        apex_ind = np.argmax([np.linalg.norm(epi_pointcloud[i] - ref) for i in range(self.pointCloudDensity)])
        return epi_pointcloud[apex_ind], ref


    def populatePointClouds(self, endo_pointcloud, epi_pointcloud, endo_blocked, epi_blocked):
        """

        :param endo_pointcloud:
        :param epi_pointcloud:
        :param endo_blocked:
        :param epi_blocked:
        :return:
        """
        #reduction_factor = len(endo_blocked) / 100
        #x = np.array([pair[0] for pair in endo_blocked])
        #y = np.array([pair[1] for pair in endo_blocked])
        #xvals = np.linspace(0, self.curveLength(endo_blocked), 100)
        #yinterp = np.interp(xvals, x, y)
        #endo_pointcloud[0] = xvals
        #endo_pointcloud[1] = yinterp


        endo_increment_old = self.curveLength(endo_blocked) / self.pointCloudDensity
        epi_increment_old = self.curveLength(epi_blocked) / self.pointCloudDensity

        cumEndoCL = self.cumulativeCurveLength(endo_blocked)
        cumEpiCL = self.cumulativeCurveLength(epi_blocked)



        endo_increment = cumEndoCL[-1] / self.pointCloudDensity
        epi_increment = cumEpiCL[-1] / self.pointCloudDensity

        endo_pointcloud[0] = np.array(endo_blocked[0])
        epi_pointcloud[0] = np.array(epi_blocked[0])
        current_x, current_y = endo_blocked[0]
        for current_ind in range(1, 100):
            argS = np.argsort(np.abs(cumEndoCL - current_ind*endo_increment))
            goto_A = argS[0]
            goto_B = argS[1]
            D = []
            norms = []
            for point_x in np.linspace(endo_blocked[goto_A][0], endo_blocked[goto_B][0], 20):
                for point_y in np.linspace(endo_blocked[goto_A][1], endo_blocked[goto_B][1], 20):
                    norms.append(np.linalg.norm(np.array([point_x, point_y]) - np.array([current_x, current_y])))
                    D.append(np.array((point_x, point_y)))

            bestnormi = np.argmin(np.abs(np.array(norms) - endo_increment))
            endo_pointcloud[current_ind] = D[bestnormi]
            current_x, current_y = D[bestnormi]

        current_x, current_y = epi_blocked[0]
        for current_ind in range(1, 100):
            argS = np.argsort(np.abs(cumEpiCL - current_ind*epi_increment))
            goto_A = argS[0]
            goto_B = argS[1]
            D = []
            norms = []
            for point_x in np.linspace(epi_blocked[goto_A][0], epi_blocked[goto_B][0], 20):
                for point_y in np.linspace(epi_blocked[goto_A][1], epi_blocked[goto_B][1], 20):
                    norms.append(np.linalg.norm(np.array([point_x, point_y]) - np.array([current_x, current_y])))
                    D.append(np.array((point_x, point_y)))

            bestnormi = np.argmin(np.abs(np.array(norms) - epi_increment))
            epi_pointcloud[current_ind] = D[bestnormi]
            current_x, current_y = D[bestnormi]
        """
        for i in range(1, self.pointCloudDensity):
            L = []
            for phi in np.linspace(0, 2*np.pi, int(2*np.pi/self.dPhi)):
                x_reach = endo_increment * np.cos(phi)
                y_reach = endo_increment * np.sin(phi)
                x_out, y_out = endo_blocked[current][0] + x_reach, endo_blocked[current][1] - y_reach
                xy_out = np.array([x_out, y_out])
                for other in endo_blocked[current+1:current+15]:
                    L.append((other, np.linalg.norm(np.array(other) - xy_out)))
            sorted(L, key=lambda x:x[1])
            endo_pointcloud[i] = np.array(L[0][0]) + 0.25
            current = endo_blocked.index(L[0][0])
        """

        return


    def acquireEndoEpiBlocks(self, start_endo_phi, centroid_x, centroid_y, mask):
        """

        :param phi:
        :param centroid_x:
        :param centroid_y:
        :param mask:
        :return:
        """
        firstRay = None
        lastRay = None
        started = False
        endo = {}
        epi = {}
        phis = np.linspace(start_endo_phi - self.dPhi, start_endo_phi - 2*np.pi, int(2*np.pi / self.dPhi) - 1)
        for i,phi in enumerate((phis)):
            ray, ray_indices = self.getRay(phi, centroid_x, centroid_y, mask)
            if np.count_nonzero(ray) == 0:
                if not started: continue
                lastRay = self.getRay(phis[i-1], centroid_x, centroid_y, mask)
                break
            if not started: started = True
            if firstRay is None:
                firstRay = ray, ray_indices

            argwhere = np.argwhere(ray == 1).flatten()

            ij_endo = ray_indices[argwhere[0]]
            ij_epi = ray_indices[argwhere[-1]]
            if ij_endo not in endo:
                endo[ij_endo] = 1
            if ij_epi not in epi:
                epi[ij_epi] = 1

        return list(endo.keys()), list(epi.keys()), firstRay, lastRay


    def getRay(self, phi, centroid_x, centroid_y, mask):
        """

        :param phi:
        :param centroid_x:
        :param centroid_y:
        :param mask:
        :return: (ndarray) shape=(N,) vector of the ray,
        """
        ray_vals = []
        inds = {}
        imshape = mask.shape
        maxR = np.sqrt(imshape[0]**2 + imshape[1]**2)
        for r in np.linspace(0, maxR, int(maxR / self.dR)):
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            j, i = round(centroid_x + x), round(centroid_y - y)
            if not self.bounds(i, j, imshape):
                break
            if (i,j) in inds:
                continue
            inds[(i,j)] = 1
            ray_vals.append(mask[i, j])
        return np.array(ray_vals), list(inds.keys())


    def getEdgePhi(self, centroid_x, centroid_y, mask):
        """

        :param centroid_x:
        :param centroid_y:
        :return: phi
        """
        startPhi = np.pi / 2
        for phi in np.linspace(startPhi, startPhi + 2*np.pi, int(2*np.pi / self.dPhi)):
            ray, ray_indices = self.getRay(phi, centroid_x, centroid_y, mask)
            if np.count_nonzero(ray) == 0:
                if self.debug:
                    mask_c = mask.copy()
                    for i,j in ray_indices:
                        mask_c[i,j] = 2
                    self.overlayContour(mask_c)
                return phi


    def showline(self, img):
        fig = go.Figure(data=go.Heatmap(z=np.flip(img, 0), y=list(range(img.shape[0]))[::-1], x=list(range(img.shape[1]))))
        fig.update_yaxes(title_text="y", type='category')
        fig.update_xaxes(title_text="x", type='category')
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        fig.show()


    def overlayContour(self, mask, endoPointCloud=None, epiPointCloud=None, apex=None, ref=None):
        fig = px.imshow(mask, color_continuous_scale='gray')
        if endoPointCloud is not None:
            fig.add_trace(go.Scatter(x=endoPointCloud[:,1], y=endoPointCloud[:,0], mode='markers+lines', marker=dict(color='#f94144', size=8)))
        if epiPointCloud is not None:
            fig.add_trace(go.Scatter(x=epiPointCloud[:,1], y=epiPointCloud[:,0], mode='markers+lines', marker=dict(color='#43aa8b', size=8)))
        if apex is not None:
            fig.add_trace(go.Scatter(x=[apex[1], ref[1]], y=[apex[0], ref[0]], marker=dict(color="#277da1", size=20)))
        fig.show()
