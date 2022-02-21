import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from utils import *

class MaskToContour():

    def __init__(self, debug=False, dPhi=0.01, dR=0.5, pointCloudDensity=100):
        self.dPhi = dPhi
        self.dR = dR
        self.debug = debug
        self.pointCloudDensity = pointCloudDensity


    def __call__(self, solid_mask, myo_mask, img):
        # Get center of ventricle
        centroid_y, centroid_x = np.argwhere(solid_mask == 1).sum(0) / np.count_nonzero(solid_mask)

        # Cast rays counter clockwise and return the phi of first all zero ray after finding ventricle
        start_endo_phi = self.getEdgePhi(centroid_x, centroid_y, myo_mask)

        # Go clockwise and populate the blocked contour of both endo and epi
        endo_blocked, epi_blocked, firstRay, lastRay = self.acquireEndoEpiBlocks(start_endo_phi, centroid_x, centroid_y, myo_mask)

        if self.debug:
            for j, i in endo_blocked:
                myo_mask[i, j] = 20

            for j, i in epi_blocked:
                myo_mask[i, j] = 40
            #self.display(None, myo_mask)

        # Get equi-distant point clouds from the blocked point sets
        endo_pointcloud, epi_pointcloud = self.populatePointClouds2Interp(endo_blocked, epi_blocked)
        endo_pointcloud, epi_pointcloud = self.populatePointClouds2Interp(endo_pointcloud, epi_pointcloud)
        endo_pointcloud, epi_pointcloud = self.populatePointClouds2Interp(endo_pointcloud, epi_pointcloud)
        endo_pointcloud, epi_pointcloud = self.populatePointClouds2Interp(endo_pointcloud, epi_pointcloud)

        #self.moveEpi(endo_pointcloud, epi_pointcloud, np.array([centroid_x, centroid_y]))

        dists = np.array([np.linalg.norm(endo_pointcloud[i] - endo_pointcloud[i+1]) for i in range(self.pointCloudDensity - 1)])
        print(dists.std())


        # Get apex
        apex, ref = self.getApex(epi_pointcloud)

        #if self.debug:
            #self.display(img, myo_mask, endo_pointcloud, epi_pointcloud, apex, ref)

        self.display(img, myo_mask, endo_pointcloud, epi_pointcloud, apex, ref)



        return endo_pointcloud, epi_pointcloud, apex


    def bounds(self, i, j, imshape):
        if (0 <= i < imshape[0]) and (0 <= j < imshape[1]):
            return True
        return False

    def moveEpi(self, endoPointCloud, epiPointCloud, centroid):
        """

        :param endoPointCloud:
        :param epiPointCloud:
        :return:
        """
        radius = 3
        normals = np.zeros(shape=(self.pointCloudDensity, 2))
        for i in range(self.pointCloudDensity):
            check = [j for j in range(i-radius, i+radius+1) if 0 <= j < self.pointCloudDensity]
            for check_i in check:
                nrm = np.linalg.norm(endoPointCloud[check_i] - epiPointCloud[i])
                if nrm < 1:
                    # Get direction to move
                    if check_i - 1 < 0:
                        normalL = np.array([0, 0])
                    else:
                        normalL = np.flip(endoPointCloud[check_i - 1] - endoPointCloud[check_i]) * [-1, 1]
                        if np.dot(normalL, centroid - endoPointCloud[check_i]) > 0:
                            normalL *= -1
                    if check_i + 1 >= self.pointCloudDensity:
                        normalR = np.array([0, 0])
                    else:
                        normalR = np.flip(endoPointCloud[check_i + 1] - endoPointCloud[check_i]) * [-1, 1]
                        if np.dot(normalR, centroid - endoPointCloud[check_i]) > 0:
                            normalR *= -1

                    epiPointCloud[i] = endoPointCloud[check_i] + ((normalR + normalL) / np.linalg.norm(normalR + normalL)) * 1.002

                    break

        # Now just apply epi_pointcloud + normals
        return normals



    def cumulativeCurveLength(self, blockedPointCloud):

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

    def populatePointClouds2Interp(self, endo_blocked, epi_blocked):
        """
        :param endo_blocked:
        :param epi_blocked:
        :return:
        """

        endo_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))
        epi_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))
        endo_pointcloud[0] = np.array(endo_blocked[0])
        epi_pointcloud[0] = np.array(epi_blocked[0])

        cumEndoCL = self.cumulativeCurveLength(endo_blocked)
        cumEpiCL = self.cumulativeCurveLength(epi_blocked)

        endo_interped_x = np.interp(x=np.linspace(0, cumEndoCL[-1], self.pointCloudDensity),
                  xp=cumEndoCL,
                  fp=np.array(endo_blocked)[:,0])

        endo_interped_y = np.interp(x=np.linspace(0, cumEndoCL[-1], self.pointCloudDensity),
                      xp=cumEndoCL,
                      fp=np.array(endo_blocked)[:,1])

        endo_pointcloud[:,0] = endo_interped_x
        endo_pointcloud[:,1] = endo_interped_y

        epi_interped_x = np.interp(x=np.linspace(0, cumEpiCL[-1], self.pointCloudDensity),
                      xp=cumEpiCL,
                      fp=np.array(epi_blocked)[:,0])

        epi_interped_y = np.interp(x=np.linspace(0, cumEpiCL[-1], self.pointCloudDensity),
                      xp=cumEpiCL,
                      fp=np.array(epi_blocked)[:,1])
        epi_pointcloud[:,0] = epi_interped_x
        epi_pointcloud[:,1] = epi_interped_y
        return endo_pointcloud, epi_pointcloud


    def populatePointClouds(self, endo_blocked, epi_blocked):
        """
        :param endo_blocked:
        :param epi_blocked:
        :return:
        """
        endo_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))
        epi_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))
        endo_pointcloud[0] = np.array(endo_blocked[0])
        epi_pointcloud[0] = np.array(epi_blocked[0])

        cumEndoCL = self.cumulativeCurveLength(endo_blocked)
        cumEpiCL = self.cumulativeCurveLength(epi_blocked)

        endo_increment = cumEndoCL[-1] / self.pointCloudDensity
        epi_increment = cumEpiCL[-1] / self.pointCloudDensity

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

        return endo_pointcloud, epi_pointcloud


    def populatePointClouds3(self, endo_blocked, epi_blocked):
        """
        :param endo_blocked:
        :param epi_blocked:
        :return:
        """

        endo_blocked, epi_blocked = np.flip(np.array(endo_blocked), 1) * [1, -1], np.flip(np.array(epi_blocked), 1)*[1, -1]
        endo_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))
        epi_pointcloud = np.zeros(shape=(self.pointCloudDensity, 2))
        endo_pointcloud[0] = np.flip(np.array(endo_blocked[0])) * [-1, 1]
        epi_pointcloud[0] = np.flip(np.array(epi_blocked[0])) * [-1, 1]

        cumEndoCL = self.cumulativeCurveLength(endo_blocked)
        cumEpiCL = self.cumulativeCurveLength(epi_blocked)

        endo_increment = cumEndoCL[-1] / self.pointCloudDensity
        epi_increment = cumEpiCL[-1] / self.pointCloudDensity

        current_x, current_y = endo_blocked[0]
        for current_ind in range(1, 100):
            argS = np.argsort(np.abs(cumEndoCL - current_ind*endo_increment))
            goto_A = argS[0]
            goto_B = argS[1]
            AB = endo_blocked[goto_B] - endo_blocked[goto_A]
            pts = []
            rejectionNorms = []
            curr2A = endo_blocked[goto_A] - np.array([current_x, current_y])
            curr2B = endo_blocked[goto_B] - np.array([current_x, current_y])
            for theta in np.linspace(np.arctan(curr2A[1]/curr2A[0]), np.arctan(curr2B[1]/curr2B[0]), 200):
                h = np.array([current_x + np.cos(theta)*endo_increment, current_y + np.sin(theta)*endo_increment])
                Ah = h - endo_blocked[goto_A]
                rejectionNorms.append(np.linalg.norm(rejection(AB, Ah)))
                pts.append(h)
            bestArg = np.argmin(rejectionNorms)
            current_x, current_y = pts[bestArg]
            endo_pointcloud[current_ind] = np.flip(pts[bestArg])  * [-1, 1]

        return endo_pointcloud, epi_pointcloud

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
        phis = np.linspace(start_endo_phi, start_endo_phi - 2*np.pi, int(2*np.pi / self.dPhi) - 1)
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
            if ij_endo not in endo:
                endo[ij_endo] = 1
            if ray_indices[argwhere[-1]] == ray_indices[argwhere[0]]:
                continue
            ij_epi = ray_indices[argwhere[-1]]
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
            j, i = round(centroid_x + x), round(centroid_y + y)
            if not self.bounds(i, j, imshape):
                break
            if (j, i) in inds:
                continue
            inds[(j,i)] = 1
            ray_vals.append(mask[i, j])
        return np.array(ray_vals), list(inds.keys())


    def getEdgePhi(self, centroid_x, centroid_y, mask):
        """

        :param centroid_x:
        :param centroid_y:
        :return: phi
        """
        hitWall = False
        startPhi = np.pi / 2
        for phi in np.linspace(startPhi, startPhi + 2*np.pi, int(2*np.pi / self.dPhi)):
            ray, ray_indices = self.getRay(phi, centroid_x, centroid_y, mask)
            if np.count_nonzero(ray) == 0:
                if not hitWall: continue
                return phi
            hitWall = True
        raise Exception("Ventricle is enclosed and could not find starting point for contouring")


    def display(self, img, mask, endoPointCloud=None, epiPointCloud=None, apex=None, ref=None):
        for base_img in (img, mask):
            if base_img is None: continue
            fig = px.imshow(base_img, color_continuous_scale='gray', origin="lower")
            if endoPointCloud is not None:
                fig.add_trace(go.Scatter(x=endoPointCloud[:,0], y=endoPointCloud[:,1], mode='markers+lines', marker=dict(color='#f94144', size=8)))
            if epiPointCloud is not None:
                fig.add_trace(go.Scatter(x=epiPointCloud[:,0], y=epiPointCloud[:,1], mode='markers+lines', marker=dict(color='#43aa8b', size=8)))
            if apex is not None:
                fig.add_trace(go.Scatter(x=[apex[0], ref[0]], y=[apex[1], ref[1]], marker=dict(color="#277da1", size=20)))

            fig.show()
