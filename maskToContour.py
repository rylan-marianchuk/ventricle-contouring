import numpy as np
import plotly.graph_objs as go
import plotly.express as px

class MaskToContour():

    def __init__(self, debug=False, dPhi=0.01, dR=0.5):
        self.dPhi = dPhi
        self.dR = dR
        self.debug = debug

    def __call__(self, solid_mask, myo_mask):
        centroid_y, centroid_x = np.argwhere(solid_mask==1).sum(0) / np.count_nonzero(solid_mask)
        start_endo_phi = self.getEdgePhi(centroid_x, centroid_y, myo_mask)
        self.overlayContour(myo_mask)
        endo, epi, firstRay, lastRay = self.acquireEndoEpi(start_endo_phi, centroid_x, centroid_y, myo_mask)
        for i,j in endo:
            myo_mask[i,j] = 20

        for i,j in epi:
            myo_mask[i,j] = 40

        self.overlayContour(myo_mask)





    def bounds(self, i, j, imshape):
        if (0 <= i < imshape[0]) and (0 <= j < imshape[1]):
            return True
        return False


    def acquireEndoEpi(self, start_endo_phi, centroid_x, centroid_y, mask):
        """

        :param phi:
        :param centroid_x:
        :param centroid_y:
        :param mask:
        :return:
        """
        firstRay = None
        lastRay = None
        endo = {}
        epi = {}
        phis = np.linspace(start_endo_phi - self.dPhi, start_endo_phi - 2*np.pi, int(2*np.pi / self.dPhi) - 1)
        for i,phi in enumerate((phis)):
            ray, ray_indices = self.getRay(phi, centroid_x, centroid_y, mask)
            if np.count_nonzero(ray) == 0:
                if i == 0: continue
                lastRay = self.getRay(phis[i-1], centroid_x, centroid_y, mask)
                break

            if firstRay is None:
                firstRay = ray, ray_indices

            argwhere = np.argwhere(ray == 1).flatten()

            ij_endo = ray_indices[argwhere[0]]
            ij_epi = ray_indices[argwhere[-1]]
            if ij_endo not in endo:
                endo[ij_endo] = 1
            if ij_epi not in epi:
                epi[ij_epi] = 1

        return endo, epi, firstRay, lastRay

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


    def overlayContour(self, mask):
        fig = px.imshow(mask, color_continuous_scale='gray')
        #fig.add_trace(go.Scatter(x=[63, 69], y=[109, 130], marker=dict(color='red', size=8)))
        fig.show()