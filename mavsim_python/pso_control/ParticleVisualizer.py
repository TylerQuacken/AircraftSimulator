import numpy as np
import matplotlib.pyplot as plt


class ParticleVisualizer():
    def __init__(self, controller):
        self.controller = controller
        U = controller.U
        self.numSims = U.shape[0]
        self.numKnotPoints = U.shape[1]
        self.nDim = U.shape[2]
        self.xLim = [controller.uMin[0], controller.uMax[0]]
        if self.nDim == 1:
            inc = -2. / controller.numSims
            self.yVal = np.arange(1., -1., inc)

        self.fig, self.axs = plt.subplots(self.numKnotPoints)
        self.fig.suptitle('Particles for each knot')

        # norm = mpl.colors.Normalize()
        # colormap = cm.Reds
        # self.map = cm.ScalarMappable(norm=norm, cmap=colormap)
        # colors = self.map.to_rgba(-controller.costs)

    def update_data(self, U):
        for i in range(self.numKnotPoints):
            Ui = U[:, i, :]
            # self.yVal = np.argsort(controller.costs) / self.numSims
            self.yVal = self.controller.costs

            if self.nDim == 1:
                self.axs[i].clear()
                colors = ['r.', 'g.', 'b.', 'c.', 'm.', 'y.', 'k.', 'w.']
                for j in range(self.controller.numSwarms):
                    lowI = self.controller.swarmIndex[j]
                    highI = self.controller.swarmIndex[j + 1]
                    self.axs[i].plot(Ui[lowI:highI, 0], self.yVal[lowI:highI],
                                     colors[j])
                # self.axs[i].axis([-100, 100, -1, 1])
                self.axs[i].set_xlim(self.xLim)
                self.axs[i].set_ylim([0, np.max(self.controller.costs)])
            elif self.nDim == 2:
                self.axs[i].plot(Ui[:, 0], Ui[:, 1], 'r.')

        plt.pause(0.001)
