import floris
import floris.layout_visualization
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


def get_figsize(
    fmodel: floris.FlorisModel, image_width: int, resolution: tuple[float, float], dpi: int
) -> tuple[float, float]:
    horizontal_plane = fmodel.calculate_horizontal_plane(
        fmodel.core.farm.hub_heights.flatten()[0], resolution
    )
    size = horizontal_plane.df[['x1', 'x2']].max() - horizontal_plane.df[['x1', 'x2']].min()
    fig_width = image_width / dpi
    fig_height = fig_width / (size.x1 / size.x2)
    return fig_width, fig_height


class FlorisCutPlane:
    ax: Axes
    fig: Figure
    canvas: FigureCanvasAgg

    image_width: int
    resolution: tuple[int, int]
    dpi: int
    cmap: str

    def __init__(
        self,
        fmodel: floris.FlorisModel,
        image_width: int = 640,
        resolution: int | tuple[int, int] = 64,
        dpi: int = 80,
        cmap: str = "Purples_r",
    ) -> None:
        # plt.ioff()
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        self.image_width = image_width
        figsize = get_figsize(fmodel, image_width, resolution, dpi)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_axis_off()
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        self.fig.tight_layout()
        self.canvas = FigureCanvasAgg(self.fig)
        self.cmap = cmap

    def render_rgb(self, fmodel: floris.FlorisModel) -> np.ndarray:
        horizontal_plane = fmodel.calculate_horizontal_plane(
            fmodel.core.farm.hub_heights.flatten()[0],
            *self.resolution,
        )
        self.ax.clear()
        floris.visualize_cut_plane(horizontal_plane, ax=self.ax, cmap=self.cmap)
        floris.layout_visualization.plot_turbine_rotors(fmodel, ax=self.ax)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
