import floris
import floris.layout_visualization
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


class FlorisCutPlane:
    ax: Axes
    fig: Figure
    canvas: FigureCanvasAgg

    cmap: str
    resolution: tuple[int, int]

    def __init__(self, resolution: tuple[int, int], cmap: str = "Purples_r") -> None:
        plt.ioff()
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasAgg(self.fig)
        self.resolution = resolution
        self.cmap = cmap

    def as_rgb(self, fmodel: floris.FlorisModel) -> np.ndarray:
        """TODO: Does not work yet

        Traceback (most recent call last):
          File "D:\Documents\Projects\wake-control-gym\wake_control_gym\main.py", line 51, in <module>
            main()
          File "D:\Documents\Projects\wake-control-gym\wake_control_gym\main.py", line 45, in main
            obs, reward, _, _, info = env.step(yaw_config)
          File "D:\Documents\Projects\wake-control-gym\wake_control_gym\wake_control_env.py", line 210, in step
            frame = self.simulator.render()
          File "D:\Documents\Projects\wake-control-gym\wake_control_gym\adapters\floris\floris_adapter.py", line 262, in render
            return self.visualisation.as_rgb(self.fmodel)
          File "D:\Documents\Projects\wake-control-gym\wake_control_gym\adapters\floris\visualisation.py", line 26, in as_rgb
            horizontal_plane = fmodel.calculate_horizontal_plane(
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\floris_model.py", line 1111, in calculate_horizontal_plane
            fmodel_viz.set_for_viz(findex_for_viz, solver_settings)
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\floris_model.py", line 986, in set_for_viz
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\floris_model.py", line 413, in set
            self._reinitialize(
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\floris_model.py", line 245, in _reinitialize
            self.core = Core.from_dict(floris_dict)
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\type_dec.py", line 227, in from_dict
            return cls(**kwargs)
          File "<attrs generated init floris.core.core.Core>", line 13, in __init__
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\core\core.py", line 115, in __attrs_post_init__
            self.grid = FlowFieldPlanarGrid(
          File "<attrs generated init floris.core.grid.FlowFieldPlanarGrid>", line 18, in __init__
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\core\grid.py", line 521, in __attrs_post_init__
            self.set_grid()
          File "D:\Documents\Projects\wake-control-gym\.venv\lib\site-packages\floris\core\grid.py", line 554, in set_grid
            float(self.planar_coordinate) - 10.0,
        TypeError: only length-1 arrays can be converted to Python scalars
        """
        horizontal_plane = fmodel.calculate_horizontal_plane(
            fmodel.core.farm.hub_heights.flatten()[0],
            *self.resolution,
        )
        self.ax.clear()
        floris.visualize_cut_plane(horizontal_plane, ax=self.ax, cmap=self.cmap)
        floris.layout_visualization.plot_turbine_rotors(fmodel, ax=self.ax)
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
