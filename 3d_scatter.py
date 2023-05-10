from usefull_graphics import *
from structures import *
import numpy as np
from averager import *

data = StreamData("myfile_3d.csv", 501, 981, 81)
resultY_start = advance_to_column(data, "Y Velocity")
scatter_3d_array(resultY_start, treeshold_up = 0.5, treeshold_down = -0.5, normalize=[-1,1], title="original", colorbar=True)