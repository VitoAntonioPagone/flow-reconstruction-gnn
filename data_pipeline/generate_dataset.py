#import matplotlib.pyplot as plt
from pymech.neksuite import readnek
from pymech.dataset import open_dataset
from pymech.dataset import open_mfdataset
from pymech.neksuite import writenek
from pymech.neksuite import readnek
# import xarray as xr
import numpy as np

### Read nek file with pymech
field = readnek('tud0.f00006')
