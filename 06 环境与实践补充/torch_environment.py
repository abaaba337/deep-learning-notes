#%%
# 基础包
import copy
from math import * 
import numpy as np
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
