import numpy as numpy
import torch
import Post.Plots as Pplot
import copy
import time
import torch
import random 

def Mixed_Initial_Training(Model_u, Model_du, lmbda,mu, TrialCoordinates):
    print()