# this python script is used to store parameters 
# that can be called by scripts in the scripts folder

import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Cases import *

zero_tol = 1e-16
round_tol = 3

rayspeed = 1
partical_dense = 100
reflect_penalty = 10
reflect_limit = 10
dying_limit = 50
num_rays = 100