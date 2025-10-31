"""
    Copyright (c) 2025 Idiap Research Institute, http://www.idiap.ch/
    Written by Amirreza Razmjoo <amirreza.razmjoo@idiap.ch>,

    This file is part of smpc_poe.

    smpc_poe is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    smpc_poe is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with smpc_poe. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
cur_dir = os.getcwd()
import argparse
parser = argparse.ArgumentParser()
    # Define arguments
parser.add_argument("--num-horizon", type=int, default=20,
                        help="Number of horizons (default: 20)")
parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of samples (default: 16)")
    # Parse them
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown
from nvidia_wrapper import IsaacWrapper
from isaacgym.torch_utils import *
import zerorpc
urdf_root = './../URDF'

agents = {"urdf":["ball_2D_mustard.urdf"],"init_pos":[[0.2,0.0,0.03]],"init_rot":[[0,0,0,1]]}
sliders = {"urdf":["mustard.urdf"],"init_pos":[[0.0,0.0,0.037]],"init_rot":[[0,0,0,1.]],"base_fix":[False]}





num_envs = args.num_envs
num_horizon = args.num_horizon


isaac = IsaacWrapper(urdf_root=urdf_root,dt=.05, agents = agents, sliders = sliders,visualize = False,num_envs=num_envs, add_plane=True,num_horizon=num_horizon)

isaac.initEnvs()
isaac.initSim()

s = zerorpc.Server(isaac)
s.bind("tcp://0.0.0.0:4242")
s.run()
