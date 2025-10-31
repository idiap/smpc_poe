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


import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--vis", type=bool, default=True, help="If to visualize the simulation (default: True)")
parser.add_argument("--mppi-noise", type=float, default=0.5, help="The amount of noise (covariance) for MPPI (default: 0.5)")
parser.add_argument("--mppi-temp", type=float, default=0.05, help="temprature term for the mppi methpd (default: 0.05)")
parser.add_argument("--state-cost", type=float, default=10.0, help="Weight for state cost (default: 10.0)")
parser.add_argument("--terminal-cost", type=float, default=1000.0, help="Weight for the terminal state cost (default: 1000.0)")
parser.add_argument("--action-cost", type=float, default=0.001, help="Weight for action cost (default: 0.001)")
parser.add_argument("--sdf-cost", type=float, default=0.0, help="Weight for distance cost (default: 0.0)")
parser.add_argument("--collision-cost", type=float, default=100000.0, help="Weight for collision cost (default: 100000.0)")
parser.add_argument("--exploration-cost", type=float, default=1.0, help="Weight for exploration cost, avoiding stucking situation (default: 1.0)")
parser.add_argument("--terminal-cond", type=float, default=0.1, help="distance in meters to be considered as reached (default: 0.1)")
parser.add_argument("--ttgo-exploration", type=float, default=0.0, help="Exploration term for ttgo method, 0.0 for highest exploration and 1 for the lowest exploration (default: 0.0)")
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
import os
cur_dir = os.getcwd()
from nvidia_wrapper import IsaacWrapper
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch
import time
import zerorpc
import io
from pytorch3d import transforms

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)

def calcQuatFromYaw(yaw_angles):
    # yaw_angles: (N,) tensor
    # Create euler angles tensor [0, 0, yaw]
    euler = torch.zeros((yaw_angles.shape[0], 3), device=yaw_angles.device)
    euler[:, 2] = yaw_angles  # yaw around Z
    
    # Make rotation matrix
    rot_matrix = transforms.euler_angles_to_matrix(euler, 'XYZ')
    
    # Convert to quaternion (wxyz)
    quat_wxyz = transforms.matrix_to_quaternion(rot_matrix)
    
    # Invert quaternion (to reverse the earlier inversion)
    quat_inv = transforms.quaternion_invert(quat_wxyz)
    
    # Reorder to xyzw to match your original input convention
    quat_xyzw = torch.cat((quat_inv[:,1:], quat_inv[:,0].view([-1,1])), dim=1)
    return quat_xyzw


urdf_root = './../URDF'
agents = {"urdf":["ball_2D_mustard.urdf"],"init_pos":[[0.2,0.0,0.03]],"init_rot":[[0,0,0,1]]}

# the moving slider should be the first one
sliders = {"urdf":["mustard.urdf","mustard_goal.urdf"],"init_pos":[[0.0,0.0,0.037],[0.0,0.0,0.037]],"init_rot":[[0,0,0,1.],[0,0,0,1.]],"base_fix":[False,True]}

num_envs = 1

isaac = IsaacWrapper(urdf_root=urdf_root,dt=.05, agents = agents,sliders = sliders,visualize = True,num_envs=num_envs, add_plane=True)

isaac.initEnvs()
isaac.initSim()


isaac.gym.simulate(isaac.sim)
isaac.gym.fetch_results(isaac.sim, True)
isaac.gym.refresh_dof_state_tensor(isaac.sim)
isaac.gym.refresh_actor_root_state_tensor(isaac.sim)
isaac.gym.refresh_net_contact_force_tensor(isaac.sim)
zero_root_tensor = isaac.root_state_tensor.clone()
zero_dof_tensor = isaac.dof_states.clone()


planner = zerorpc.Client()
planner.connect("tcp://127.0.0.1:4242")

target_points = torch.tensor([[-.2, 0.],
                              [-.2,-.2],
                              [-.2, .2],
                              [ 0., .2],
                              [ 0.,-.2],
                              [ .2, 0.],
                              [ .2, .2],
                              [ .2, -.2]
                            ]).to('cuda:0')


du1 = 2/isaac.factors[5].shape[1]
sigma =args.mppi_noise/du1
sigma_mppi = args.mppi_noise * torch.eye(2).to('cuda:0')
sigma_mppi_data = torch_to_bytes(sigma_mppi)

wx_t = torch.tensor([args.state_cost]).to('cuda:0')
wx_T = torch.tensor([args.terminal_cost]).to('cuda:0')
wu_t = torch.tensor([args.action_cost]).to('cuda:0')
w_sdf = torch.tensor([args.sdf_cost]).to('cuda:0')
w_coll = torch.tensor([args.collision_cost]).to('cuda:0')
w_exp = torch.tensor([args.exploration_cost]).to('cuda:0')
term_cond = torch.tensor([args.terminal_cond]).to('cuda:0')

# TTMPPI
ttmppi_cost = np.zeros([8,4])
ttmppi_action_cost = np.zeros([8,4])
ttmppi_pose_cost = np.zeros([8,4])
ttmppi_rot_cost = np.zeros([8,4])
ttmppi_steps = np.zeros([8,4])
for tar in range(target_points.shape[0]):
    for orient in range(4):
        print("checking for condition ", tar, orient)
        action0 = torch.zeros([isaac.num_dofs,]).to('cuda:0')
        planner.setCostTerms(torch_to_bytes(target_points[tar,:]),torch_to_bytes(torch.tensor([(orient-1)*torch.pi/4]).to('cuda:0')),torch_to_bytes(wx_t),torch_to_bytes(wx_T),torch_to_bytes(wu_t),torch_to_bytes(w_sdf),torch_to_bytes(w_coll),torch_to_bytes(w_exp),torch_to_bytes(term_cond))
        isaac.setCostTerms(torch_to_bytes(target_points[tar,:]),torch_to_bytes(torch.tensor([(orient-1)*torch.pi/4]).to('cuda:0')),torch_to_bytes(wx_t),torch_to_bytes(wx_T),torch_to_bytes(wu_t),torch_to_bytes(w_sdf),torch_to_bytes(w_coll),torch_to_bytes(w_exp),torch_to_bytes(term_cond))
        planner.setupTTMPPI(sigma,args.ttgo_exploration,args.mppi_temp)
        quat_orient = calcQuatFromYaw(torch.tensor([(orient-1)*torch.pi/4]).to('cuda:0'))
        zero_root_tensor[2,:2] = target_points[tar,:]
        zero_root_tensor[2,2] += 0.03  # to be above the ground
        zero_root_tensor[2,3] = quat_orient[0,0]
        zero_root_tensor[2,4] = quat_orient[0,1]
        zero_root_tensor[2,5] = quat_orient[0,2]
        zero_root_tensor[2,6] = quat_orient[0,3]
        isaac.resetStateAll(torch_to_bytes(zero_root_tensor),torch_to_bytes(zero_dof_tensor))
        for i in range(500):
            t1 = time.time()
            print("step: ", i , end = '\r')
            isaac.gym.set_dof_velocity_target_tensor(isaac.sim,gymtorch.unwrap_tensor(action0))
            isaac.gym.simulate(isaac.sim)
            isaac.gym.fetch_results(isaac.sim, True)
            isaac.gym.step_graphics(isaac.sim)
            isaac.gym.draw_viewer(isaac.viewer, isaac.sim, False)
            isaac.gym.refresh_dof_state_tensor(isaac.sim)
            isaac.gym.refresh_actor_root_state_tensor(isaac.sim)
            isaac.gym.refresh_net_contact_force_tensor(isaac.sim)
            ttmppi_cost[tar,orient] += isaac.calcCost(scale = 1).item()
            ttmppi_action_cost[tar,orient] += torch.norm(action0).item()
            ttmppi_pose_cost[tar,orient] += isaac.calcPosCost().item()
            ttmppi_rot_cost[tar,orient] += isaac.calcOrnCost()
            obj_pos = isaac.calcObjPos()
            orn_cost= isaac.calcOrnCost()
            res_target = obj_pos - target_points[tar,:]
            if torch.norm(res_target)<term_cond.item() and orn_cost < 0.2:
                break
            planner.resetStateAll(torch_to_bytes(isaac.root_state_tensor[:2,:]),torch_to_bytes(isaac.dof_states))
            action0_data = planner.TTMPPI()
            action0 = bytes_to_torch(action0_data)
        ttmppi_steps[tar,orient] = i
        print("reached the target at {} steps:",i)
        print("the cost value is ",ttmppi_cost[tar,orient])
        isaac.resetStateAll(torch_to_bytes(zero_root_tensor),torch_to_bytes(zero_dof_tensor))



print('Done!')