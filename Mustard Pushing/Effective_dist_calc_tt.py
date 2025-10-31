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

import sys
cur_dir = sys.path[0]
from nvidia_wrapper import IsaacWrapper
from isaacgym.torch_utils import *
import torch
import time
import io
import tensorly
import time
from pytorch3d import transforms


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)
urdf_root = './../URDF'

agents = {"urdf":["ball_2D_mustard.urdf"],"init_pos":[[0.4,0.0,0.03]],"init_rot":[[0,0,0,1]]}
sliders = {"urdf":["mustard.urdf"],"init_pos":[[0.0,0.0,0.037]],"init_rot":[[0.,0.,0.,1.]],"base_fix":[False]}

num_envs = 100
num_horizon = 64

def yawtoQuat(yaw_angles):
    rpy = torch.zeros([num_envs,3]).to('cuda:0')
    rpy[:,2] = yaw_angles
    des_rot_matrix = transforms.euler_angles_to_matrix(rpy,'XYZ')
    des_quat = transforms.matrix_to_quaternion(des_rot_matrix)
    return des_quat

isaac = IsaacWrapper(urdf_root=urdf_root,dt=.05, agents = agents, sliders = sliders,visualize = False,num_envs=num_envs, add_plane=True)

isaac.initEnvs()
isaac.initSim()


isaac.gym.simulate(isaac.sim)
isaac.gym.fetch_results(isaac.sim, True)
isaac.gym.refresh_dof_state_tensor(isaac.sim)
isaac.gym.refresh_actor_root_state_tensor(isaac.sim)
isaac.gym.refresh_net_contact_force_tensor(isaac.sim)
zero_root_tensor = isaac.root_state_tensor.clone()
zero_dof_tensor = isaac.dof_states.clone()

tic = time.time()
domain = [torch.linspace(-.3,.3,10).to('cuda:0')]*2 + [torch.linspace(-torch.pi,torch.pi,10).to('cuda:0')] + [torch.linspace(-.4,.4,10).to('cuda:0')]*2 + [torch.linspace(-1.,1.,10).to('cuda:0')]*2
X_obj, Y_obj, YAW_obj, X_rob, Y_rob, DX, DY  = torch.meshgrid(domain[0],domain[1],domain[2],domain[3],domain[4],domain[5],domain[6])

X_obj = X_obj.unsqueeze(dim = 0)
Y_obj = Y_obj.unsqueeze(dim = 0)
YAW_obj = YAW_obj.unsqueeze(dim = 0)
X_rob = X_rob.unsqueeze(dim = 0)
Y_rob = Y_rob.unsqueeze(dim = 0)
DX = DX.unsqueeze(dim = 0)
DY = DY.unsqueeze(dim = 0)
Inputs = torch.cat((X_obj,Y_obj,YAW_obj, X_rob, Y_rob, DX,DY),dim = 0)
inputs = Inputs.view([7,-1]).transpose(dim0=0,dim1=1)

results = torch.zeros([10**7,]).to('cuda:0')
n_epochs = int(inputs.shape[0]/num_envs)


root_state_tensor = torch.zeros([num_envs*2,13],dtype = torch.float32).to('cuda:0')
root_state_tensor[::2,6] = 1
root_state_tensor[::2,2] = 0.03
root_state_tensor[::2,0] = 0.4
dof_state_tensor = torch.zeros([num_envs*2,2],dtype = torch.float32).to('cuda:0')
train = True
if train:
    for i in range(n_epochs):
        progress = i/n_epochs*100
        print('Gathering Data: {:.2f} %'.format(progress), end='\r')
        inputs_i = inputs[i*num_envs:(i+1)*num_envs]
        action = inputs_i[:,5:].to(torch.float32).contiguous()
        desired_robot_state = inputs_i[:,3:5].to(torch.float32)
        desired_obj_state = inputs_i[:,:2].to(torch.float32)
        desired_yaw_obj = inputs_i[:,2].to(torch.float32)
        desired_quat_obj = yawtoQuat(desired_yaw_obj)
        root_state_tensor[1::2,:2] = desired_obj_state 
        root_state_tensor[1::2,6] = desired_quat_obj[:,0]
        root_state_tensor[1::2,3] = desired_quat_obj[:,1]
        root_state_tensor[1::2,4] = desired_quat_obj[:,2]
        root_state_tensor[1::2,5] = desired_quat_obj[:,3]
        dof_state_tensor[:,0] = desired_robot_state.reshape([-1,]) - root_state_tensor[::2,:2].reshape([-1,])
        results_i = isaac.calcEffectivenessMustard(torch_to_bytes(root_state_tensor),torch_to_bytes(dof_state_tensor),torch_to_bytes(action))
        isaac.visPoints()
        results[i*num_envs:(i+1)*num_envs] = results_i



    results = results.reshape([10,10,10,10,10,10,10])
    print('Saving raw results...')
    np.save(cur_dir + '/raw_results.npy',results.to('cpu'))
results  = np.load(cur_dir + '/raw_results.npy')
print('Calculating TT decomposition...')
factors = tensorly.decomposition.tensor_train(results,30000)

for i in range(7):
    np.save(cur_dir + '/factor_{}.npy'.format(i),factors[i])

print('Factors saved.')