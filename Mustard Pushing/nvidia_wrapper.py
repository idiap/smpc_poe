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
curr_dir = os.getcwd()

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
import time

from tt_utils_ttgo import refine_domain, refine_cores, domain2idx, stochastic_top_k
from pytorch3d import transforms
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


import io

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)

class IsaacWrapper:
    def __init__(self, urdf_root, agents, sliders, dt = 1.0/60.0, visualize = False, add_plane = False, num_envs = 64, env_per_row = None, spacing = 2.0, num_horizon = 15):
        self.args = gymutil.parse_arguments()
        self.gym = gymapi.acquire_gym()
        
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = dt
        self.sim_params.substeps = 2
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = self.args.num_threads
        self.sim_params.physx.use_gpu = self.args.use_gpu

        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline

        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        
        # Initialize the viewer
        self.visualize = visualize
        if visualize:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
            
            self.cam_pos = gymapi.Vec3(4, 3, 3)
            self.cam_target = gymapi.Vec3(-4, -3, 0)
            
        # Add the ground plane
        if add_plane:
            self.plane_params = gymapi.PlaneParams()
            self.plane_params.normal = gymapi.Vec3(0, 0, 1)
            self.gym.add_ground(self.sim, self.plane_params)

        try:
            self.factors = []
            for i in range(7):
                factor_i = np.load(curr_dir + '/factor_{}.npy'.format(i))
                self.factors.append(torch.from_numpy(factor_i).to('cuda:0').to(torch.float32))

            
            
            self.domain = [torch.linspace(-.3,.3,10).to('cuda:0')]*2 + [torch.linspace(-torch.pi,torch.pi,10).to('cuda:0')] + [torch.linspace(-.4,.4,10).to('cuda:0')]*2 + [torch.linspace(-1.,1.,10).to('cuda:0')]*2
            self.domain_state_action = refine_domain(domain=self.domain, site_list=[0,1,2,3,4,5,6],scale_factor=10, device=device)
            self.domain_action = self.domain_state_action[5:7]
            self.factors = refine_cores(tt_cores=self.factors, site_list=[0,1,2,3,4,5,6],scale_factor=10, device=device)
            
            self.cores_gaussian = [torch.arange(self.domain_state_action[5].shape[0],dtype = torch.float32).view(1,-1,1).repeat(self.factors[5].shape[0],1,self.factors[5].shape[2]).to('cuda:0')] + [torch.arange(self.domain_state_action[6].shape[0],dtype = torch.float32).view(1,-1,1).repeat(self.factors[6].shape[0],1,self.factors[6].shape[2]).to('cuda:0')]
        except:
            pass
        
        # Load robot    
        self.asset_root = curr_dir + '/' + urdf_root
        self.agents_props = {"assets":[],"dof_props":[],"default_dof_states":[],"num_dofs":[],"init_pose":[]}
        for i in range(len(agents["urdf"])):   
            asset_file = agents["urdf"][i]
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.flip_visual_attachments = False
            asset_options.armature = 0.01
            asset_options.disable_gravity = False

            asset_agent = self.gym.load_asset(self.sim, self.asset_root, asset_file, asset_options)
            self.agents_props["assets"].append(asset_agent)

            dof_props = self.gym.get_asset_dof_properties(asset_agent)
            lower_limits = dof_props['lower']
            upper_limits = dof_props['upper']
            mids = 0.5 * (lower_limits + upper_limits)
            num_dofs = len(dof_props)


            # set DOF control properties (except grippers)
            dof_props["driveMode"][:num_dofs].fill(gymapi.DOF_MODE_VEL)
            dof_props["stiffness"][:num_dofs].fill(0.0)
            dof_props["damping"][:num_dofs].fill(6000.0)

            default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"][:num_dofs] = mids[:num_dofs]

            init_pose = gymapi.Transform()
            init_pose.p = gymapi.Vec3(agents["init_pos"][i][0], agents["init_pos"][i][1], agents["init_pos"][i][2])
            init_pose.r = gymapi.Quat(agents["init_rot"][i][0], agents["init_rot"][i][1], agents["init_rot"][i][2], agents["init_rot"][i][3])

            self.agents_props["dof_props"].append(dof_props)
            self.agents_props["default_dof_states"].append(default_dof_state)
            self.agents_props["num_dofs"].append(num_dofs)
            self.agents_props["init_pose"].append(init_pose)

        self.num_dofs = sum(self.agents_props["num_dofs"])
        self.n_agents = len(self.agents_props["assets"])
        
        # Load slider   
        self.sliders_props = {"assets":[],"init_pose":[],"base_fix":[]}    
        for i in range(len(sliders["urdf"])):
            asset_file = sliders["urdf"][i]
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = sliders["base_fix"][i]
            asset_options.flip_visual_attachments = False
            asset_options.armature = 0.01
            asset_options.disable_gravity = False
            asset = self.gym.load_asset(self.sim, self.asset_root, asset_file, asset_options)
            
            init_pose = gymapi.Transform()
            init_pose.p = gymapi.Vec3(sliders["init_pos"][i][0], sliders["init_pos"][i][1], sliders["init_pos"][i][2])
            init_pose.r = gymapi.Quat(sliders["init_rot"][i][0], sliders["init_rot"][i][1], sliders["init_rot"][i][2], sliders["init_rot"][i][3])

            self.sliders_props["assets"].append(asset)
            self.sliders_props["init_pose"].append(init_pose)
            self.sliders_props["base_fix"].append(sliders["base_fix"][i])

        self.num_sliders = len(self.sliders_props["assets"])
        self.num_horizon = num_horizon
        self.num_envs = num_envs
        if env_per_row == None:
            env_per_row = int(math.sqrt(self.num_envs))
        self.env_per_row = env_per_row
        self.spacing = spacing
        self.env_lower = gymapi.Vec3(-spacing, -spacing,0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing * 3)
      
        self.state_holder = torch.zeros([self.num_envs,5]).to('cuda:0')

        self.sample_action = torch.zeros([self.num_envs,self.num_horizon,self.num_dofs]).to('cuda:0')
        aligned_points = np.load(curr_dir + '/mustared_aligned_points.npy')
        self.point_clouds = torch.from_numpy(aligned_points).to('cuda:0',dtype = torch.float32).transpose(dim0=0,dim1=1)
        self.point_clouds = self.point_clouds.view([1,self.point_clouds.shape[0],self.point_clouds.shape[1]]).repeat([self.num_envs,1,1])

    def initEnvs(self,):
        self.envs = []
        for i in range(self.num_envs):
            # Create env
            print("created envs:", int(i/self.num_envs* 100), end = "\r")
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.env_per_row)
            self.envs.append(env)

            # Add agents
            for j in range(len(self.agents_props["assets"])):
                asset_handle = self.gym.create_actor(env, self.agents_props["assets"][j],self.agents_props["init_pose"][j], "agent_{}".format(j), i, 0)
                self.gym.set_actor_dof_states(env, asset_handle, self.agents_props["default_dof_states"][j], gymapi.STATE_ALL)
                # Set DOF control properties
                self.gym.set_actor_dof_properties(env, asset_handle, self.agents_props["dof_props"][j])

            # Add slider
            for j in range(len(self.sliders_props["assets"])):
                self.gym.create_actor(env, self.sliders_props["assets"][j], self.sliders_props["init_pose"][j], "slider_{}".format(j), i, 0)

          
        self.middle_env = self.envs[self.num_envs // 2 + self.env_per_row // 2]
        if self.visualize:
            self.gym.viewer_camera_look_at(self.viewer, self.middle_env, self.cam_pos, self.cam_target)

    def initSim(self,):
        self.gym.prepare_sim(self.sim)

        # Rigid body state tensor
        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)

        # DOF state tensor
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, self.num_dofs, 1)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, self.num_dofs, 1)

        self.state_target = torch.zeros([self.num_envs*self.num_dofs,2]).to('cuda:0',dtype = torch.float32)

        self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor)
        self.state_target_position = torch.zeros([self.num_envs,self.num_dofs]).to("cuda:0")

        self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor)

        self.net_cf_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(self.net_cf_tensor)
        self.mean = torch.zeros([self.num_horizon,self.num_dofs]).to('cuda:0')
        self.noise_cov = 2 * torch.eye(self.num_dofs).repeat([self.num_horizon,1,1]).to('cuda:0')

    def Vis(self,duration = 5):
        if not self.visualize:
            raise Exception("Visualization parameter is set to False")
        else:
            n_iter = int(duration/self.sim_params.dt)
            for i in range(n_iter):
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)

                time.sleep(self.sim_params.dt)

    def FK(self,q_base,q_robot):
        
        self.root_state_tensor = q_base
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor))
        self.state_target_position = tensor_clamp(q_robot,self.lower_limits_tensor, self.upper_limits_tensor).to("cuda:0")
        self.state_target[:,0] = self.state_target_position.view([self.num_dofs*self.num_envs,])
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.state_target))
 
        # Step rendering
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        if self.visualize:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
        
        # Get current hand poses
        pos_cur = self.rb_states[self.hand_idxs, :3].view(self.num_envs, -1,3)
        orn_cur = self.rb_states[self.hand_idxs, 3:7].view(self.num_envs, -1,4)
        time.sleep(1)

        return pos_cur, orn_cur
    
    def resetStateAll(self,root_data, dof_data):
        
        root_tensor = bytes_to_torch(root_data)
        dof_tensor = bytes_to_torch(dof_data)
        
        self.root_tensor = root_tensor.repeat([self.num_envs,1]).to('cuda:0')
        self.root_tensor_cur = self.root_tensor.clone()
        self.dof_tensor = dof_tensor.repeat([self.num_envs,1]).to('cuda:0')
        self.dof_tensor_cur = dof_tensor.clone()
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.root_tensor))
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.dof_tensor))
        self.gym.simulate(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.visualize:
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

    def resetStateInd(self,root_data, dof_data):
        root_tensor = bytes_to_torch(root_data)
        dof_tensor = bytes_to_torch(dof_data)
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(root_tensor))
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(dof_tensor))
        self.gym.simulate(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.visualize:
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

    def step(self,action):
        self.gym.set_dof_velocity_target_tensor(self.sim,gymtorch.unwrap_tensor(action))
        self.gym.simulate(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
    def calcRobObjDist(self):
        obj_pos_and_rot =  self.calcObjPosandRot().view([-1,7])
        obj_pos = obj_pos_and_rot[:,:3].unsqueeze(1).repeat([1,self.point_clouds.shape[1],1])
        obj_rot_quat = torch.cat((obj_pos_and_rot[:,6].view([-1,1]),obj_pos_and_rot[:,3:6].view([-1,3])),dim = 1)
        obj_rot_quat_ext = obj_rot_quat.unsqueeze(1).repeat([1,self.point_clouds.shape[1],1])
        transfered_points = transforms.quaternion_apply(obj_rot_quat_ext,self.point_clouds) + obj_pos
        robot_pos = self.calcRobotPos3D(0).view([-1,3]).unsqueeze(1).repeat([1,self.point_clouds.shape[1],1])
        dist_all = torch.norm(robot_pos - transfered_points,dim=2)
        dist, _ = torch.min(dist_all,dim=1)
        return dist
    

    def visPoints(self):
        obj_pos_and_rot =  self.calcObjPosandRot().view([-1,7])
        obj_pos = obj_pos_and_rot[:,:3].unsqueeze(1).repeat([1,self.point_clouds.shape[1],1])
        obj_rot_quat = torch.cat((obj_pos_and_rot[:,6].view([-1,1]),obj_pos_and_rot[:,3:6].view([-1,3])),dim = 1)
        obj_rot_quat_ext = obj_rot_quat.unsqueeze(1).repeat([1,self.point_clouds.shape[1],1])
        transfered_points = transforms.quaternion_apply(obj_rot_quat_ext,self.point_clouds) + obj_pos
        for i in range(self.num_envs):
            points = transfered_points[i,:,:].squeeze()
            points_pre = points[:-1,:]
            points_post = points[1:,:]
            line_points_1D = torch.cat((points_pre,points_post),dim = 1)


        

    def calcEffectivenessMustard(self,root_data, dof_data, action_data):
        action_tensor = bytes_to_torch(action_data)
        self.resetStateInd(root_data,dof_data)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        cur_obj_pos = self.calcObjPos()
        cur_dist_rob_obj = self.calcRobObjDist()
        self.step(action_tensor)
        next_obj_pos = self.calcObjPos()
        next_obj_feas =  1 - self.checkColl(next_obj_pos,scale = 0.75)
        next_dist_rob_obj = self.calcRobObjDist()
        obj_move_length = torch.norm(next_obj_pos - cur_obj_pos,dim = 1)
        got_closer =  (next_dist_rob_obj < (cur_dist_rob_obj + 2e-2)) *1
        obj_move = (obj_move_length >1e-2) * 1
        effectiveness = ((obj_move + got_closer)>0) * 1
        effectiveness = effectiveness * next_obj_feas 
        return effectiveness
    
    def calcEffectivenessMustardFlow(self,root_data, dof_data, action_data):
        action_tensor = bytes_to_torch(action_data)
        self.resetStateInd(root_data,dof_data)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        cur_obj_pos = self.calcObjPos()
        cur_obj_pose = self.calcObjPosandRot()
        cur_dist_rob_obj = self.calcRobObjDist()
        self.step(action_tensor)
        next_obj_pos = self.calcObjPos()
        next_obj_pose = self.calcObjPosandRot()
        jump_next_obj_pos = ((next_obj_pose[:,2] - cur_obj_pose[:,2]) > 2e-2) * 1
        next_obj_feas =  1 - self.checkColl(next_obj_pos)
        next_dist_rob_obj = self.calcRobObjDist()
        obj_move_length = torch.norm(next_obj_pos - cur_obj_pos,dim = 1)
        got_closer =  (next_dist_rob_obj < (cur_dist_rob_obj + 0)) *1
        obj_move = (obj_move_length >1e-2) * 1
        effectiveness = ((obj_move + got_closer)>0) * 1
        effectiveness = effectiveness * next_obj_feas 
        return effectiveness

    def calcEffectiveness(self,root_data, dof_data, action_data):
        action_tensor = bytes_to_torch(action_data)
        self.resetStateInd(root_data,dof_data)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        cur_rob_pos = self.calcRobotPos(0)
        cur_obj_pos = self.calcObjPos()
        cur_dist_rob_obj = torch.norm(cur_obj_pos - cur_rob_pos,dim=1)
        self.step(action_tensor)
        next_rob_pos = self.calcRobotPos(0)
        next_obj_pos = self.calcObjPos()
        next_obj_feas =  1 - self.checkColl(next_obj_pos)
        next_dist_rob_obj = torch.norm(next_obj_pos - next_rob_pos,dim=1)
        obj_move_length = torch.norm(next_obj_pos - cur_obj_pos,dim = 1)
        got_closer =  (next_dist_rob_obj < (cur_dist_rob_obj + 1e-3/5)) *1
        obj_move = (obj_move_length >1e-2) * 1
        effectiveness = ((obj_move + got_closer)>0) * 1
        effectiveness = effectiveness * next_obj_feas
        return effectiveness

    def calcRobotPos(self,agent_index):
        agent_dof = self.dof_states.view([self.num_envs,self.num_dofs,-1])
        agents_root = self.root_state_tensor.view([self.num_envs,-1,13])
        agent_root = agents_root[:,agent_index,:3].squeeze()
        agent_pos = agent_root[:,:2] + agent_dof[:,:,0].squeeze()
        return agent_pos[:,:2].clone()
    
    def calcRobotPos3D(self,agent_index):
        agent_dof = self.dof_states.view([self.num_envs,self.num_dofs,-1])
        agents_root = self.root_state_tensor.view([self.num_envs,-1,13])
        agent_root = agents_root[:,agent_index,:3].squeeze(1)
        agent_pos = agent_root[:,:2] + agent_dof[:,:,0].squeeze()
        agent_pos_3d = torch.cat((agent_pos.view([-1,2]),agent_root[:,2].view([-1,1])),dim = 1)
        return agent_pos_3d.clone()

    def calcObjPos(self):
        obj_pos = self.root_state_tensor.view([self.num_envs,-1,13])[:,self.n_agents,:].squeeze()
        obj_pos = obj_pos.view([-1,13])
        return obj_pos[:,:2].clone()
    
    def calcObjPosandRot(self):
        obj_pos = self.root_state_tensor.view([self.num_envs,-1,13])[:,self.n_agents,:].squeeze()
        obj_pos = obj_pos.view([-1,13])
        return obj_pos[:,:7].clone()
    
    def calcObjYaw(self,quat_xyzw): 
        new_quat_wxyz = torch.cat((quat_xyzw[:,3].view([-1,1]),quat_xyzw[:,:3]),dim=1)
        main_quat_inverse = transforms.quaternion_invert(new_quat_wxyz) 
        des_matrix_calc = transforms.quaternion_to_matrix(main_quat_inverse)
        des_euler_calc = transforms.matrix_to_euler_angles(des_matrix_calc,'XYZ')
        yaw_angles = des_euler_calc[:,2]
        return yaw_angles
    
    def setCostTerms(self,target_pos, target_orn, wx_t, wx_T, wu_t , w_sdf,w_coll,w_exp,termination_cond):
        self.target_pos = bytes_to_torch(target_pos)
        self.target_orn = bytes_to_torch(target_orn)
        self.cos_target_orn = torch.cos(self.target_orn)
        self.sin_target_orn = torch.sin(self.target_orn)
        self.wx_t = bytes_to_torch(wx_t).item()
        self.wx_T = bytes_to_torch(wx_T).item()
        self.wu_t = bytes_to_torch(wu_t).item()
        self.w_sdf = bytes_to_torch(w_sdf).item()
        self.w_coll = bytes_to_torch(w_coll).item()
        self.w_exp = bytes_to_torch(w_exp).item()
        self.term_cond = bytes_to_torch(termination_cond).item() * 0.9

    def checkColl(self,state,scale = 1):
        x_low_check = (state[:,0] < -.4 * scale) * 1
        x_up_check = (state[:,0] > .4 * scale) * 1
        y_low_check = (state[:,1] < -.4* scale) * 1
        y_up_check = (state[:,1] > .4* scale) * 1
        all_check = x_low_check + x_up_check + y_low_check + y_up_check
        coll= (all_check>0)*1
        return coll
    
    def calcPosCost(self):
        obj_pos = self.root_state_tensor.view([self.num_envs,-1,13])[:,self.n_agents,:].squeeze(1)
        res_pos =  self.target_pos - obj_pos[:,:2]
        pos_cost = torch.norm(res_pos,dim=1)
        return pos_cost 
    
    def calcOrnCost(self):
        obj_pos = self.root_state_tensor.view([self.num_envs,-1,13])[:,self.n_agents,:].squeeze(1)
        obj_yaw = self.calcObjYaw(obj_pos[:,3:7])
        cos_obj_yaw = torch.cos(obj_yaw)
        sin_obj_yaw = torch.sin(obj_yaw)
        cos_obj = cos_obj_yaw*self.cos_target_orn + sin_obj_yaw * self.sin_target_orn
        res_orn = torch.acos(torch.clip(cos_obj, -0.99,0.99))
        orn_cost = torch.norm(res_orn.view([1,-1]),dim=0)      
        return orn_cost

    def calcCost(self,scale = 0.6):
        obj_pos = self.root_state_tensor.view([self.num_envs,-1,13])[:,self.n_agents,:].squeeze(1)
        obj_yaw = self.calcObjYaw(obj_pos[:,3:7])
        cos_obj_yaw = torch.cos(obj_yaw)
        sin_obj_yaw = torch.sin(obj_yaw)
        res_pos =  self.target_pos - obj_pos[:,:2]
        cos_obj = cos_obj_yaw*self.cos_target_orn + sin_obj_yaw * self.sin_target_orn
        res_orn = torch.acos(torch.clip(cos_obj, -0.99,0.99))
        pos_cost = torch.norm(res_pos,dim=1)
        orn_cost = torch.norm(res_orn.view([1,-1]),dim=0)        
        cost = self.wx_t * pos_cost + 0.1 * self.wx_t * orn_cost
        agents_dof = self.dof_states.view([self.num_envs,self.num_dofs,-1])
        agents_root = self.root_state_tensor.view([self.num_envs,-1,13])
        dof_count = 0
        coll_boundary_objct = self.checkColl(obj_pos,scale = scale)
        cost += self.w_coll * (1 * coll_boundary_objct)
        for i in range(len(self.agents_props["assets"])):
            agent_root = agents_root[:,i,:3].squeeze(1)
            agent_dof = agents_dof[:,dof_count:dof_count + self.agents_props["num_dofs"][i],:]
            dof_count += self.agents_props["num_dofs"][i]
            agent_pos = agent_root[:,:2] + agent_dof[:,:,0].squeeze()
            coll_boundary_agent = self.checkColl(agent_pos)
            agent_obj_res =  obj_pos[:,:2] - agent_pos[:,:2]
            agent_obj_dist = torch.norm(agent_obj_res,dim=1)
            push_align_agent = torch.sum(agent_obj_res[:,0:2]*res_pos, 1)/(agent_obj_dist*pos_cost+1e-6)
            cost += self.w_coll * ( coll_boundary_agent) #+ self.w_sdf * (self.calcRobObjDist())
        return  cost

    def calcThetaRot(self):
        agent_pos = self.calcRobotPos(0)
        agent_obj_res =  self.zero_obj_pos[:,:2] - agent_pos[:,:2]
        norm_agent_obj_res = torch.norm(agent_obj_res,dim = 1)
        muliplied_norm = norm_agent_obj_res * self.norm_zero_obj_rob
        cos_agnet_obj = torch.sum(-agent_obj_res * self.zero_obj_rob,dim=1)
        normalized_cos = torch.divide(cos_agnet_obj,muliplied_norm + 1e-6)
        normalized_cos = torch.clip(normalized_cos,-0.99,0.99)
        theta_obj_agnet = torch.acos(normalized_cos)
        external_mult = -agent_obj_res[:,0] * self.zero_obj_rob[:,1] + agent_obj_res[:,1]*self.zero_obj_rob[:,0]
        sign = torch.sign(external_mult)
        # print(sign.shape)
        return theta_obj_agnet * sign
    
    def setupTTMPPI(self,sigma,alpha,temp):
        self.sigma  = sigma
        self.alpha = alpha
        self.temp = temp
        self.mean = torch.zeros([self.num_horizon,self.num_dofs]).to('cuda:0')
        self.mean0 = self.mean[0,:]
        self.solver = 'TTMPPI'

    def TTMPPI(self):
        mean_idx = domain2idx(self.mean,self.domain_action,device = 'cuda')
        self.cost = torch.zeros([self.num_envs,]).to('cuda')
        rob_zero_pos = self.calcRobotPos(0)
        self.zero_obj_pos = self.calcObjPos()
        self.zero_obj_rob = rob_zero_pos - self.zero_obj_pos
        self.norm_zero_obj_rob = torch.norm(self.zero_obj_rob,dim = 1)
        has_reached = torch.zeros([self.num_envs,]).to('cuda:0')
        initial_cost = self.calcPosCost().clone()
        for t in range(self.num_horizon):
            core_1_gaussian = torch.exp(-((self.cores_gaussian[0]-mean_idx[t,0])/(self.sigma))**2)
            core_2_gaussian = torch.exp(-((self.cores_gaussian[1]-mean_idx[t,1])/(self.sigma))**2)

            core_dx = self.factors[5] * core_1_gaussian
            core_dy = self.factors[6] * core_2_gaussian

            obj_pose_rot = self.calcObjPosandRot()
            self.state_holder[:,:2] = obj_pose_rot[:,:2]
            self.state_holder[:,2] = self.calcObjYaw(obj_pose_rot[:,3:7])
            self.state_holder[:,3:5] = self.calcRobotPos(0)
            samples = stochastic_top_k(tt_cores = [self.factors[0].clone(),self.factors[1].clone(),self.factors[2].clone(),self.factors[3].clone(),self.factors[4].clone(),core_dx, core_dy],domain = self.domain_state_action, n_samples=1, alpha=self.alpha,x = self.state_holder,device = 'cuda:0')[:,0,:]
            action_i = samples[:,5:].contiguous()
            action_i[0,:] = 0
            self.sample_action[:,t,:] = action_i
            
            self.gym.set_dof_velocity_target_tensor(self.sim,gymtorch.unwrap_tensor(action_i))
            self.gym.simulate(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            has_reached += ((self.calcPosCost()<self.term_cond) * 1) * ((self.calcOrnCost()<0.2) * 1)
            # theta_robot = self.calcThetaRot()
            stage_cost = self.calcCost(scale = 0.6)
            if torch.any(torch.isnan(stage_cost)):
                print('===============================')
                print("found the nan value in the stage cost")
            self.cost += (1 - (has_reached>0)*1) * self.calcCost(scale = 0.6)#+ 0 * (1 - (has_reached>0)*1)* (theta_robot-torch.pi/36)**2 
        theta_robot = self.calcThetaRot()
        if torch.any(torch.isnan(theta_robot)):
                print('===============================')
                print("found the nan value in the theta_robot")
        terminal_pos_cost = self.calcPosCost()
        if torch.any(torch.isnan(terminal_pos_cost)):
                print('===============================')
                print("found the nan value in the terminal_pos_cost")
        terminal_orn_cost= self.calcOrnCost()
        self.cost += self.wx_T * (1 - (has_reached>0)*1) * (terminal_pos_cost  + terminal_orn_cost)
        mean_terminal_pos = torch.mean(self.cost )
        moved_prob = abs(torch.min(terminal_pos_cost) - torch.min(initial_cost))
        stuck_probablity = torch.exp(-moved_prob)
        min_cost = torch.min(self.cost)
        max_cost = torch.max(self.cost)
        if stuck_probablity >= 0.92:
            self.cost +=  self.w_exp * min_cost * (theta_robot+torch.pi/2)**2 
        min_cost = torch.min(self.cost)    
        print("--------------------")
        print("Getting stuck flag:", stuck_probablity)
        print("Min. cost among samples:", min_cost)
        print("Max. cost among samples:", max_cost)
        if torch.any(torch.isnan(min_cost)):
            print(self.cost)
            print('----------------')
            print(theta_robot)
            print('-----------------')
        normalized_cost = self.cost/(min_cost + 1)
        weights = torch.exp(-normalized_cost/self.temp)
        weighted_action = torch.einsum('i,ipq->ipq',weights,self.sample_action)
        den = torch.sum(weights)
        if den == 0:
            den += 1e-12
        mean_action = torch.sum(weighted_action,dim=0)/den
        action0 = mean_action[0,:]
        self.mean = torch.cat((mean_action[1:,:],torch.zeros([1,self.num_dofs]).to('cuda:0')),dim=0).to('cuda:0')
        return torch_to_bytes(action0)

    def setupMPPI(self,sigma_data,temp):
        sigma = bytes_to_torch(sigma_data)
        self.sigma  = sigma.repeat([self.num_horizon,1,1])
        self.temp = temp
        self.mean = torch.zeros([self.num_horizon,self.num_dofs]).to('cuda:0')
        self.mean0 = self.mean[0,:]
        self.solver = 'MPPI'

    def MPPI(self):
        distrib = MultivariateNormal(loc=self.mean, covariance_matrix=self.sigma)
        self.sample_action = distrib.sample_n(self.num_envs)
        self.sample_action = torch.clip(self.sample_action,-1,1)
        self.sample_action[0,:] = 0
        self.action_cost = torch.norm(torch.norm(self.sample_action,dim=2),dim=1)
        self.cost = self.wu_t * self.action_cost 
        rob_zero_pos = self.calcRobotPos(0)
        self.zero_obj_pos = self.calcObjPos()
        self.zero_obj_rob = rob_zero_pos - self.zero_obj_pos
        self.norm_zero_obj_rob = torch.norm(self.zero_obj_rob,dim = 1)
        has_reached = torch.zeros([self.num_envs,]).to('cuda:0')
        initial_cost = self.calcPosCost()
        for t in range(self.num_horizon):
            action_i = self.sample_action[:,t,:].squeeze().contiguous()
            self.gym.set_dof_velocity_target_tensor(self.sim,gymtorch.unwrap_tensor(action_i))
            self.gym.simulate(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            has_reached += ((self.calcPosCost()<self.term_cond) * 1) * ((self.calcOrnCost()<0.2) * 1)
            stage_cost = self.calcCost(scale = 0.6)
            if torch.any(torch.isnan(stage_cost)):
                print('===============================')
                print("found the nan value in the stage cost")
            self.cost += (1 - (has_reached>0)*1) * stage_cost

        theta_robot = self.calcThetaRot()
        terminal_pos_cost = self.calcPosCost()
        terminal_orn_cost= self.calcOrnCost()
        if torch.any(torch.isnan(theta_robot)):
                print('===============================')
                print("found the nan value in the theta_robot")
        if torch.any(torch.isnan(terminal_pos_cost)):
                print('===============================')
                print("found the nan value in the terminal_pos_cost")
        # self.cost += .1 * self.wx_T * (1 - (has_reached>0)*1) *terminal_orn_cost 
        self.cost += self.wx_T * (1 - (has_reached>0)*1) * (terminal_pos_cost + terminal_orn_cost)#+ self.w_exp* (1 - (has_reached>0)*1) * torch.exp(-((theta_robot+torch.pi/2)/.1)**2)
        moved_prob = abs(torch.min(terminal_pos_cost) - torch.min(initial_cost))
        stuck_probablity = torch.exp(-moved_prob)
        min_cost = torch.min(self.cost)
        max_cost = torch.max(self.cost)
        print('===========================')
        print("Min. cost among samples:", min_cost)
        print("Max. cost among samples:", max_cost)
        if stuck_probablity >= 0.92:
            self.cost +=  self.w_exp * min_cost * (theta_robot+torch.pi/2)**2 
        min_cost = torch.min(self.cost) + 1
        normalized_cost = self.cost/min_cost
        weights = torch.exp(-normalized_cost/self.temp)
        weighted_action = torch.einsum('i,ipq->ipq',weights,self.sample_action)
        # print(weights.shape)
        den = torch.sum(weights)
        if den == 0:
            den += 1e-12
        mean_action = torch.sum(weighted_action,dim=0)/den
        action0 = mean_action[0,:]
        self.mean = torch.cat((mean_action[1:,:],torch.zeros([1,self.num_dofs]).to('cuda:0')),dim=0).to('cuda:0')

        return torch_to_bytes(action0)
    
    def setupFLOWMPPI(self,sigma_task_data,sigma_latent,temp,portion):
        sigma_task = bytes_to_torch(sigma_task_data)
        self.sigma_task  = sigma_task.repeat([self.num_horizon,1,1])
        self.sigma_latent = sigma_latent
        self.sigma_product = (1/sigma_latent + 1)**(-1)
        self.temp = temp
        self.portion = portion
        self.mean = torch.zeros([self.num_horizon,self.num_dofs]).to('cuda:0')
        self.mean0 = self.mean[0,:]

    def FLOWMPPI(self):
        distrib = MultivariateNormal(loc=self.mean, covariance_matrix=self.sigma_task)
        sample_action_task = distrib.sample_n(int(self.num_envs* self.portion))
        sample_action_task = torch.clip(sample_action_task,-0.99,0.99)
        self.sample_action = torch.zeros([self.num_envs,self.num_horizon,2]).to('cuda:0')
        self.sample_action[:sample_action_task.shape[0],:,:] = sample_action_task 
        self.cost = torch.zeros([self.num_envs,]).to('cuda:0') 
        rob_zero_pos = self.calcRobotPos(0)
        self.zero_obj_pos = self.calcObjPos()
        self.zero_obj_rob = rob_zero_pos - self.zero_obj_pos
        self.norm_zero_obj_rob = torch.norm(self.zero_obj_rob,dim = 1)
        has_reached = torch.zeros([self.num_envs,]).to('cuda:0')
        initial_cost = self.calcPosCost()
        with torch.no_grad():
            for t in range(self.num_horizon):
                mean_action = self.mean[t,:].view([-1,2]).repeat([self.num_envs - sample_action_task.shape[0],1])
                mean_action = torch.clip(mean_action,-0.99,0.99)
                robot_pos = self.calcRobotPos(0)
                obj_pose_rot = self.calcObjPosandRot()
                yaw_angles = self.calcObjYaw(obj_pose_rot[:,3:7])
                
                State_i = torch.cat((obj_pose_rot[:,:2],yaw_angles.view([-1,1]),robot_pos[:,:2]),dim=1)
                x_data = torch.cat((State_i[-mean_action.shape[0]:,:],mean_action),dim = 1)
                z_data, _ = self.realNVP.inverse(x_data.clone())
                if torch.any(torch.isnan(z_data)):
                    print("=============")
                    print("nan is in z data")
                    print(x_data)
                mean_prod_z = self.sigma_product * (1/self.sigma_latent * z_data[:,5:])
                new_samples = torch.normal(mean_prod_z,self.sigma_product)
                new_z_data = torch.cat((z_data[:,:5],new_samples),dim=1)
                new_x_data, _  = self.realNVP.forward(new_z_data.clone())
                new_x_data[0,5:] =0
                self.sample_action[sample_action_task.shape[0]:,t,:] = new_x_data[:,5:]
                self.gym.set_dof_velocity_target_tensor(self.sim,gymtorch.unwrap_tensor(self.sample_action[:,t,:].squeeze().contiguous()))
                self.gym.simulate(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
                has_reached += ((self.calcPosCost()<self.term_cond) * 1) * ((self.calcOrnCost()<0.2) * 1)
                stage_cost = self.calcCost(scale = 0.6)
                if torch.any(torch.isnan(stage_cost)):
                    print('===============================')
                    print("found the nan value in the stage_cost")
                self.cost += (1 - (has_reached>0)*1) * self.calcCost(scale = 0.6)

            theta_robot = self.calcThetaRot()
            terminal_pos_cost = self.calcPosCost()
            terminal_orn_cost= self.calcOrnCost()
            if torch.any(torch.isnan(theta_robot)):
                print('===============================')
                print("found the nan value in the theta_robot")
            if torch.any(torch.isnan(terminal_pos_cost)):
                print('===============================')
                print("found the nan value in the terminal_pos_cost")
            self.cost += self.wx_T * (1 - (has_reached>0)*1) * (terminal_pos_cost + terminal_orn_cost)#+ self.w_exp* (1 - (has_reached>0)*1) * torch.exp(-((theta_robot-torch.pi)/.1)**2)
            moved_prob = abs(torch.min(terminal_pos_cost) - torch.min(initial_cost))
            stuck_probablity = torch.exp(-moved_prob)
            min_cost = torch.min(self.cost)
            if stuck_probablity >= 0.92:
                self.cost +=  self.w_exp * min_cost * (theta_robot+torch.pi/2)**2 
            min_cost = torch.min(self.cost) + 1
            normalized_cost = self.cost/min_cost
            weights = torch.exp(-normalized_cost/self.temp)
            weighted_action = torch.einsum('i,ipq->ipq',weights,self.sample_action)
            den = torch.sum(weights)
            if den == 0:
                den += 1e-12
            mean_action = torch.sum(weighted_action,dim=0)/den
            action0 = mean_action[0,:]
            self.mean = torch.cat((mean_action[1:,:],torch.zeros([1,self.num_dofs]).to('cuda:0')),dim=0).to('cuda:0')

            return torch_to_bytes(action0)
    
    def refineCores(self,scale,sites):
        self.domain = refine_domain(domain=self.domain, site_list=sites,scale_factor=scale, device=device)
        self.main_cores = refine_cores(tt_cores=self.factors, site_list=sites,scale_factor=scale, device=device)


def euler_to_quaternion(roll, pitch, yaw):
    roll_half = roll / 2
    pitch_half = pitch / 2
    yaw_half = yaw / 2

    # Calculate sin and cos values
    cos_roll_half = torch.cos(roll_half)
    sin_roll_half = torch.sin(roll_half)
    cos_pitch_half = torch.cos(pitch_half)
    sin_pitch_half = torch.sin(pitch_half)
    cos_yaw_half = torch.cos(yaw_half)
    sin_yaw_half = torch.sin(yaw_half)

    # Calculate quaternion components
    w = cos_roll_half * cos_pitch_half * cos_yaw_half + sin_roll_half * sin_pitch_half * sin_yaw_half
    x = sin_roll_half * cos_pitch_half * cos_yaw_half - cos_roll_half * sin_pitch_half * sin_yaw_half
    y = cos_roll_half * sin_pitch_half * cos_yaw_half + sin_roll_half * cos_pitch_half * sin_yaw_half
    z = cos_roll_half * cos_pitch_half * sin_yaw_half - sin_roll_half * sin_pitch_half * cos_yaw_half

    quat = torch.cat((x.view([-1,1]),y.view([-1,1]),z.view([-1,1]),w.view([-1,1])),dim=1)
    return quat

