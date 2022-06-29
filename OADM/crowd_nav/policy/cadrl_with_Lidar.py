import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.CoppeliaAgent.lidar_data import lidar_data


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value

class CadrlWithLidar(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL_WITH_LIDAR'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.joint_state_dim = 249
        self.state_list = list()
        self.local_score_max = None

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('cadrl', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.multiagent_training = config.getboolean('cadrl', 'multiagent_training')
        logging.info('Policy: CADRL without occupancy map')

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds =[(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]#[v_pref*(i+1)/2 for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(-110*np.pi/180, 120*np.pi/180, self.rotation_samples, endpoint=False)

        else:
            rotations = np.linspace(-110*np.pi/180, 120*np.pi/180, self.rotation_samples, endpoint=False)
        action_space = [ActionXY(0,0) if holonomic else ActionRot(0,0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space


    def predict(self,state,risk_map,robot_orientation,map,diff_map):
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        max_min_value = float('-inf')
        max_action = None
        for action in self.action_space:
            if self.kinematics=='holonomic':
                v = np.sqrt(action.vx**2+action.vy**2)
                r = np.arctan2(action.vy,action.vx)
            else:
                v = action.v
                r = action.r
            if v!=0:
                action_map = lidar_data().action_range_danger(state.self_state.v_pref, diff_map, v, risk_map)
                local_score,angle_map = self.local_evaluate(r, action_map)
                global_score = self.global_evaluate(state,action)
                combine_score = self.combine_score(local_score, global_score, angle_map)
            else:
                combine_score = -0.5
            if combine_score > max_min_value:
                max_min_value = combine_score
                max_action = action
        #print("value =",max_min_value)
        #print("action = ",max_action)


        return max_action

    def local_evaluate(self,r,action_map):
        angle_map = np.zeros((5, 2))
        angle = r * 180 / np.pi + 122
        angle_index = int(angle / 10)
        angle_map[:, [0]] = action_map[:, [angle_index - 1]]
        angle_map[:, [1]] = action_map[:, [angle_index]]
        local_score = np.sum(np.sum(angle_map[i]) for i in range(len(angle_map)))
        angle_map_max = np.ones(angle_map.shape)
        angle_map_max[[0],:] = 0
        for j in range(len(angle_map_max)):
            for k in range(len(angle_map_max[0])):
                if j ==0:
                    angle_map_max[j][k] = angle_map_max[j][k]
                else:
                    angle_map_max[j][k] = angle_map_max[j][k] *(0.9**(j-1))
                if angle_map_max[j][k]>1:
                    angle_map_max[j][k] = 1
        self.local_score_max = np.sum(np.sum(angle_map_max[i]) for i in range(len(angle_map_max)))
        local_score = local_score/self.local_score_max*10
        local_score = self.Normalization(-local_score)
        return local_score,angle_map


    def global_evaluate(self,state,action):
        state = state.self_state
        vector1 = [state.gx-state.px,state.gy-state.py]
        distance = np.sqrt(vector1[0]**2+vector1[1]**2)
        vector2 = [action.v * np.cos(action.r+state.theta),action.v*np.sin(action.r+state.theta)]
        velocity = np.sqrt(vector2[0]**2+vector2[1]**2)
        global_state = np.dot(vector1,vector2)/(distance*velocity)
        angle_score = self.Normalization(global_state*10)
        return angle_score

    def combine_score(self,local_score,global_score,action_map):
        local_weight = np.sum(np.sum(action_map[i]) for i in range(len(action_map)))
        local_weight = 0.1*np.exp(0.33*local_weight)
        global_weight = 1-local_weight

        combine_score = local_score * local_weight + global_weight*global_score
        #print("combine_score=", combine_score)
        return combine_score


    def Normalization(self,x):
        return (np.arctan(x)*2/np.pi)





        '''
    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = [next_px, next_py, action.vx, action.vy, state.radius,
                                       state.gx, state.gy, state.v_pref, state.theta]
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)

                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                next_state = [next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                       state.v_pref, next_theta]

        else:
            raise ValueError('Type error')

        return next_state


    def predict(self, state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed
stat
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have tstateo be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_min_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                next_batch_map, reward, done, info = self.env.onestep_lookahead(action)
                batch_next_states = torch.tensor([next_self_state+next_batch_map])
                # VALUE UPDATE
                output = self.model(batch_next_states.float())
                min_value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * output.data.item()
                self.action_values.append(min_value)
                if min_value > max_min_value:
                    max_min_value = min_value
                    max_action = action

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action
        
    def transform(self, state):
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
        px = state.self_state.px
        py = state.self_state.py
        vx = state.self_state.vx
        vy = state.self_state.vy
        radius = state.self_state.radius
        gx = state.self_state.gx
        gy = state.self_state.gy
        v_pref = state.self_state.v_pref
        theta = state.self_state.theta
        self_state = [px, py, vx, vy, radius, gx, gy,
                      v_pref, theta]
        state = torch.Tensor(self_state+state.human_states).to(self.device)
        return state
    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)
state
        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
        

    def build_occupancy_maps(self,state,ob):
         
         
         

         
         for i , humanCoordinate in enumerate (ob):
             px = humanCoordinate[0] - state.px
             py = humanCoordinate[1] - state.py
             velocity_angle = np.arctan2(state.vy,state.vx)
             human_Orientation = np.arctan2(py,px)
             distance = np.linalg.norm([px,py],axis=0)
             rotation = human_Orientation - velocity_angle
             trans_px = distance * np.cos(rotation)
             trans_py = distance * np.sin(rotation)
             human_x_index = np.floor(trans_px / self.cell_size + self.cell_num/2)
             human_y_index = np.floor(trans_py / self.cell_size + self.cell_num/2)
             human_x_index[human_x_index<0] = float('-inf')
             human_y_index[human_y_index<0] = float('-inf')
             human_x_index[human_x_index>=self.cell_num] = float('-inf')
             human_y_index[human_y_index>=self.cell_num] = float('-inf')
             grid_indices = self.cell_num * human_y_index + human_x_index
             occupancy_map = np.isin(range(self.cell_num**2),grid_indices)
             occupancy_maps.append([occupancy_map.astype(int)])
         return torch.from_numpy(np.concatenate(occupancy_maps,axis=0)).float()
            '''