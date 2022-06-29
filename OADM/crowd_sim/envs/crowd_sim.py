import logging
import gym
import numpy as np
import rvo2
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
import time
from crowd_sim.envs.remoteAPI import sim
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None

        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.client = None
        self.start_time = None
        self.end_time = None
        self.last_map = np.zeros((5,24))
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self.test_num = 0
        self.robot_action = None
        self.sign = 0




    def configure(self, config,clientID):
        self.client = clientID
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))


    def set_robot(self, robot):
        self.robot = robot
        ad_rdy_ev = threading.Event()
        ad_rdy_ev.set()

        t = threading.Thread(target=self.plot, args=())
        t.daemon = True
        t.start()




    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):

        human = Human(self.config, 'humans', self.client)
        if self.randomize_attributes:
            human.sample_random_attributes()
        
        while True :
            collide = False
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            angle2 = np.random.random()*np.pi*2
            gx = self.circle_radius*np.cos(angle2) + px_noise
            gy = self.circle_radius*np.sin(angle2) + py_noise
            while px*gx >= 0 or py*gy >=0:
                angle2 = np.random.random() * np.pi * 2
                gx = self.circle_radius * np.cos(angle2) + px_noise
                gy = self.circle_radius * np.sin(angle2) + py_noise
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if py>=0 and not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        
        human = Human(self.config, 'humans', self.client)
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        

        return human


    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans',self.client)
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def generate_fixed_human(self,humans_point):
        self.humans = []
        for i in range(self.human_num):
            human = Human(self.config, 'humans', self.client)
            human_point = humans_point[i]
            px = human_point[0]
            py = human_point[1]
            gx = human_point[2]
            gy = human_point[3]
            human.set(px, py, gx, gy, 0, 0, 0)
            self.humans.append(human)

    def reset(self, phase, test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        sim.simxStopSimulation(self.client,sim.simx_opmode_oneshot)
        time.sleep(3)
        sim.simxStartSimulation(self.client,sim.simx_opmode_oneshot)
        time.sleep(3)
        sim.simxAddStatusbarMessage(self.client,'Hello',sim.simx_opmode_oneshot)
        self.last_map = np.zeros((5,24))
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        self.start_time = time.time()
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius+1, 0, 0, np.pi / 2)
            if self.human_num==5:
                humans_point_data = np.load(
                    "/home/yy/research/cadrl_with_lidar/CrowdNav-master/crowd_sim/envs/humans_position/five_human_r_4.npy")
                humans_point = humans_point_data[self.test_num]
                self.generate_fixed_human(humans_point)
                self.test_num = self.test_num + 1
            if self.human_num ==7:

                humans_point_data = np.load(
                    "/home/yy/research/cadrl_with_lidar/CrowdNav-master/crowd_sim/envs/humans_position/seven_human_r_6.npy")
                humans_point = humans_point_data[self.test_num]
                self.generate_fixed_human(humans_point)
                self.test_num = self.test_num + 1
            if self.human_num ==3:
                humans_point_data = np.load(
                    "/home/yy/research/cadrl_with_lidar/CrowdNav-master/crowd_sim/envs/humans_position/three_human_r_2.npy")
                humans_point = humans_point_data[self.test_num]
                self.generate_fixed_human(humans_point)
                self.test_num = self.test_num + 1

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        next_batch_map = []
        self.robot.agent.get_real_position()
        # get current observation
        if self.robot.sensor == 'coordinates':
            ob,diff,map = self.robot.get_lidar_reading(self.last_map)
            for i in range(5):
                for j in range(24):
                    next_batch_map.append(ob[i][j])
            ob = next_batch_map
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError
        sim.simxPauseSimulation(self.client, sim.simx_opmode_oneshot)
        time.sleep(3)
        sim.simxStartSimulation(self.client, sim.simx_opmode_oneshot)


        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        step_starttime = time.time()
        self.robot_action = action
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            #obstacle = ObservableState(0,0,0,0,0.5)
            #ob += [obstacle]
            #ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        #collision = False

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = self.robot.agent.get_real_position()
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < 0.5
        self.end_time = time.time()
        self.global_time = self.end_time - self.start_time

        collision = 0
        ob, diff, map = self.robot.get_lidar_reading(self.last_map)
        if sum(map[0])>0:
            collision = 1
        danger = False
        if map[0].all()> 0 :
            danger = True

        #print("collision=",collision)
        if self.global_time >= self.time_limit - 1:

            done = True
            info = Timeout()
            if done:
                print("timeout")
        elif collision:

            done = True
            info = Collision()
            if done:
                print("collision")
        elif reaching_goal:

            done = True
            info = ReachGoal()
            if done:
                print("reachgoal")
        elif danger:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS

            done = False
            info = Danger(dmin)
        else:

            done = False
            info = Nothing()



        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
                #if i == 0:
                 #   print("action=",human_action)
            step_endtime = time.time()
            step_time = step_endtime - step_starttime
            #time.sleep(0.3-step_time)
            #self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            if done  :
                self.robot.agent.stop()
                for human in self.humans:
                    human.agent.stop()


            next_batch_map = []
            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob,diff,map = self.robot.get_lidar_reading(self.last_map)
                for i in range(5):
                    for j in range(24):
                        next_batch_map.append(ob[i][j])
                ob = next_batch_map
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        else:
            if self.robot.sensor == 'coordinates':
                ob = self.robot.get_next_observable_state_map(self.last_map)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError



        return ob, done, info



    def plot(self):
        danger_map, diff_map,map = self.robot.get_lidar_reading(self.last_map)
        if self.sign == 1:
            action_map = diff_map
        else:
            action_map = np.zeros((5, 24))
        self.last_map = danger_map
        theta = np.linspace(-120, 120, 24) / 180 * np.pi
        r = np.linspace(0, 5, 5)
        theta, r = np.meshgrid(theta, r)
        figure = plt.figure(figsize=(10, 10))
        #ax = plt.subplot(2, 2, 1, projection='polar')
        #ax2 = figure.add_subplot(2, 2, 2, projection='polar')
        #ax3 = figure.add_subplot(2, 2, 3, projection='polar')
        ax4 = figure.add_subplot(projection='polar')
        self.fig = figure
        #self.ax = ax
        #self.ax2 = ax2
        #self.ax3 = ax3

        self.ax4 = ax4

        # figure the first picture
        '''
        ax.set_thetagrids(np.arange(-120.0, 120.0, 10.0), fontsize=7)
        ax.set_thetamin(-120.0)
        ax.set_thetamax(120.0)
        ax.set_title('danger_map')
        ax.set_rgrids(radii=np.linspace(0, 5, 6))
        ax.set_facecolor('w')
        im = ax.contourf(theta, r, danger_map, cmap='Reds', levels=np.linspace(0, 1, 11))
        figure.colorbar(mappable=im, ax=ax)
        
        # figure the second picture
        ax2.set_thetagrids(np.arange(-120.0, 120.0, 10.0), fontsize=7)
        ax2.set_thetamin(-120.0)
        ax2.set_thetamax(120.0)
        ax2.set_title('danger_diff_map')
        ax2.set_rgrids(radii=np.linspace(0, 5, 6))
        ax.set_facecolor('w')
        im2 = ax2.contourf(theta, r, diff_map, cmap='coolwarm', levels=np.linspace(-1, 1, 11))
        figure.colorbar(mappable=im2, ax=ax2)
        # figure the third picture

        ax3.set_thetagrids(np.arange(-120.0, 120.0, 10.0), fontsize=7)
        ax3.set_thetamin(-120.0)
        ax3.set_thetamax(120.0)
        ax3.set_title('last_map')
        ax3.set_rgrids(radii=np.linspace(0, 5, 6))
        im3 = ax3.contourf(theta, r, self.last_map, cmap='Reds', levels=np.linspace(0, 1, 11))
        figure.colorbar(mappable=im3, ax=ax3)
        '''
        # figure the fourth picture
        ax4.set_thetagrids(np.arange(-120.0, 120.0, 10.0), fontsize=7)
        ax4.set_thetamin(-120.0)
        ax4.set_thetamax(120.0)
        ax4.set_title('action_map')
        ax4.set_rgrids(radii=np.linspace(0, 5, 6))
        #ax4.text(5,-5,'action: {}'.format(self.robot_action))
        im4 = ax4.contourf(theta, r, action_map, cmap='coolwarm', levels=np.linspace(-1, 1, 11))
        figure.colorbar(mappable=im4, ax=ax4)
        #plt.text(10,0,'action: {}'.format(self.robot_action))
        ani = FuncAnimation(figure, self.update, interval=1)
        plt.show()

    def update(self, i):
        risk_map, diff_map,map = self.robot.get_lidar_reading(self.last_map)
        if self.robot_action != None:
            action_map = self.robot.action_risk_map(self.last_map,self.robot_action.v)
            robot_orientation = self.robot.get_orientation()
            #print(robot_orientation*180/np.pi)
            #plt.text(0, 0, 'v: {:.2f},r:{:.2f}'.format(self.robot_action.v, (self.robot_action.r)*180/np.pi))
        else:
            action_map = risk_map
        theta = np.linspace(-120, 120, 24) / 180 * np.pi
        r = np.linspace(0, 5, 5)
        theta, r = np.meshgrid(theta, r)
        #im = self.ax.contourf(theta, r, risk_map, cmap='Reds', levels=np.linspace(0, 1, 11))
        #im2 = self.ax2.contourf(theta, r, diff_map, cmap='coolwarm', levels=np.linspace(-1, 1, 11))
        #im3 = self.ax3.contourf(theta, r, self.last_map, cmap='Reds', levels=np.linspace(0, 1, 11))
        im4 = self.ax4.contourf(theta, r, action_map, cmap='coolwarm', levels=np.linspace(-1, 1, 11))

        self.last_map = risk_map


        return  im4










