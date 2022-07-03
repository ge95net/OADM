import numpy as np
from .CoppeliaAgent import CoppeliaAgent
class lidar_data():
    def __init__(self):
        self.agent = None
        self.last_risk_map = None
        self.risk_map = None
        self.obstacle = list()
    def handel_data(self,agent,last_map):
        '''get the measured data from the lidar of vrep'''
        self.agent = agent
        measured_data = CoppeliaAgent.get_lidar_reading(self.agent)
        map = self.map(measured_data)
        self.risk_map = self.danger_map(map)
        diff_map = self.risk_map - last_map
        rotations = np.linspace(10, 240 , 23, endpoint=False)

        return self.risk_map , diff_map,map



    '''
    def rotate_point(self,px,py,pz):
        rotate the coordinate from world coordinate frame to robot coordinate frame
        print("px=", px, "py=", py)
        alpha = CoppeliaAgent.get_Orientation(self.agent)
        theta =  alpha - np.pi/2
        rotated_px = np.cos(theta)*px - np.sin(theta)*px
        rotated_py = np.sin(theta)*py + np.cos(theta)*py
        robot_px , robot_py = CoppeliaAgent.get_real_position(self.agent)
        print("robot_px=", robot_px, "robot_py=", robot_py)
        prismatic_x = robot_px
        prismatic_y = robot_py
        new_px = prismatic_x + rotated_px
        new_py = prismatic_y + rotated_py
        new_pz = pz
        print("new_px=", new_px, "new_py=", new_py)

        return  new_px,new_py,new_pz
    '''
    def map(self,measured_data):
        a = [0 for i in range(24)]
        b = [a for i in range(5)]
        c = np.array(b)
        map = np.array(b)
        for l in range(0, len(measured_data), 4):
            px = measured_data[l]
            py = measured_data[l + 1]
            pz = measured_data[l + 2]
            distance = measured_data[l + 3]
            if distance < 4.4:
                distance_index = np.floor(distance + 0.6)
                angle_index = np.floor(self.angle_calculate(px, py, pz))
                c[int(distance_index)][int(angle_index)] += 1


        for j in range(5):
            for k in range(24):
                if (c[j][k] > 7 - j):
                    map[j][k] += 1
        return map


    def angle_calculate(self,px,py,pz):
        '''calculate the angle between the point and the zero angle of the lidar_scan_Range'''
        angle_between_axis = np.arctan2(py,px)
        angle_between_axis = angle_between_axis * 360/(2*np.pi)
        angle = 120 + angle_between_axis
        angle_index = np.floor(angle / 10)
        return  angle_index

    def danger_map(self , map):
        risk_map = np.zeros(map.shape)
        obstacle_pos = list()

        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] == 1:
                    obstacle_pos.append([i,j])

        self.obstacle = obstacle_pos
        for l in range(len(obstacle_pos)):
            x, y = obstacle_pos[l][0], obstacle_pos[l][1]
            for m in range(len(map)):
                for n in range(len(map[0])):
                    trans_mx = (m) * np.cos((10*n-30)*np.pi/180)
                    trans_ny = (m) * np.sin((10*n-30)*np.pi/180)
                    trans_x = (x) * np.cos((10*y-30)*np.pi/180)
                    trans_y = (x) * np.sin((10*y-30)*np.pi/180)
                    distance = np.sqrt((trans_x - trans_mx)**2 + (trans_y - trans_ny)**2)
                    if distance<1.9 and m!=0: #and risk_map[m][n]<1:
                        '''
                        if m>=x:
                            danger_value = self.normal_distribution(distance) * (0.9**(m-x))
                        else:
                            danger_value = self.normal_distribution(distance)
                        '''
                        danger_value =  self.normal_distribution(distance)
                        risk_map[m][n] += danger_value

                        if risk_map[m][n]>=1:
                            risk_map[m][n] = 1

        return  risk_map


    def normal_distribution(self,x):
        miu = 0
        sigma = 3.5
        gaussion = np.exp(-1/2 * (x-miu)**2 / (sigma**2))
        return gaussion


    def action_range_danger(self,v_pred,diff_map,v,risk_map):
        alpha = 0.9* v/v_pred
        action_map =diff_map + risk_map
        for j in range(len(action_map)):
            for k in range(len(action_map[0])):
                if j ==0:
                    action_map[j][k] = action_map[j][k]
                else:
                    action_map[j][k] = action_map[j][k] *(alpha**(j-1))
                if action_map[j][k]>1:
                    action_map[j][k] = 1
        return action_map






