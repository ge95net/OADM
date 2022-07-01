import crowd_sim.envs.utils.agent
from crowd_sim.envs.remoteAPI import sim
from crowd_sim.envs.remoteAPI import simConst
import numpy as np
human_name = "Pioneer_p3dx"


class CoppeliaAgent():
    def __init__(self, client, name = None, id = None):
        self.client = client
        if name:
            self.name = name
            self.id = id
        else:
            self.id = id
            self.name = human_name + "#"+str(id)
        ars , self.agent = sim.simxGetObjectHandle(self.client,self.name,sim.simx_opmode_blocking)


    def setPosition(self,px,py):
        inputInts = []
        inputFloats = [px,py,0.17]
        inputStrings = []
        inputBuffer = bytearray()
        sim.simxCallScriptFunction(self.client,self.name,sim.sim_scripttype_childscript,'setPosition',
                                   inputInts,inputFloats,inputStrings,inputBuffer,sim.simx_opmode_oneshot)

    def setOrientation(self,gx, gy):
        inputInts = []
        inputFloats = [0,0,np.arctan2(gy,gx)]
        inputStrings = []
        inputBuffer = bytearray()
        sim.simxCallScriptFunction(self.client, self.name, sim.sim_scripttype_childscript, 'setOrientation',
                                       inputInts, inputFloats, inputStrings, inputBuffer, sim.simx_opmode_oneshot)
    def get_lidar_reading(self):
        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        res,retInts,retFloats,retStrings,retBuffer = sim.simxCallScriptFunction(self.client, self.name, sim.sim_scripttype_childscript, 'get_lidar_reading',
                                       inputInts, inputFloats, inputStrings, inputBuffer, sim.simx_opmode_oneshot)
        #print("dist=",retFloats)
        return  retFloats

    def get_first_lidar_reading(self):
        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.client, self.name,
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'get_lidar_reading',
                                                                                    inputInts, inputFloats,
                                                                                    inputStrings, inputBuffer,
                                                                                    sim.simx_opmode_remove)
        # print("dist=",retFloats)
        return retFloats

    def get_first_position(self):
        successful, pos = sim.simxGetObjectPosition(self.client, self.agent, -1, sim.simx_opmode_remove)
        return pos[0], pos[1]

    def set_drive_ins(self,vx,vy,gx,gy):
        inputInts = []
        inputFloats = [vx,vy,gx,gy]
        inputStrings = []
        inputBuffer = bytearray()
        sim.simxCallScriptFunction(self.client,self.name, sim.sim_scripttype_childscript,
                                   'set_drive_ins',
                                   inputInts, inputFloats, inputStrings, inputBuffer, sim.simx_opmode_oneshot)

    def get_real_position(self):
        successful, pos = sim.simxGetObjectPosition(self.client,self.agent,-1,sim.simx_opmode_oneshot)
        return pos[0],pos[1]


    def stop(self):
        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        sim.simxCallScriptFunction(self.client,self.name, sim.sim_scripttype_childscript,
                                   'finish',
                                   inputInts, inputFloats, inputStrings, inputBuffer, sim.simx_opmode_blocking)

    def get_Orientation(self):
        res , orientation = sim.simxGetObjectOrientation(self.client,self.agent,-1,sim.simx_opmode_oneshot)

        return orientation[2]

    def collision_detection(self):
        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        res,retInts,retFloats,retStrings,retBuffer = sim.simxCallScriptFunction(self.client, self.name, sim.sim_scripttype_childscript,
                                   'sysCall_sensing',
                                   inputInts, inputFloats, inputStrings, inputBuffer, sim.simx_opmode_oneshot)
        #print(retInts)
        return retInts





