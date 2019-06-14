import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.total_distance = np.linalg.norm(self.target_pos - self.sim.init_pose[:3])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # check if the direction is right
        curr_pose, targ_pose = self.sim.pose[:3], self.target_pos
        if np.linalg.norm(curr_pose) * np.linalg.norm(targ_pose)>0:
            direction_is_right = np.dot(curr_pose, targ_pose) / (np.linalg.norm(curr_pose) * np.linalg.norm(targ_pose))
        else:
            direction_is_right = 0
        
        reward = 100 * self.destination_reached + direction_is_right - 0.1 * self.distance_left - 100 * self.crashed
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            # check for distance left, task success, and crash
            self.distance_left_abs = np.linalg.norm(self.target_pos - self.sim.pose[:3])
            self.distance_left = self.distance_left_abs / self.total_distance
            self.destination_reached = self.distance_left < 0.01
            self.crashed = (self.sim.pose[2]<0)
            
            if self.destination_reached or self.crashed:
                done = True
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state