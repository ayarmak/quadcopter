import numpy as np
import pandas as pd
import copy
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class FlightRecorder():
    
    def __init__(self, sim, log_rotors=True):
        self.sim = sim
        self.log_rotors = log_rotors
        self.labels = ['done', 
          'x', 'y', 'z', 'phi', 'theta', 'psi', 
          'x_velocity', 'y_velocity', 'z_velocity', 
          'phi_velocity', 'theta_velocity', 'psi_velocity'] \
            + (['rotor {}'.format(i) for i in range(1,5)] if self.log_rotors else [])
        self.df = pd.DataFrame({x: [] for x in self.labels})
        self.df.done = self.df.done.astype(bool)
    
    def update(self, rotor_speeds=None):
        self.df.loc[self.sim.time] = [self.sim.done, (*self.sim.pose), (*self.sim.v), (*self.sim.angular_v)] \
            +([(*rotor_speeds)] if self.log_rotors else [])
        
    def plot(self, title=None):
        
        num_plots = 4 if self.log_rotors else 3
        fig, ax = plt.subplots(1, num_plots, figsize = (num_plots*4, 4))
        suptitle = title if title else 'Cockpit Measurements'
        plt.suptitle(suptitle, fontweight='bold', fontsize=14, y=1.01)
        titles = ['XYZ-coordinates', 'Euler angles', 'Velocities'] + (['Rotor Speeds'] if self.log_rotors else [])
        limits = [(0, 10), (-0.5, 0.5), (-0.5, 0.5)] + ([(0.0, 900.0)] if self.log_rotors else [])
        data = [['x', 'y', 'z'], ['phi', 'theta', 'psi'], 
              ['x_velocity', 'y_velocity', 'z_velocity', 
              'phi_velocity', 'theta_velocity', 'psi_velocity']] \
                + ([['rotor {}'.format(i) for i in range(1,5)]] if self.log_rotors else [])

        done_times = self.df.index[self.df.done]

        for i, (title, fields, lim) in enumerate(zip(titles, data, limits)):

            # plot the data point and label x-axis
            self.df[fields].plot(ax=ax[i], title=title)
            ax[i].set_xlabel('Time')

            # update y-axis limits to avoid visually misleading with small changes
            y_padding = (lim[1] - lim[0]) * 0.06
            y_lower, y_upper = ax[i].get_ylim()
            new_y_lower, new_y_upper = min(y_lower, lim[0] - y_padding), max(y_upper, lim[1] + y_padding)
            ax[i].set_ylim(new_y_lower, new_y_upper)

            # shade out timesteps after the end of the episode
            if not done_times.empty:
                x = done_times.min()
                width = ax[i].get_xlim()[1] - x
                y = new_y_lower
                height = new_y_upper-new_y_lower
                rect = patches.Rectangle((x, y), width, height, facecolor='grey', alpha=.2)
                ax[i].add_patch(rect)
                handles, labels = ax[i].get_legend_handles_labels()
                ax[i].legend(handles+[rect], labels+['Episode ended'])
         
        return fig, ax
                
    def reset(self):
        self.df.drop(self.df.index, inplace=True)
