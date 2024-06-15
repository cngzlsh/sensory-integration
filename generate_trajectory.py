# this file generates a random trajectory in a square arena using RiaB

import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
import numpy as np
import pandas as pd
import pathlib
import json
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--time', type=int, default=300)
parser.add_argument('--save_folder', type=str, default='./data/trajectories/')
parser.add_argument('--dt', type=float, default=0.02)
parser.add_argument('--motion', type=str, default='circular')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--arena_shape', type=str, default='rect')
parser.add_argument('--thigmotaxis', type=float, default=0.6)
parser.add_argument('--switch_motion_pattern_halfway', type=bool, default=True)
args = parser.parse_args()

np.random.seed(args.seed)

# parameters
time = args.time # secs
save_path = pathlib.Path(args.save_folder)

Env = Environment()
if args.arena_shape[:4] == 'rect':
    Env = Environment(
        params={'boundary':
                [[0,0],
                [1,0],
                [1,2],
                [0,2]]}
    )
Ag = Agent(Env)

Ag.params['rotational_velocity_std'] = 1.0
Ag.params['speed_mean'] = 0.06

if args.arena_shape == 'square':
    Ag.pos = np.array([0.5,0.5])
elif args.arena_shape[:4] == 'rect':
    Ag.pos = np.array([0.5,1])


Ag.thigmotaxis = args.thigmotaxis
Ag.head_direction = np.array([0,0])
Ag.dt = args.dt
Ag.rotational_velocity_std = 1.0
Ag.speed_mean = 0.06

# main trajectory loop
for i in tqdm(range(int(time /args.dt))):
    if args.motion == 'circular':
        if args.arena_shape == 'square':
            [x,y] = Ag.pos - np.array([0.5,0.5]) #distance from the centre of the Environment
            [vx,vy] = [-0.3*y,0.3*x]
        elif args.arena_shape[:4] == 'rect':
            [x,y] = Ag.pos - np.array([0.5, 1]) #distance from the centre of the Environment
            [vx,vy] = [-0.3*y,0.6*x]
        Ag.update(dt=args.dt, drift_velocity=np.array([vx,vy]), drift_to_random_strength_ratio=0.1)
    elif args.motion == 'brownian':
        Ag.update(dt = args.dt)
    
    if args.switch_motion_pattern_halfway:
        if i == int(time /args.dt / 2):
            if args.motion == 'circular':
                args.motion = 'brownian'
            elif args.motion == 'brownian':
                args.motion = 'circular'

fig, ax = Ag.plot_trajectory()
# ratinabox.utils.save_figure(fig, save_path / f'{time}s_dt_{args.dt}_{args.motion}_riab_trajectory.png')
if args.switch_motion_pattern_halfway:
    fig.savefig(save_path / f'{time}s_dt_{args.dt}_switch_{args.arena_shape}_riab_trajectory.png', bbox_inches='tight', dpi=200)
else:
    fig.savefig(save_path / f'{time}s_dt_{args.dt}_{args.motion}_{args.arena_shape}_riab_trajectory.png', bbox_inches='tight', dpi=200)

Ag.history['pos'] = [{'loc': x} for x in Ag.history['pos']]
Ag.history['head_direction'] = [{'head_direction': x} for x in Ag.history['head_direction']]
Ag.history['vel'] = [{'vel': x} for x in Ag.history['vel']]

if args.switch_motion_pattern_halfway:
    with open(save_path / f'{time}s_dt_{args.dt}_switch_{args.arena_shape}_riab_trajectory.json', 'w') as f:
        json.dump(Ag.history, f)
else:
    with open(save_path / f'{time}s_dt_{args.dt}_{args.motion}_{args.arena_shape}_riab_trajectory.json', 'w') as f:
        json.dump(Ag.history, f)
