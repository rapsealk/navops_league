#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import time
from datetime import datetime
from collections import namedtuple

import numpy as np

#import mlagents
#import mlagents_envs
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

BEHAVIOR_NAME = "Warship?team=0"


Observation = namedtuple('Observation',
                         ('decision_steps', 'terminal_steps'))


def main():
    """
    file_name: is the name of the environment binary (located in the root directory of the python project)
    worker_id: indicates which port to use for communication with the environment. For use in parallel training regimes such as A3C.
    seed: indicates the seed to use when generating random numbers during the training process.
          In environments which are deterministic, setting the seed enables reproducible experimentation by ensuring that the environment and trainers utilize the same random seed.
    side_channels: provides a way to exchange data with the Unity simulation that is not related to the reinforcement learning loop.
                   For example: configurations or properties.
                   More on them in the "Modifying the environment from Python"(https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md#modifying-the-environment-from-python) section.
    ---
    env.reset()
    env.step()
    env.close()
    """
    channel = EngineConfigurationChannel()

    filename = "BlackWhale" # "3DBall"
    env = UnityEnvironment(file_name=filename, seed=1, side_channels=[channel])

    channel.set_configuration_parameters(time_scale=10.0)

    env.reset()
    behavior_names = env.behavior_specs.keys()
    # print('behavior_names:', behavior_names)
    # for name in behavior_names:
    #     print('behavior_name:', name)   # Mummy?team=0
    behavior_names = [name for name in behavior_names]
    print('behavior_names:', behavior_names)

    #while True:
    #    env.step()
    #    time.sleep(1)

    # decision_steps, terminal_steps = env.get_steps(behavior_name="Warship?team=0")
    obs1 = Observation(*env.get_steps(behavior_name=behavior_names[0]))
    obs2 = Observation(*env.get_steps(behavior_name=behavior_names[1]))
    """
    print('DecisionSteps')
    print('- observation:', decision_steps.obs)
    print('- reward:', decision_steps.reward)
    print('- agent_id:', decision_steps.agent_id)
    print('- action_mask:', decision_steps.action_mask)

    print('TerminalSteps')
    print('- observation:', terminal_steps.obs)
    print('- reward:', terminal_steps.reward)
    print('- agent_id:', terminal_steps.agent_id)
    print('- interrupted:', terminal_steps.interrupted)
    """

    while True:
        # print('decision_stpes.agent_id:', decision_steps.agent_id)
        # print('decision_steps.observation:', decision_steps.obs)
        # print('decision_steps.reward:', decision_steps.reward)
        # print('decision_steps.action_mask:', decision_steps.action_mask)
        # print('terminal_steps.agent_id:', terminal_steps.agent_id)
        for team_id, (decision_steps, terminal_steps) in enumerate([obs1, obs2]):
            for i in decision_steps.agent_id:
                if i in terminal_steps.agent_id:
                    continue
                action = np.zeros(shape=(1, 6))
                action[0, np.random.randint(0, 6)] = 1
                env.set_action_for_agent(behavior_name="Warship?team={}".format(team_id), agent_id=i, action=action)
        print('[%s] step' % datetime.now().isoformat())
        env.step()

        # decision_steps, terminal_steps = env.get_steps(behavior_name="Warship?team=0")
        obs1 = Observation(*env.get_steps(behavior_name=behavior_names[0]))
        obs2 = Observation(*env.get_steps(behavior_name=behavior_names[1]))
        #print('DecisionSteps')
        #print('- reward:', decision_steps.reward)
        # print('- decision.agent_id:', decision_steps.agent_id)
        #print('- action_mask:', decision_steps.action_mask)
        #print('TerminalSteps')
        #print('- reward:', terminal_steps.reward)
        # print('- terminal.agent_id:', terminal_steps.agent_id)
        #print('- interrupted:', terminal_steps.interrupted)

        #if len(terminal_steps.interrupted) > 0:
        #    env.reset()
        # time.sleep(1)

    env.close()


if __name__ == "__main__":
    main()
