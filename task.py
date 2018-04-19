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
      

    def get_reward(self):
        """ Simple reward based on the distance to target. Counts XYZ coordinates """
        reward = 0.
        dist = np.linalg.norm(np.array(self.sim.pose[:3])-np.array(self.target_pos))
        reward -= dist
        
        return reward

    def step(self, rotor_speeds):
        """Policy for take off action"""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)

            if self.sim.pose[2] >= self.target_pos[2]:
                reward = 200
                done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

    def step_takeoff_landing(self, rotor_speeds):
        """Not finished policy for more complex action sequence go up than go down"""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            
            if self.takeoff:
                reward += self.get_reward_option_3() 
                pose_all.append(self.sim.pose)
                
                if self.sim.pose[2] >= self.landing_pos[2]:
                    reward = 100
                    done = True
            else:
                reward += self.get_reward_option_2() 
                pose_all.append(self.sim.pose)
                if self.sim.pose[2] >= self.target_pos[2]:
                    reward += 5
                    self.takeoff = True
                
        next_state = np.concatenate(pose_all)
        return next_state, reward, done


if __name__ == '__main__':
    #test cases
    init_pose = np.asarray([[0., 0., 0., 0., 0., 0.], [3., 3., 0., 0., 0., 0.]])

    rotor_speeds_up = np.array([900.,900.,900.,900.])
    rotor_speeds_off = np.array([34.,400.,300.,300.])




    def testrun(task, rotor_speed):
        next_state, reward, done = task.step(rotor_speed)
        print("Position {}; Velocity {}; Reward {}".format(task.sim.pose, task.sim.v, reward))

        return done, reward

    # Test 1 - happy end
    task = Task(init_pose=init_pose[0])
    task.action_repeat = 1
    task.reset()
    counter = 0
    while True:

        done, reward = testrun(task, rotor_speeds_up)
        counter += 1
        if done and reward >= 100:
            print("Successfully finished test 1")
            break
        elif counter >= 500:
            print("Failure for test 1. Too many attempts")
            break
        elif done:
            print("Failure for test 1. Drone crashed")

    # Test 2 - failure
    task = Task(init_pose=init_pose[1])
    task.action_repeat = 1
    task.reset()
    counter = 0
    while True:

        done, reward = testrun(task, rotor_speeds_off)
        counter += 1
        if done and reward >= 100:
            print("Successfully finished test 2")
            break
        elif counter >= 100:
            print("Failure for test 2. Too many attempts")
            break
        elif done:
            print("Failure for test 2. Drone crashed")

