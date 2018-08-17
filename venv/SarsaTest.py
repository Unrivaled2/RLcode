from maze_env import Maze
from SarsaTable import SarsaTable

def update():
    for i in range(100):
        observation = env.reset()
        RL.eligiblity_trace = 0
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)
            action_ = action
            observation = observation_
            if done:
                break;
    print("game over")
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100,update)
    env.mainloop()
