import tyro
from pathlib import Path
from tqdm import trange
from collections import defaultdict
import gc  # noqa: E402
import cv2  # noqa: E402
from tqdm import tqdm  # noqa: E402
import gymnasium
import numpy as np
import jax
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from dataclasses import dataclass
import warnings


def spec(tree):
    def maybe_shape(x):
        # shape if has else type
        return x.shape if hasattr(x, 'shape') else type(x)

    return jax.tree.map(lambda x: maybe_shape(x), tree)


def hwc2chw(imgs):
    if isinstance(imgs, dict):
        return jax.tree.map(lambda x: x.transpose(2, 0, 1), imgs)
    else:
        return imgs.transpose(2, 0, 1)


def safe_render(env):
    imgs = env.render()
    if imgs is not None:
        # im = hwc2chw(imgs)
        im = np.array(imgs).astype(np.uint8)
        # print(spec(im))
        cv2.imshow('obs', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1000 // 20) == ord('q'):
            # if cv2.waitKey(1) == ord('q'):
            return True
    return False


def save_dataset(cfg, dataset, total_train_steps, total_steps):
    print('Total steps:', total_steps)

    suffix = f'{cfg.env_name}-{cfg.dataset_type}.npz'
    train_path = cfg.save_path / suffix
    val_path = cfg.save_path / suffix.replace('.npz', '-val.npz')
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        elif k == 'button_states':
            dtype = np.int64
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


@dataclass
class Config:  # FLAGS
    seed: int = 0
    env_name: str = 'cube-double-v0'
    max_episode_steps: int = 1001
    num_episodes: int = int(1e3)

    dataset_type: str = 'play'
    save_path: str = Path.cwd()
    noise: float = 0.1
    noise_smoothing: float = 0.5
    min_norm: float = 0.4
    p_random_action: float = 0.0

    action_type: str = 'relative'  # 'relative' or 'absolute'
    headless: bool = False


def main(cfg: Config):
    warnings.filterwarnings('ignore')

    # Initialize environment.
    env = gymnasium.make(
        cfg.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=cfg.max_episode_steps,
        action_type=cfg.action_type,
    )

    oracle_type = 'plan' if cfg.dataset_type == 'play' else 'markov'
    agents = {
        'markov': CubeMarkovOracle(env=env, min_norm=cfg.min_norm, action_type=cfg.action_type),
        # 'plan': CubePlanOracle(env=env, noise=cfg.noise, noise_smoothing=cfg.noise_smoothing, action_type=cfg.action_type),
    }
    agent = agents['markov']

    # Set the cube stacking probability for this episode.
    if 'single' in cfg.env_name:
        p_stack = 0.0
    elif 'double' in cfg.env_name:
        p_stack = np.random.uniform(0.0, 0.25)
    elif 'triple' in cfg.env_name:
        p_stack = np.random.uniform(0.05, 0.35)
    elif 'quadruple' in cfg.env_name:
        p_stack = np.random.uniform(0.1, 0.5)
    elif 'octuple' in cfg.env_name:
        p_stack = np.random.uniform(0.0, 0.35)
    else:
        p_stack = 0.5

    ob, info = env.reset()
    # pprint(spec(info))
    agent.reset(ob, info)

    """
    for i in tqdm(range(1000)):
        act = env.action_space.sample()
        obs, reward, term,trunc, info = env.step(act)
    """

    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = cfg.num_episodes
    num_val_episodes = cfg.num_episodes // 10

    print(('total', cfg.num_episodes + cfg.num_episodes // 10))
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        obs, info = env.reset()

        if oracle_type == 'markov':
            # Set the action noise level for this episode.
            xi = np.random.uniform(0, cfg.noise)
        agent.reset(ob, info)

        done = False
        step = 0
        ep_qpos = []  # for health check

        # for _i in range(cfg.max_episode_steps):
        # for _i in tqdm(range(cfg.max_episode_steps),leave=False):
        while not done:
            if np.random.rand() < cfg.p_random_action:
                # Sample a random action.
                action = env.action_space.sample()
            else:
                # Get an action from the oracle.
                action = agent.select_action(ob, info)
                action = np.array(action)

                if oracle_type == 'markov':
                    # Add Gaussian noise to the action.
                    action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)

            # action = np.clip(action, -1, 1)
            # pprint(np.array(action).round(2))
            next_ob, reward, terminated, truncated, info = env.step(action)
            # terminated = env.unwrapped._success
            done = terminated or truncated
            # pprint((terminated, truncated, done))

            if agent.done:
                # Set a new task when the current task is done.
                agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                agent.reset(agent_ob, agent_info)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])
            ep_qpos.append(info['prev_qpos'])

            ob = next_ob
            step += 1

            if not cfg.headless and safe_render(env):
                env.close()
                quit()
            if done:
                break

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

            # print(spec(imgs))

    save_dataset(cfg, dataset, total_train_steps, total_steps)

    env.close()
    del env
    del agent
    gc.collect()
    # print('done')


if __name__ == '__main__':
    main(tyro.cli(Config))
    quit()

    from multiprocessing import Pool
    from functools import partial

    _main = partial(main, tyro.cli(Config))

    def job(*args, **kwargs):
        _main()

    n = int(1e3)

    with Pool(processes=4) as pool:
        for result in pool.imap_unordered(
            job,
            tqdm(range(n), total=n, desc='Episodes', leave=False),
            chunksize=1,
        ):
            pass
