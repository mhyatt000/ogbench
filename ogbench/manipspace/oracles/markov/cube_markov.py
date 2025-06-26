import numpy as np

from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle
from ogbench.manipspace import lie


class CubeMarkovOracle(MarkovOracle):
    def __init__(self, max_step=200, action_type='relative', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_step = max_step
        assert action_type in ['relative', 'absolute']
        self._action_type = action_type

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._max_step = 200
        self._final_pos = np.random.uniform(*self._env.unwrapped._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)

    def select_action(self, ob, info):
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        gripper_opening = info['proprio/gripper_opening']  # noqa

        target_block = info['privileged/target_block']
        block_pos = info[f'privileged/block_{target_block}_pos']
        block_yaw = self.shortest_yaw(effector_yaw, info[f'privileged/block_{target_block}_yaw'][0])
        target_pos = info['privileged/target_block_pos']
        target_yaw = self.shortest_yaw(effector_yaw, info['privileged/target_block_yaw'][0])

        block_above_offset = np.array([0, 0, 0.18])
        above_threshold = 0.16
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        gripper_open = info['proprio/gripper_contact'] < 0.1
        above = effector_pos[2] > above_threshold
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.02
        target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.04
        target_pos_aligned = np.linalg.norm(target_pos - block_pos) <= 0.02
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04

        gain_pos = 5
        gain_yaw = 3
        action = np.zeros(5)
        if not target_pos_aligned:
            if not xy_aligned:
                self.print_phase('1: Move above the block')
                action = np.zeros(5)
                diff = block_pos + block_above_offset - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif not pos_aligned:
                self.print_phase('2: Move to the block')
                diff = block_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif pos_aligned and not gripper_closed:
                self.print_phase('3: Grasp')
                diff = block_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and not above and not target_xy_aligned:
                self.print_phase('4: Move in the air')
                diff = np.array([block_pos[0], block_pos[1], block_above_offset[2] * 2]) - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and above and not target_xy_aligned:
                self.print_phase('5: Move above the target')
                diff = target_pos + block_above_offset - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            else:
                self.print_phase('6: Move to the target')
                diff = target_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
        else:
            if not gripper_open:
                self.print_phase('7: Release')
                diff = target_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = -1
            elif gripper_open and not above:
                self.print_phase('8: Move in the air')
                diff = np.array([block_pos[0], block_pos[1], above_threshold * 2]) - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            else:
                self.print_phase('9: Move to the final position')
                diff = self._final_pos - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1

            if final_pos_aligned:
                self._done = True

        action = np.clip(action, -1, 1)
        if self._debug:
            print(action)

        self._step += 1
        if self._step == self._max_step:
            self._done = True

        return action if self._action_type == 'relative' else self.rel2abs(action)

    def rel2abs(self, action):
        action_range = np.array([0.05, 0.05, 0.05, 0.3, 1.0])
        action_low = -action_range
        action_high = action_range

        def unnormalize_action(action):
            """Unnormalize the action to the range [action_low, action_high]."""
            return 0.5 * (action + 1) * (action_high - action_low) + action_low

        env = self._env.unwrapped
        action = unnormalize_action(action)
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

        # Compute target effector pose based on the relative action.
        effector_pos = env._data.site_xpos[env._pinch_site_id].copy()
        effector_yaw = lie.SO3.from_matrix(
            env._data.site_xmat[env._pinch_site_id].copy().reshape(3, 3)
        ).compute_yaw_radians()
        gripper_opening = np.array(np.clip([env._data.qpos[env._gripper_opening_joint_id] / 0.8], 0, 1))
        target_effector_translation = effector_pos + a_pos
        target_effector_orientation = (  # noqa
            lie.SO3.from_z_radians(a_ori) @ lie.SO3.from_z_radians(effector_yaw) @ env._effector_down_rotation.inverse()
        )
        target_gripper_opening = gripper_opening + a_gripper

        act = np.concatenate(
            [
                target_effector_translation,
                np.array([effector_yaw + a_ori]),
                target_gripper_opening,
            ]
        )
        return act
