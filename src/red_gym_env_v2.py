import os
import re
import uuid
import json
import importlib
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy

# from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from global_map import local_to_global, GLOBAL_MAP_SHAPE
from config import *

event_flags_start = 0xD747
event_flags_end = 0xD87E  # expand for SS Anne # old - 0xD7F6
museum_ticket = (0xD754, 0)


def load_map_module(map_name):
    path = MAP_MODULE_PATH / f"{map_name}.py"
    if not path.exists():
        print(f"[WARN] Map module not found: {path}")
        return None, None, None, None

    spec = importlib.util.spec_from_file_location(map_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tile_type, mod.map_connection, mod.tile_map, mod.coll_map


def parse_object_sprites(asm_path):
    if not os.path.exists(asm_path):
        print(f"[WARN] asm not found: {asm_path}")
        return []

    sprite_names = []
    pattern = re.compile(r"object_event\s+\d+,\s*\d+,\s*([A-Z0-9_]+)")

    with open(asm_path, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                sprite = match.group(1)
                sprite_names.append(sprite)
    return sprite_names


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = 3

        # added resources
        self.map_names = load_json(MAP_NAMES_PATH)
        self.charmap = load_json(CHARMAP_PATH)
        self.item_names = load_json(ITEM_NAMES_PATH)
        self.species_names = load_json(SPECIES_NAMES_PATH)
        self.type_names = load_json(TYPE_NAMES_PATH)
        self.move_names = load_json(MOVE_NAMES_PATH)
        self.asm_dir = ASM_DIR

        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        self.essential_map_locations = {
            v: i
            for i, v in enumerate(
                [40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65]
            )
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open(EVENTS_PATH) as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.enc_freqs = 8

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(
                    low=0, high=255, shape=self.output_shape, dtype=np.uint8
                ),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                    dtype=np.uint8,
                ),
                "recent_actions": spaces.MultiDiscrete(
                    [len(self.valid_actions)] * self.frame_stacks
                ),
            }
        )

        head = "null" if config["headless"] else "SDL2"

        # log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            # debugging=False,
            # disable_input=False,
            window=head,
        )

        # self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options={}):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats = []

        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)

        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        self.base_event_flags = sum(
            [
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ]
        )

        self.current_event_flags_set = {}

        # experiment!
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2, 2, 1))
            ).astype(np.uint8)
        return game_pixels_render

    def _get_obs(self):

        screen = self.render()

        self.update_recent_screens(screen)

        # normalize to approx 0-1
        level_sum = 0.02 * sum(
            [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        )

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": self.fourier_encode(level_sum),
            "badges": np.array(
                [int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8
            ),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
        }

        return observation

    def step(self, action):

        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()

        self.party_size = self.read_m(0xD163)

        new_reward = self.update_reward()

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_and_print_info(step_limit_reached, obs)

        # create a map of all event flags set, with names where possible
        # if step_limit_reached:
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        # TODO this currently seems to be broken!
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1

        return obs, new_reward, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
            }
        )

    def start_video(self):

        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])
        self.map_frame_writer.add_image(self.get_explore_map())

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        # if not in battle
        if self.read_m(0xD057) == 0:
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1
            # self.seen_coords[coord_string] = self.step_count

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0] - self.coords_pad : c[0] + self.coords_pad,
                c[1] - self.coords_pad : c[1] + self.coords_pad,
            ]
        return repeat(out, "h w -> (h h2) (w w2)", h2=2, w2=2)

    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:, :, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:, :, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:, :, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

    def read_m(self, addr):
        # return self.pyboy.get_memory_value(addr)
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit)
            for i in range(event_flags_start, event_flags_end)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 4,
            # "level": self.reward_scale * self.get_levels_reward(),
            "heal": self.reward_scale * self.total_healing_rew * 10,
            # "op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            # "dead": self.reward_scale * self.died_count * -0.1,
            "badge": self.reward_scale * self.get_badges() * 10,
            "explore": self.reward_scale
            * self.explore_weight
            * len(self.seen_coords)
            * 0.1,
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05,
        }

        return state_scores

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max(
                [
                    self.read_m(a)
                    for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
                ]
            )
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    def read_hp_fraction(self):
        hp_sum = sum(
            [
                self.read_hp(add)
                for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
            ]
        )
        max_hp_sum = sum(
            [
                self.read_hp(add)
                for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
            ]
        )
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(
            self.max_map_progress, self.get_map_progress(map_idx)
        )

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    ##### custom functions for logging

    def take_screenshot(self, dir_name, img_name):

        os.makedirs(dir_name, exist_ok=True)

        # Ensure filename ends with .png
        if not img_name.endswith(".png"):
            img_name += ".png"

        # Save the new screenshot
        img = self.pyboy.screen.image
        img.save(os.path.join(dir_name, img_name))
        return img

    def decode_tilemap(self):
        mem = self.pyboy.memory
        TILEMAP_ADDR = 0xC3A0
        SCREEN_WIDTH = 20
        SCREEN_HEIGHT = 18
        charmap = self.charmap

        tile_lines = []
        for row in range(SCREEN_HEIGHT):
            line = []
            for col in range(SCREEN_WIDTH):
                addr = TILEMAP_ADDR + row * SCREEN_WIDTH + col
                byte = mem[addr]
                ch = charmap.get(str(byte), " ")
                if ch == "<NULL>":
                    ch = " "
                line.append(ch)
            tile_lines.append(line)
        return tile_lines

    def get_filtered_screen_text(self, tile_lines):
        lines = []
        for line in tile_lines:
            line_str = "".join(line).strip()  # Without blank line
            if "QRSTUVWXYZ():;[]" in line_str:
                return "N/A"
            if line_str:
                lines.append(line_str)
        return "\n".join(lines) if lines else "N/A"

    def find_selection_box(self, tile_lines):
        height = len(tile_lines)
        width = len(tile_lines[0]) if tile_lines else 0

        # 1. Find ▶ cursor
        cursor_y, cursor_x = None, None
        for y_idx, row_chars in enumerate(tile_lines):
            for x_idx, char_val in enumerate(row_chars):
                if char_val == "▶":
                    cursor_y, cursor_x = y_idx, x_idx
                    break
            if cursor_y is not None:
                break
        if cursor_y is None:
            return None

        # 2. Find vertical │ borders for the line containing the cursor
        def find_vertical_border_char_x(line_chars, start_x_coord, direction_step):
            current_x = start_x_coord
            while 0 <= current_x < len(line_chars):
                if line_chars[current_x] == "│":
                    return current_x
                current_x += direction_step
            return None

        x1 = find_vertical_border_char_x(tile_lines[cursor_y], cursor_x, -1)
        x2 = find_vertical_border_char_x(tile_lines[cursor_y], cursor_x, 1)

        if x1 is None or x2 is None or x2 <= x1:
            return None

        # 3. Find horizontal borders (y1 and y2) using the Lua-like logic

        # Find y1 (upwards)
        found_y1 = None
        for y_scan in range(cursor_y - 1, -1, -1):
            # Check if tile_lines[y_scan] is long enough
            if len(tile_lines[y_scan]) <= x1 or len(tile_lines[y_scan]) <= x2 - 1:
                continue  # Line is too short

            is_line_with_dash = False
            for x_scan in range(x1 + 1, x2):
                if tile_lines[y_scan][x_scan] == "─":
                    is_line_with_dash = True
                    break
            if is_line_with_dash:
                found_y1 = y_scan
                break

        # Find y2 (downwards)
        found_y2 = None
        for y_scan in range(cursor_y + 1, height):
            # Check if tile_lines[y_scan] is long enough
            if len(tile_lines[y_scan]) <= x1 or len(tile_lines[y_scan]) <= x2 - 1:
                continue  # Line is too short

            is_line_with_dash = False
            for x_scan in range(x1 + 1, x2):
                if tile_lines[y_scan][x_scan] == "─":
                    is_line_with_dash = True
                    break
            if is_line_with_dash:
                found_y2 = y_scan
                break

        if found_y1 is None or found_y2 is None:
            # This can happen if the box is at the very edge of the screen
            # or if the horizontal borders don't use '─'
            # print(f"Debug: Lua-like horizontal borders not found. y1={found_y1}, y2={found_y2}")

            # Fallback to check for explicit corners if Lua-like borders fail
            # This part is from the previous Pythonic suggestion
            # Check for y1 with corners
            y1_corner_fallback = None
            for y_s in range(cursor_y - 1, -1, -1):
                line = tile_lines[y_s]
                if (
                    len(line) > x2
                    and line[x1] in ["┌", "├"]
                    and line[x2] in ["┐", "┤"]
                    and all(c == "─" for c in line[x1 + 1 : x2])
                ):
                    y1_corner_fallback = y_s
                    break
            if found_y1 is None:
                found_y1 = y1_corner_fallback
            if (
                found_y1 is None
                and len(tile_lines[0]) > x2
                and tile_lines[0][x1] in ["┌", "├"]
                and tile_lines[0][x2] in ["┐", "┤"]
                and all(c == "─" for c in tile_lines[0][x1 + 1 : x2])
            ):
                found_y1 = 0

            # Check for y2 with corners
            y2_corner_fallback = None
            for y_s in range(cursor_y + 1, height):
                line = tile_lines[y_s]
                if (
                    len(line) > x2
                    and line[x1] in ["└", "├"]
                    and line[x2] in ["┘", "┤"]
                    and all(c == "─" for c in line[x1 + 1 : x2])
                ):
                    y2_corner_fallback = y_s
                    break
            if found_y2 is None:
                found_y2 = y2_corner_fallback
            if (
                found_y2 is None
                and len(tile_lines[height - 1]) > x2
                and tile_lines[height - 1][x1] in ["└", "├"]
                and tile_lines[height - 1][x2] in ["┘", "┤"]
                and all(c == "─" for c in tile_lines[height - 1][x1 + 1 : x2])
            ):
                found_y2 = height - 1

            if found_y1 is None or found_y2 is None:
                return None  # Still not found after fallbacks

        return {"x1": x1, "x2": x2, "y1": found_y1, "y2": found_y2}

    def extract_selection_box_text(self, tile_lines, box):
        lines = []
        for y in range(box["y1"] + 1, box["y2"]):
            line = tile_lines[y][box["x1"] + 1 : box["x2"]]
            line_str = "".join(line).strip()
            if line_str:
                lines.append(line_str)
        return lines

    def get_dialog(self):
        tile_lines = self.decode_tilemap()

        # [Filtered Screen Text]
        text = "[Filtered Screen Text]\n"
        text += self.get_filtered_screen_text(tile_lines) + "\n"

        # [Selection Box Text]
        text += "\n[Selection Box Text]\n"
        box = self.find_selection_box(tile_lines)
        if box:
            lines = self.extract_selection_box_text(tile_lines, box)
            text += "----------------\n"
            for line in lines:
                text += line + "\n"
            text += "----------------\n"
        else:
            text += "N/A\n"

        return text

    def get_inventory(self):
        mem = self.pyboy.memory
        item_names = self.item_names

        item_count = mem[0xD31D]
        base = 0xD31E

        text = "[Bag]\n"
        text += f"({item_count} items):\n" if item_count > 0 else "N/A\n"

        for i in range(item_count):
            addr = base + i * 2
            item_id = mem[addr]
            quantity = mem[addr + 1]

            if item_id == 0:
                continue  # Empty slot

            name = item_names.get(str(item_id), f"Unknown Item (ID:{item_id:02X})")
            text += f"- {name} × {quantity}\n"

        return text

    def get_map_visual(self, coll_map, player_x, player_y, object_coords):
        lines = []
        for sy in range(max(0, player_y - 4), player_y + 5):
            line = ""
            for sx in range(max(0, player_x - 4), player_x + 5):
                if (sx, sy) in object_coords:
                    cell = object_coords[(sx, sy)]
                elif coll_map and sy < len(coll_map) and sx < len(coll_map[sy]):
                    cell = coll_map[sy][sx]
                else:
                    cell = "?"
                line += f"({sx:2d},{sy:2d}): {cell}\t"
            lines.append(line)
        return "\n".join(lines)

    def get_object_coords(self, player_x, player_y):
        mem = self.pyboy.memory
        object_coords = {}

        map_id = mem[0xD35E]
        map_name = self.map_names.get(str(map_id), f"UNKNOWN_{map_id}")

        asm_path = os.path.join(self.asm_dir, f"{map_name}.asm")
        sprite_list = parse_object_sprites(asm_path)

        for i in range(1, 16):
            base = 0xC100 + i * 16
            if mem[base + 2] == 0xFF:
                continue

            y_delta = ((mem[base + 4] + 4) % 256 - (mem[0xC100 + 4] + 4)) // 16
            x_delta = (mem[base + 6] - mem[0xC100 + 6]) // 16

            obj_x = player_x + x_delta
            obj_y = player_y + y_delta

            sprite_name = sprite_list[i - 1] if i - 1 < len(sprite_list) else f"OBJ_{i}"
            object_coords[(obj_x, obj_y)] = f"{sprite_name}_{i}"

        return object_coords

    def get_player_pos(self):
        mem = self.pyboy.memory
        map_id = mem[0xD35E]
        map_name = self.map_names.get(str(map_id), f"UNKNOWN_{map_id}")
        player_x = mem[0xD362]
        player_y = mem[0xD361]
        return (player_x, player_y, map_name)

    def get_map_info(self):
        mem = self.pyboy.memory
        map_id = mem[0xD35E]
        map_name = self.map_names.get(str(map_id), f"UNKNOWN_{map_id}")

        max_width = mem[0xD369] * 2 - 1
        max_height = mem[0xD368] * 2 - 1
        player_x = mem[0xD362]
        player_y = mem[0xD361]
        direction_code = mem[0xC109]

        direction_map = {0: "down", 4: "up", 8: "left", 12: "right"}
        facing = direction_map.get(direction_code, "None")

        # Load map module
        tile_type, map_connection, tile_map, coll_map = load_map_module(map_name)

        # Extract object's location
        object_coords = self.get_object_coords(player_x, player_y)

        text = "[Map Info]\n"
        text += f"Map Name: {map_name}, (x_max , y_max): ({max_width}, {max_height})\n"
        text += f"Map type: {tile_type or 'UNKNOWN'}\n"
        text += f"Expansion direction: {map_connection or 'None'}\n"
        text += f"Your position (x, y): ({player_x}, {player_y})\n"
        text += f"Your facing direction: {facing}\n"
        text += "Action instruction\n"
        text += " - up: (x, y) -> (x, y-1)\n"
        text += " - down: (x, y) -> (x, y+1)\n"
        text += " - left: (x, y) -> (x-1, y)\n"
        text += " - right: (x, y) -> (x+1, y)\n"

        text += "\nMap on Screen:\n"
        if self.get_battle_state() != "Field":
            text += "Not in Field State"
            return text
        for sy in range(max(0, player_y - 4), min(player_y + 4, max_height) + 1):
            for sx in range(max(0, player_x - 4), min(player_x + 5, max_width) + 1):
                key = (sx, sy)
                if key in object_coords:
                    cell = object_coords[key]
                elif coll_map and sy < len(coll_map) and sx < len(coll_map[sy]):
                    cell = coll_map[sy][sx]
                else:
                    cell = "?"
                text += f"({sx:2d}, {sy:2d}): {cell}\t"
            text += "\n"

        return text

    # def get_battle_state(self):
    #     mem = self.pyboy.memory
    #     wIsInBattle = mem[0xD057]
    #     wLinkState = mem[0xD72E]
    #     title_check = mem[0xC0EF]
    #     map_id = mem[0xD35C]

    #     if (title_check == 0x1F and map_id == 0x00) or title_check is None:
    #         return "Title"
    #     elif wLinkState == 0x05:
    #         return "LinkBattle"
    #     elif wIsInBattle == 0x01:
    #         return "WildBattle"
    #     elif wIsInBattle == 0x02:
    #         return "TrainerBattle"
    #     else:
    #         return "Field"

    def get_battle_state(self):
        mem = self.pyboy.memory
        wIsInBattle = mem[0xD057]
        wLinkState = mem[0xD72E]
        title_check = mem[0xC0EF]
        map_id = mem[0xD35C]

        if (title_check == 0x1F and map_id == 0x00) or title_check is None:
            battle_state = "Title"
        elif wLinkState == 0x05:
            battle_state = "LinkBattle"
        elif wIsInBattle == 0x01:
            battle_state = "WildBattle"
        elif wIsInBattle == 0x02:
            battle_state = "TrainerBattle"
        else:
            battle_state = "Field"

        dialog_text = self.get_dialog()
        has_dialog = "[Filtered Screen Text]\nN/A" not in dialog_text

        if has_dialog and battle_state == "Field":
            if "CONTINUE" in dialog_text and "NEW GAME" in dialog_text:
                battle_state = "Title"
            else:
                battle_state = "Dialog"

        return battle_state

    def get_enemy_info(self):
        mem = self.pyboy.memory
        species_id = mem[0xCFE5]
        level = mem[0xCFF3]
        hp = (mem[0xCFE6] << 8) + (mem[0xCFE7])
        max_hp = (mem[0xCFF4] << 8) + (mem[0xCFF5])
        status = mem[0xCFE9]

        battle_state = self.get_battle_state()
        text = "\n[Enemy Pokemon]\n"

        if battle_state in ["WildBattle", "TrainerBattle", "LinkBattle"]:
            name = self.species_names.get(str(species_id), f"UNKNOWN_{species_id}")
            text += f"Name: {name}\n"
            text += f"Level: {level}\n"

            if max_hp > 0:
                hp_pct = int((hp / max_hp) * 100)
                text += f"HP_percentage: {hp_pct}%\n"
            else:
                text += f"HP_percentage: Unknown\n"

            status_text = {
                0: "Normal state",
                1: "Sleep",
                2: "Sleep",
                3: "Sleep",
                4: "Sleep",
                5: "Sleep",
                6: "Sleep",
                7: "Sleep",
                8: "Poisoned",
                16: "Burned",
                32: "Frozen",
                64: "Paralyzed",
            }.get(status, "Normal state")
            text += f"Status: {status_text}\n"

        else:
            text += "- Not in battle\n"

        return text

    def get_active_pokemon_name(self):
        mem = self.pyboy.memory
        active_name = ""
        for i in range(11):
            b = mem[0xD009 + i]
            ch = self.charmap.get(str(b), "")
            if ch in ["<NULL>", "@"]:
                break
            active_name += ch
        return active_name

    def get_party_info(self):
        mem = self.pyboy.memory
        active_name = self.get_active_pokemon_name()
        text = "\n[Current Party]\n"

        for i in range(6):
            base = 0xD16B + i * 0x2C
            species_id = mem[base]
            if species_id == 0:
                text += "No more Pokemons\n"
                break

            level = mem[base + 0x21]
            hp = (mem[base + 1] << 8) + mem[base + 2]
            max_hp = (mem[base + 0x22] << 8) + mem[base + 0x23]

            # nickname
            name_bytes = mem[0xD2B5 + i * 11 : 0xD2B5 + (i + 1) * 11]
            nickname = ""
            for b in name_bytes:
                ch = self.charmap.get(str(b), "")
                if ch in ["<NULL>", "@"]:
                    break
                nickname += ch

            species_name = self.species_names.get(
                str(species_id), f"UNKNOWN_{species_id}"
            )
            type1 = self.type_names.get(
                str(mem[base + 0x05]), f"UNKNOWN_{mem[base + 0x05]}"
            )
            type2 = self.type_names.get(
                str(mem[base + 0x06]), f"UNKNOWN_{mem[base + 0x06]}"
            )

            # moves and PP
            moves = []
            for j in range(4):
                move_id = mem[0xD173 + i * 0x2C + j]
                pp = mem[0xD188 + i * 0x2C + j]
                move_name = self.move_names.get(str(move_id), "Not Learned")
                moves.append(f"{move_name}(pp={pp})")

            status = mem[base + 4]
            status_text = {
                0: "Normal state",
                1: "Sleep",
                2: "Sleep",
                3: "Sleep",
                4: "Sleep",
                5: "Sleep",
                6: "Sleep",
                7: "Sleep",
                8: "Poisoned",
                16: "Burned",
                32: "Frozen",
                64: "Paralyzed",
            }.get(status, "Normal state")

            prefix = (
                "[In-battle] Name:"
                if nickname == active_name and "Battle" in self.get_battle_state()
                else "Name: "
            )
            if type1 != type2:
                text += f"{prefix}{nickname}, Species: {species_name}, Level: {level}, Status: {status_text}, Type: {type1}/{type2}, HP: {hp}/{max_hp}"
            else:
                text += f"{prefix}{nickname}, Species: {species_name}, Level: {level}, Status: {status_text}, Type: {type1}, HP: {hp}/{max_hp}, Moves: "
            text += ", ".join(moves) + "\n"

        return text

    def get_badge_info(self):
        mem = self.pyboy.memory
        badge_mask = mem[0xD356]
        badge_names = [
            "Boulder",
            "Cascade",
            "Thunder",
            "Rainbow",
            "Soul",
            "Marsh",
            "Volcano",
            "Earth",
        ]
        badges = [badge_names[i] for i in range(8) if badge_mask & (1 << i)]

        text = "\n[Badge List]\n"
        text += ", ".join(badges) if badges else "N/A"
        text += "\n"
        return text

    def get_money_info(self):
        mem = self.pyboy.memory

        def bcd_to_int(bcd):
            return (bcd >> 4) * 10 + (bcd & 0xF)

        money = (
            bcd_to_int(mem[0xD347]) * 10000
            + bcd_to_int(mem[0xD348]) * 100
            + bcd_to_int(mem[0xD349])
        )
        return f"\n[Current Money]: ¥{money}\n"

    def get_state(self):
        dialog_text = self.get_dialog()
        has_dialog = "[Filtered Screen Text]\nN/A" not in dialog_text

        battle_state = self.get_battle_state()
        if has_dialog and battle_state == "Field":
            battle_state = "Dialog"

        parts = []
        parts.append("State: " + battle_state + "\n")
        parts.append(self.get_dialog())
        parts.append(self.get_enemy_info())
        parts.append(self.get_party_info())
        parts.append(self.get_badge_info())
        parts.append(self.get_inventory())
        parts.append(self.get_money_info())
        parts.append(self.get_map_info())

        return "\n".join(parts)

    def parse_game_state(self):

        text = self.get_state()
        result = {}

        # 1. State
        state_match = re.search(r"State:\s*(\w+)", text)
        result["state"] = state_match.group(1) if state_match else None

        # 2. Filtered Screen Text
        filtered_text = re.search(
            r"\[Filtered Screen Text\]\n(.*?)(?=\[Selection Box Text\])",
            text,
            re.DOTALL,
        )
        text_tmp = filtered_text.group(1).strip()
        result["filtered_screen_text"] = text_tmp if text_tmp != "" else "N/A"

        # 3. Selection Box Text
        selection_box = re.search(
            r"\[Selection Box Text\]\n(.*?)(?=\[Enemy Pokemon\])", text, re.DOTALL
        )
        text_tmp = selection_box.group(1).strip()
        result["selection_box_text"] = text_tmp if text_tmp != "" else "N/A"

        # 4. Enemy Pokemon
        enemy_pokemon = {}
        enemy_section = re.search(
            r"\[Enemy Pokemon\]\n(.*?)(?=\[Current Party\])", text, re.DOTALL
        )
        if enemy_section:
            for line in enemy_section.group(1).splitlines():
                if ": " in line:
                    key, value = line.split(": ", 1)
                    enemy_pokemon[key.strip()] = value.strip()
        result["enemy_pokemon"] = enemy_pokemon

        # 5. Your Party
        party_match = re.search(
            r"\[Current Party\]\n(.*?)(?=\[Badge List\])", text, re.DOTALL
        )
        result["your_party"] = party_match.group(1).strip() if party_match else ""

        # 6. Badge List
        badge_match = re.search(r"\[Badge List\]\n(.*?)(?=\[Bag\])", text, re.DOTALL)
        result["badge_list"] = badge_match.group(1).strip() if badge_match else ""

        # 7. Inventory
        inventory_match = re.search(
            r"\[Bag\]\n(.*?)(?=\[Current Money\])", text, re.DOTALL
        )
        result["inventory"] = (
            inventory_match.group(1).strip() if inventory_match else ""
        )

        # 8. Current Money
        money_match = re.search(r"\[Current Money\]:\s*¥(\d+)", text)
        result["money"] = int(money_match.group(1)) if money_match else 0

        # 9. Map Info
        map_info = {}
        map_section = re.search(r"\[Map Info\]\n(.*)", text, re.DOTALL)
        if map_section:
            map_text = map_section.group(1)
            map_name_match = re.search(r"Map Name:\s*(.*?),", map_text)
            map_info["map_name"] = map_name_match.group(1) if map_name_match else None

            map_type_match = re.search(r"Map type:\s*(.*)", map_text)
            map_info["map_type"] = (
                map_type_match.group(1).strip() if map_type_match else None
            )

            expansion_match = re.search(r"Expansion direction:\s*(.*)", map_text)
            map_info["expansion_direction"] = (
                expansion_match.group(1).strip() if expansion_match else None
            )

            coords_match = re.search(
                r"\(x_max , y_max\):\s*\((\d+),\s*(\d+)\)", map_text
            )
            map_info["x_max"] = int(coords_match.group(1)) if coords_match else None
            map_info["y_max"] = int(coords_match.group(2)) if coords_match else None

            pos_match = re.search(r"Your position \(x, y\): \((\d+), (\d+)\)", map_text)
            map_info["player_pos_x"] = int(pos_match.group(1)) if pos_match else None
            map_info["player_pos_y"] = int(pos_match.group(2)) if pos_match else None

            facing_match = re.search(r"Your facing direction:\s*(\w+)", map_text)
            map_info["facing"] = facing_match.group(1) if facing_match else None

            try:
                # Optional: extract action instructions and screen map
                map_info["map_screen_raw"] = (
                    re.search(r"Map on Screen:\n(.+)", map_text, re.DOTALL)
                    .group(1)
                    .strip()
                )
            except:
                map_info["map_screen_raw"] = None

        result["map_info"] = map_info

        return result

    def save_sav_file(self, path):
        with open(path, "wb") as f:
            f.write(self.pyboy.cartridge.savefile)
        print(f"[INFO] .sav File Saved: {path}")
