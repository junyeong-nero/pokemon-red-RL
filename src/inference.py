import datetime
import json
import sys
import time
import traceback
from pathlib import Path

from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from config import AGENT_ENABLED, INIT_STATE_PATH, ROM_PATH, RUNS_DIR, STATE_LOG_DIR
from monitor import VideoMonitor


class Evaluator:
    """Simple rule-based progression tracker used during inference."""

    def __init__(self):
        self.score = 0
        self.prev_state_dict = None
        self.map_flag = False
        self.ball_flag = False
        self.catch_flag = False
        self.pewter_flag = False
        self.leader_flag = False

    @staticmethod
    def _get_map_name(state_dict):
        return (state_dict.get("map_info", {}) or {}).get("map_name", "") or ""

    def evaluate(self, state_dict):
        map_name = self._get_map_name(state_dict)
        prev_map_name = self._get_map_name(self.prev_state_dict or {})

        map_screen_raw = (state_dict.get("map_info", {}) or {}).get(
            "map_screen_raw", ""
        ) or ""
        your_party = state_dict.get("your_party", "") or ""
        inventory = state_dict.get("inventory", "") or ""
        badge_list = state_dict.get("badge_list", "") or ""
        current_state = state_dict.get("state", "") or ""
        prev_state = (self.prev_state_dict or {}).get("state", "") or ""

        if self.score == 0:
            if (
                prev_map_name
                and map_name
                and prev_map_name != map_name
                and "RedsHouse" not in map_name
            ):
                self.score += 1
        elif self.score == 1:
            if "SPRITE_OAK" in map_screen_raw:
                self.score += 1
        elif self.score == 2:
            if "Name" in your_party:
                self.score += 1
        elif self.score == 3:
            if (
                prev_state
                and current_state
                and prev_state != current_state
                and "Battle" in prev_state
            ):
                self.score += 1
        elif self.score == 4:
            if "Viridian" in map_name:
                self.score += 1
        elif self.score == 5:
            if "OAK's PARCEL" in inventory:
                self.score += 1
        elif self.score == 6:
            if "OAK's PARCEL" not in inventory:
                self.score += 1
        # elif self.score > 6:
        #     if not self.map_flag and "TOWN MAP" in inventory:
        #         self.score += 1
        #         self.map_flag = True
        #     if not self.ball_flag and "BALL" in inventory:
        #         self.score += 1
        #         self.ball_flag = True
        #     if not self.catch_flag and "\nName" in your_party:
        #         self.score += 1
        #         self.catch_flag = True
        #     if not self.pewter_flag and "Pewter" in map_name:
        #         self.score += 1
        #         self.pewter_flag = True
        #     if not self.leader_flag and "Boulder" in badge_list:
        #         self.score += 1
        #         self.leader_flag = True

        done = self.score >= 7
        self.prev_state_dict = state_dict
        return self.score, done


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        # env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def get_most_recent_zip_with_age(folder_path: Path):
    folder_path = Path(folder_path)
    zip_files = list(folder_path.glob("*.zip"))

    if not zip_files:
        return None, None

    most_recent_zip = max(zip_files, key=lambda p: p.stat().st_mtime)

    current_time = time.time()
    modification_time = most_recent_zip.stat().st_mtime
    age_in_hours = (current_time - modification_time) / 3600

    return most_recent_zip, age_in_hours


def _default_log_path(prefix: str = "state_log") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_LOG_DIR / f"{prefix}_{timestamp}.jsonl"


class JsonStateLogger:
    """Append game state snapshots to a JSONL file for later analysis."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else _default_log_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def __enter__(self) -> "JsonStateLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(
        self,
        *,
        step: int,
        state_text: str,
        terminated: bool,
        truncated: bool,
        info: dict | None = None,
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "step": step,
            "state_text": state_text,
            "terminated": terminated,
            "truncated": truncated,
        }
        if info:
            payload["info"] = info
        self._fh.write(json.dumps(payload, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _safe_close(name: str, closer):
    if closer is None:
        return
    try:
        closer()
    except Exception as exc:
        print(f"[WARN] failed to close {name}: {exc}", file=sys.stderr)


def main():

    # Reuse the shared runs directory to avoid creating per-run session folders during inference.
    sess_path = RUNS_DIR
    ep_length = 2**23

    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": INIT_STATE_PATH,
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": ROM_PATH,
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    env = None
    evaluator = Evaluator()
    exit_code = 0
    try:
        env = make_env(
            0, env_config
        )()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

        # env_checker.check_env(env)
        file_name = None
        most_recent_checkpoint, time_since = get_most_recent_zip_with_age(RUNS_DIR)
        if most_recent_checkpoint is not None:
            file_name = str(most_recent_checkpoint)
            print(f"using checkpoint: {file_name}, which is {time_since} hours old")
        else:
            raise FileNotFoundError(f"No checkpoint zip files found in {RUNS_DIR}")

        # could optionally manually specify a checkpoint here
        # file_name = "runs/poke_41943040_steps.zip"
        print("\nloading checkpoint")
        model = PPO.load(
            file_name, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0}
        )

        obs, info = env.reset()

        with VideoMonitor() as monitor, JsonStateLogger() as state_logger:
            print(f"recording video to {monitor.output_path}")
            print(f"logging game state to {state_logger.path}")
            terminated = False
            truncated = False
            info = {}
            while True:
                if AGENT_ENABLED:
                    action, _states = model.predict(obs, deterministic=False)
                    obs, rewards, terminated, truncated, info = env.step(action)
                else:
                    env.pyboy.tick(1, True)
                    obs = env._get_obs()
                    truncated = env.step_count >= env.max_steps - 1
                    terminated = False
                    info = {}

                frame = env.render(reduce_res=False)[:, :, 0]

                state_text = env.get_state()
                state_dict = env.parse_game_state()
                score, done = evaluator.evaluate(state_dict)
                state_logger.write(
                    step=env.step_count,
                    state_text=state_text,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                print(state_text)
                print(f"score: {score}/7")
                monitor.write(frame)

                if truncated or terminated or done:
                    break
    except Exception as exc:
        exit_code = 1
        print(f"[ERROR] inference failed: {exc}", file=sys.stderr)
        traceback.print_exc()
    finally:
        _safe_close("environment", getattr(env, "close", None))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
