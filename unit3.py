import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "logs" / "dqn.yml"   # adjust if it’s elsewhere
LOGDIR = HERE / "logs"

def run_cmd(args):
    subprocess.run([sys.executable, "-m"] + args, check=True)

run_cmd([
    "rl_zoo3.train",
    "--algo", "dqn",
    "--env", "SpaceInvadersNoFrameskip-v4",
    "-c", str(CONFIG),
    "-f", str(LOGDIR)
])

run_cmd([
    "rl_zoo3.enjoy",
    "--algo", "dqn",
    "--env", "SpaceInvadersNoFrameskip-v4",
    "--no-render",
    "--n-timesteps", "5000",
    "-f", str(LOGDIR)
])
