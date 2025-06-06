"""
The script was based on the original script from the Pretraining Subnet repository.
https://github.com/macrocosm-os/pretraining/blob/main/scripts/start_validator.py

This script runs a validator process and automatically updates it when a new version is released.
Command-line arguments will be forwarded to validator (`neurons/validator.py`), so you can pass
them like this:
    python3 scripts/start_validator.py --wallet.name=my-wallet
Auto-updates are enabled by default and will make sure that the latest version is always running
by pulling the latest version from git and upgrading python packages. This is done periodically.
Local changes may prevent the update, but they will be preserved.

The script will use the same virtual environment as the one used to run it. If you want to run
validator within virtual environment, run this auto-update script from the virtual environment.

Pm2 is required for this script. This script will start a pm2 process using the name provided by
the --pm2_name argument.
"""

import argparse
import logging
import subprocess
import sys
import time
import os
from datetime import timedelta
from shlex import split
from typing import List
from argparse import Namespace
from pathlib import Path

log = logging.getLogger(__name__)
UPDATES_CHECK_TIME = timedelta(minutes=5)
CURRENT_WORKING_DIR = Path(__file__).parent.parent

ECOSYSTEM_CONFIG_PATH = CURRENT_WORKING_DIR / "config" / "ecosystem.config.js"  # Path to the pm2 ecosystem config file

def get_version() -> str:
    """Extract the version as current git commit hash"""
    result = subprocess.run(
        split("git rev-parse HEAD"),
        check=True,
        capture_output=True,
        cwd=CURRENT_WORKING_DIR,
    )
    commit = result.stdout.decode().strip()
    assert len(commit) == 40, f"Invalid commit hash: {commit}"
    return commit[:8]


def generate_pm2_config(pm2_name: str, args: List[str]) -> None:
    """
    Generate a pm2 ecosystem config file to run the validator.
    """
    config_content = f"""
        module.exports = {{
            apps: [
            {{
                name: '{pm2_name}',
                script: 'neurons/validator.py',
                interpreter: '{sys.executable}',
                autorestart: true,
                restart_delay: 30000,
                max_restarts: 100,
                env: {{
                PYTHONPATH: '{os.environ.get('PYTHONPATH', '')}:./',
                }},
                args: '{' '.join(args)}'
            }}
            ]
        }};
    """
    with open(ECOSYSTEM_CONFIG_PATH, "w") as f:
        f.write(config_content)
    log.info("Generated pm2 ecosystem config at: %s", ECOSYSTEM_CONFIG_PATH)


def start_validator_process(pm2_name: str, args: List[str]) -> subprocess.Popen:
    """
    Spawn a new python process running neurons.validator using pm2.
    """
    assert sys.executable, "Failed to get python executable"
    generate_pm2_config(pm2_name, args)  # Generate the pm2 config file

    log.info("Starting validator process with pm2, name: %s", pm2_name)
    process = subprocess.Popen(
        [
            "pm2",
            "start",
            str(ECOSYSTEM_CONFIG_PATH)
        ],
        cwd=CURRENT_WORKING_DIR,
    )
    process.pm2_name = pm2_name

    return process


def stop_validator_process(process: subprocess.Popen) -> None:
    """Stop the validator process"""
    subprocess.run(
        ("pm2", "delete", process.pm2_name), cwd=CURRENT_WORKING_DIR, check=True
    )


def pull_latest_version() -> None:
    """
    Pull the latest version from git.
    This uses `git pull --rebase`, so if any changes were made to the local repository,
    this will try to apply them on top of origin's changes. This is intentional, as we
    don't want to overwrite any local changes. However, if there are any conflicts,
    this will abort the rebase and return to the original state.
    The conflicts are expected to happen rarely since validator is expected
    to be used as-is.
    """
    try:
        subprocess.run(
            split("git pull --rebase --autostash"), check=True, cwd=CURRENT_WORKING_DIR
        )
    except subprocess.CalledProcessError as exc:
        log.error("Failed to pull, reverting: %s", exc)
        subprocess.run(split("git rebase --abort"), check=True, cwd=CURRENT_WORKING_DIR)


def upgrade_packages() -> None:
    """
    Upgrade python packages by running `pip install --upgrade -r requirements.txt`.
    Notice: this won't work if some package in `requirements.txt` is downgraded.
    Ignored as this is unlikely to happen.
    """
    log.info("Upgrading packages")
    try:
        subprocess.run(
            split(f"{sys.executable} -m pip install --upgrade -r requirements.txt"),
            check=True,
            cwd=CURRENT_WORKING_DIR,
        )
    except subprocess.CalledProcessError as exc:
        log.error("Failed to upgrade packages, proceeding anyway. %s", exc)


def main(pm2_name: str, args_namespace: Namespace, extra_args: List[str]) -> None:
    """
    Run the validator process and automatically update it when a new version is released.
    This will check for updates every `UPDATES_CHECK_TIME` and update the validator
    if a new version is available. Update is performed as simple `git pull --rebase`.
    """

    args_list = []
    for key, value in vars(args_namespace).items():
        if value != '' and value is not None:
            args_list.append(f"--{key}")
            if not isinstance(value, bool):
                args_list.append(str(value))

    args_list.extend(extra_args)

    validator = start_validator_process(pm2_name, args_list)
    current_version = latest_version = get_version()
    log.info("Current version: %s", current_version)

    try:
        while True:
            pull_latest_version()
            latest_version = get_version()
            log.info("Latest version: %s", latest_version)

            if latest_version != current_version:
                log.info(
                    "Upgraded to latest version: %s -> %s",
                    current_version,
                    latest_version,
                )
                upgrade_packages()

                stop_validator_process(validator)
                validator = start_validator_process(pm2_name, args_list)
                current_version = latest_version

            time.sleep(UPDATES_CHECK_TIME.total_seconds())

    finally:
        stop_validator_process(validator)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Automatically update and restart the validator process when a new version is released.",
        epilog="Example usage: python start_validator.py --pm2_name 'net9vali' --wallet_name 'wallet1' --wallet_hotkey 'key123'",
    )

    parser.add_argument(
        "--pm2_name", default="cancer_ai_vali", help="Name of the PM2 process."
    )

    parser.add_argument(
        "--wallet.name", default="validator", help="Name of the wallet."
    )

    parser.add_argument(
        "--wallet.hotkey", default="default", help="Name of the hotkey."
    )

    parser.add_argument(
        "--subtensor.network", default="finney", help="Name of the network."
    )

    parser.add_argument(
        "--netuid", default="76", help="Netuid of the network."
    )

    parser.add_argument(
        "--logging.debug", default=1, help="Enable debug logging."
    )

    parser.add_argument(
        "--hf_token", default="", help="Access token for Hugging Face."
    )

    flags, extra_args = parser.parse_known_args()
    main(flags.pm2_name, flags, extra_args)
