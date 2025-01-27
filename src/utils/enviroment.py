from datetime import datetime, date
from typing import Optional

from config import settings


def env():
    return settings.ENV


def is_local():
    return settings.ENV == "dev"


def is_env_prod():
    return settings.ENV == "prod"

def runner_type():
    return settings.get("RUNNER_TYPE", "").lower()