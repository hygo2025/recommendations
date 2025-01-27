from datetime import datetime, timedelta

def now_sub_days(subtract_days: int = 1) -> datetime:
    return datetime.now() - timedelta(days=subtract_days)