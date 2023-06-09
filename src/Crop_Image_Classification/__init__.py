import os
import sys
import logging
from datetime import datetime

logging_string = "[%(asctime)s] || %(filename)s || %(lineno)d || %(name)s || %(funcName)s() || %(lineno)s || %(levelname)s || %(message)s"

log_dir = 'logs'


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y%m%d%H')}"

log_file_path = os.path.join(log_dir, get_current_time_stamp(), 'running_logs.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format=logging_string,
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger()
