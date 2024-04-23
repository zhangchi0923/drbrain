import datetime
import logging
import os

def get_logger(log_id, pth):
    date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exe_logger = logging.getLogger()
    exe_logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(pth, 'log_' + date_time + '_' + log_id))
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    exe_logger.addHandler(handler)
    return exe_logger