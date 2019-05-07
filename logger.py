import logging
import sys

formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s] %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)
