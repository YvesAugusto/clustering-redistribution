import logging


# configure the logging
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
# # create the logger
logger = logging.getLogger('Face-Recognition')
