import time
import yaml
from collect_data import collect_data
from typing import Dict
from loguru import logger


def main(configure: Dict[str, object]) -> None:

    proc_step = configure["options"]["proc_step"]

    if proc_step == 'collect':
        logger.info('start collecting data')
        collect_data.collect_data(configure)
        logger.info('finished stacking')

    else:
        raise ValueError('unexpected proc_step: %s' % proc_step)

    elapsed_time = time.time() - start_time
    time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    logger.info("elapsed_time: %s" % time_str)


if __name__ == '__main__':
    start_time = time.time()
    # Load the configuration settings from a YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    main(config)
