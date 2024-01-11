import time
import yaml
from proc_data import proc_data
from typing import Dict
from loguru import logger


def main(configure: Dict[str, object]) -> None:

    proc_step = configure["options"]["proc_step"]

    if proc_step == 'is2proc':
        logger.info('start: processing ICESat-2 heights')
        proc_data.is2proc(configure)
        logger.info('end: processing ICESat-2 heights')

    elif proc_step == 'snow_off':
        logger.info('start: computing snow off correction')
        proc_data.snow_off(configure)
        logger.info('end: computing snow off correction')

    elif proc_step == 'snow_depth':
        logger.info('start: computing snow depth')
        proc_data.snow_depth_proc(configure)
        logger.info('end: computing snow depth')

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
