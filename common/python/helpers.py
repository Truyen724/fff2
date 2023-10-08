
import logging as log


def resolution(value):
    try:
        result = [int(v) for v in value.split('x')]
        if len(result) != 2:
            raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    except ValueError:
        raise RuntimeError('Correct format of --output_resolution parameter is "width"x"height".')
    return result

def log_latency_per_stage(*pipeline_metrics):
    stages = ('Decoding', 'Preprocessing', 'Inference', 'Postprocessing', 'Rendering')
    for stage, latency in zip(stages, pipeline_metrics):
        log.info('\t{}:\t{:.1f} ms'.format(stage, latency))
