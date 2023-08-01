""" SeismicPro-specific functions for Benchmark """

import sys

import numpy as np

sys.path.insert(0, '..')
from seismicpro import Survey, make_prestack_segy


def make_benchmark_data(path):
    """Generate a SEG-Y file with specific geometry so that CDP gathers contain the same number of
    traces and construct survey objects for benchmark.
    """
    # The geometry defined below should be changed only together with survey filtering parameters
    # to ensure that after filtering all the gathers \ supergathers have the same number of traces
    make_prestack_segy(path, fmt=1, survey_size=(400, 400), sources_step=(5, 5), receivers_step=(5, 5),
                       activation_dist=(50, 50), bin_size=(10, 10))

    # Load headers and add synthetic FirstBreak times
    sur = Survey(path, header_index=['INLINE_3D', 'CROSSLINE_3D'],
                 header_cols='offset', name='raw')
    sur['FirstBreak'] = np.abs(sur['offset'] / 1.5 + np.random.normal(0, 30, size=sur.n_traces))

    def edge_lines_filter(line, num_lines):
        return (line >= line.min() + num_lines) & (line <= line.max() - num_lines)

    # Drop three lines of CDPs from each side of the survey, since they have less traces than central ones
    survey = (sur.filter(edge_lines_filter, 'CROSSLINE_3D', num_lines=3)
                 .filter(edge_lines_filter, 'INLINE_3D', num_lines=3))

    sg_survey = survey.generate_supergathers(origin=0, size=3, step=1)
    # Drop one line of supergathers from each side of the survey, since they have less traces than central ones
    sg_survey = (sg_survey.filter(edge_lines_filter, 'SUPERGATHER_CROSSLINE_3D', num_lines=1)
                          .filter(edge_lines_filter, 'SUPERGATHER_INLINE_3D', num_lines=1))

    return survey, sg_survey
