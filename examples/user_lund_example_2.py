import ast
import datetime
from operator import itemgetter
from os import path
import urllib

import numpy as np

from ocw.dataset import Dataset, Bounds
import ocw.data_source.local as local
import ocw.data_source.rcmed as rcmed
import ocw.dataset_processor as dsp
import ocw.evaluation as evaluation
import ocw.metrics as metrics
import ocw.plotter as plotter

EU_CORDEX_PR_KNMI = "pr_EUR-44i_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-RACMO22E_v1_mon.nc"
EU_CORDEX_PR_DNI = "pr_EUR-44i_ECMWF-ERAINT_evaluation_r1i1p1_DMI-HIRHAM5_v1_mon.nc"
EU_CORDEX_PR_HMS = "pr_EUR-44i_ECMWF-ERAINT_evaluation_r1i1p1_HMS-ALADIN52_v1_mon.nc"
EU_CORDEX_PR_MOHC = "pr_EUR-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon.nc"
CRU_31_PR = "GLOBAL_CRU_CTL_CRU-TS31_MM_0.5deg_1980-2008_pr.nc"

# Configure your evaluation here. The evaluation bounds are determined by
# the lat/lon/time values that are set here. If you set the lat/lon values
# outside of the range of the datasets' values you will get an error. Your
# start/end time values should be in 12 month intervals due to the metrics
# being used.
ref_dataset = local.load_file(CRU_31_PR, "pr")
ref_dataset.name = "cru_31_pr"
target_dataset = local.load_file(EU_CORDEX_PR_DNI, "pr")
target_dataset.name = "eu_cordex_pr_dni"
target_dataset2 = local.load_file(EU_CORDEX_PR_HMS, "pr")
target_dataset2.name = "eu_cordex_pr_hms"
target_dataset3 = local.load_file(EU_CORDEX_PR_KNMI, "pr")
target_dataset3.name = "eu_cordex_pr_knmi"
target_dataset4 = local.load_file(EU_CORDEX_PR_MOHC, "pr")
target_dataset4.name = "eu_cordex_pr_mohc"
target_datasets = [target_dataset, target_dataset2, target_dataset3, target_dataset4]
LAT_MIN = 22
LAT_MAX = 71
LON_MIN = -43
LON_MAX = 64
START = datetime.datetime(1999, 1, 1)
END = datetime.datetime(2000, 12, 1)
SEASON_MONTH_START = 1
SEASON_MONTH_END = 12

EVAL_BOUNDS = Bounds(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, START, END)

# Normalize the time values of our datasets so they fall on expected days
# of the month. For example, monthly data will be normalized so that:
# 15 Jan 2014, 15 Feb 2014 => 1 Jan 2014, 1 Feb 2014
ref_dataset = dsp.normalize_dataset_datetimes(ref_dataset, 'monthly')
target_datasets = [dsp.normalize_dataset_datetimes(target, 'monthly')
                   for target in target_datasets]

# Subset down the evaluation datasets to our selected evaluation bounds.
ref_dataset = dsp.subset(EVAL_BOUNDS, ref_dataset)
target_datasets = [dsp.subset(EVAL_BOUNDS, target)
                   for target in target_datasets]

# Do a monthly temporal rebin of the evaluation datasets.
ref_dataset = dsp.temporal_rebin(ref_dataset, datetime.timedelta(days=30))
target_datasets = [dsp.temporal_rebin(target, datetime.timedelta(days=30))
                   for target in target_datasets]

# Spatially regrid onto a 1 degree lat/lon grid within our evaluation bounds.
new_lats = np.arange(LAT_MIN, LAT_MAX, 1.0)
new_lons = np.arange(LON_MIN, LON_MAX, 1.0)
ref_dataset = dsp.spatial_regrid(ref_dataset, new_lats, new_lons)
target_datasets = [dsp.spatial_regrid(target, new_lats, new_lons)
                   for target in target_datasets]

# Load the datasets for the evaluation.
mean_bias = metrics.MeanBias()
# These versions of the metrics require seasonal bounds prior to running
# the metrics. You should set these values above in the evaluation
# configuration section.
spatial_std_dev_ratio = metrics.SeasonalSpatialStdDevRatio(month_start=SEASON_MONTH_START, month_end=SEASON_MONTH_END)
pattern_correlation = metrics.SeasonalPatternCorrelation(month_start=SEASON_MONTH_START, month_end=SEASON_MONTH_END)

# Create our example evaluation.
example_eval = evaluation.Evaluation(ref_dataset, # Reference dataset for the evaluation
                                    # 1 or more target datasets for the evaluation
                                    #[target_dataset, target_dataset2, target_dataset3, target_dataset4],
                                    target_datasets,
                                    # 1 ore more metrics to use in the evaluation
                                    [mean_bias, spatial_std_dev_ratio, pattern_correlation])
example_eval.run()

spatial_stddev_ratio = []
spatial_correlation = []
taylor_dataset_names = [] # List of datasets names for the Taylor diagram
for i, target in enumerate(example_eval.target_datasets):
    # For each target dataset in the evaluation, draw a contour map
    plotter.draw_contour_map(example_eval.results[i][0],
                             new_lats,
                             new_lons,
                             'lund_{}_{}_time_averaged_bias'.format(ref_dataset.name, target.name),
                             gridshape=(1, 1),
                             ptitle='Time Averaged Bias')

    taylor_dataset_names.append(target.name)
    # Grab Data for a Taylor Diagram
    spatial_stddev_ratio.append(example_eval.results[i][1])
    # Pattern correlation results are a tuple, so we need to index and grab
    # the component we care about.
    spatial_correlation.append(example_eval.results[i][2][0])

taylor_data = np.array([spatial_stddev_ratio, spatial_correlation]).transpose()
plotter.draw_taylor_diagram(taylor_data,
                            taylor_dataset_names,
                            ref_dataset.name,
                            fname='lund_taylor_diagram',
                            frameon=False)
