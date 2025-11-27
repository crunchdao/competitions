import os
from typing import Any, List, Tuple

import crunch
import crunch.utils
import numpy
import pandas
import scanpy
import scipy.stats
from crunch.scoring import ScoredMetric
from numpy.typing import NDArray

PREDICTION_FILE_NAME = "prediction.h5ad"
PROGRAM_PROPORTION_FILE_NAME = "predict_program_proportion.csv"


class ParticipantVisibleError(Exception):
    """unstructured exception for errors related to participant visibility."""
    pass


tracer = crunch.utils.Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str,
):
    pass


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    phase_type: crunch.api.PhaseType,
    target_and_metrics: List[Tuple[crunch.api.Target, List[crunch.api.Metric]]],
):
    pearson_metric = _find_metric_by_name(target_and_metrics, "pearson-delta")
    mmd_metric = _find_metric_by_name(target_and_metrics, "mmd")
    l1_distance_metric = _find_metric_by_name(target_and_metrics, "l1-distance")

    with tracer.log("Load ground truth: TF150"):
        gtruth_adata = scanpy.read_h5ad(os.path.join(data_directory_path, "150_adata_with_labels_valid.h5ad"))

    with tracer.log("Load ground truth: predict_program_proportion"):
        gtruth_proportion = pandas.read_csv(os.path.join(data_directory_path, "150_adata_with_labels_state_prop_valid.csv"))

    with tracer.log("Load prediction: prediction"):
        pred_adata = scanpy.read_h5ad(os.path.join(prediction_directory_path, PREDICTION_FILE_NAME))

        with tracer.log("Convert to array"):
            gtruth_X = _as_numpy_array(gtruth_adata.X)
            pred_X = _as_numpy_array(pred_adata.X)

    with tracer.log("Load prediction: predict_program_proportion"):
        pred_proportion = pandas.read_csv(os.path.join(prediction_directory_path, PROGRAM_PROPORTION_FILE_NAME))

    with tracer.log("Extract perturbed centroid"):
        perturbed_centroid = gtruth_adata.uns["perturbed_centroid_train"]

    del gtruth_adata
    del pred_adata

    with tracer.log("Compute pearson(X)"):
        person_value = _pearson(
            gtruth_X,
            pred_X,
            perturbed_centroid,
        )

    with tracer.log("Compute L1_DISTANCE(X)"):
        l1_distance_value = _l1_distance(
            gtruth_proportion,
            pred_proportion,
            gene_col_name="gene",
        )

    with tracer.log("Compute MMD(X)"):
        # mmd_value = 9999.99
        mmd_value = _mmd(
            gtruth_X,
            pred_X,
        )

    print("person(X): {}".format(person_value))
    print("mmd(X): {}".format(mmd_value))
    print("l1_distance(X): {}".format(l1_distance_value))

    return {
        pearson_metric.id: ScoredMetric(
            value=person_value,
        ),
        mmd_metric.id: ScoredMetric(
            value=mmd_value,
        ),
        l1_distance_metric.id: ScoredMetric(
            value=l1_distance_value,
        ),
    }


def _pearson(
    gtruth_X: NDArray[numpy.float64],
    pred_X: NDArray[numpy.float64],
    perturbed_centroid: NDArray[numpy.float64],
) -> float:
    gtruth_X_target = gtruth_X.mean(axis=0)
    pred_X_target = pred_X.mean(axis=0)

    return scipy.stats.pearsonr(
        gtruth_X_target - perturbed_centroid,
        pred_X_target - perturbed_centroid,
    ).statistic


def _mmd(
    gtruth_X: NDArray[numpy.float64],
    pred_X: NDArray[numpy.float64],
) -> float:
    def compute_mmd_batch(X_batch, Y_batch, kernel_mul, kernel_num, fix_sigma, kernel_func):
        num_batch_element = X_batch.shape[0]

        kernels = kernel_func(X_batch, Y_batch, kernel_mul, kernel_num, fix_sigma)
        XX = kernels[:num_batch_element, :num_batch_element]
        YY = kernels[num_batch_element:, num_batch_element:]
        XY = kernels[:num_batch_element, num_batch_element:]
        YX = kernels[num_batch_element:, :num_batch_element]
        mmd_val = numpy.sum(XX + YY - XY - YX)

        return mmd_val, num_batch_element ** 2

    def gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
        '''
        '''
        # Getting the L2 distance
        n_samples = int(source.shape[0]) + int(target.shape[0])
        total = numpy.concatenate([source, target], axis=0)

        total0 = numpy.broadcast_to(numpy.expand_dims(total, axis=0), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        total1 = numpy.broadcast_to(numpy.expand_dims(total, axis=1), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        L2_distance = numpy.sum((total0 - total1)**2, axis=2)

        # Now we are ready to scale this using multiple bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = numpy.sum(L2_distance) / (n_samples**2 - n_samples)
        # Now we will create the multiple bandwidth list
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [numpy.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    kernel_mul = 2.0
    kernel_num = 5
    fix_sigma = None

    num_source = gtruth_X.shape[0]
    num_batches = 30 * 5  # (30 for 20k samples)
    batch_size = num_source // num_batches

    results = [
        compute_mmd_batch(
            gtruth_X[bidx * batch_size:(bidx + 1) * batch_size, :],
            pred_X[bidx * batch_size:(bidx + 1) * batch_size, :],
            kernel_mul, kernel_num, fix_sigma, gaussian_kernel
        )
        for bidx in tracer.loop(range(num_batches), lambda bidx: f"Processing batch {bidx} of {num_batches}")
    ]

    mmd_dist_sum = sum(r[0] for r in results)
    num_element = sum(r[1] for r in results)

    mmd_dist = mmd_dist_sum / num_element
    return mmd_dist


def _l1_distance(
    true_state_proportion_df: pandas.DataFrame,
    pred_state_proprotion_df: pandas.DataFrame,
    gene_col_name: str,
) -> float:
    # Reading the prediction proportion filename
    # Going over all the genes that were perturbed in this set
    unique_perturb_genes = list(true_state_proportion_df[gene_col_name].unique())

    all_l1_loss_list = []
    for gene in unique_perturb_genes:
        # Slicing the column with this gene
        true_gene_df = true_state_proportion_df[true_state_proportion_df[gene_col_name] == gene]
        pred_gene_df = pred_state_proprotion_df[pred_state_proprotion_df[gene_col_name] == gene]

        assert true_gene_df.shape[0] == 1 and pred_gene_df.shape[0] == 1, "More than one prediction for state"

        # Getting the L1 loss for main  pre, adipo and other
        l1_three = (
            numpy.abs(true_gene_df.iloc[0]["pre_adipo"] - pred_gene_df.iloc[0]["pre_adipo"]) +
            numpy.abs(true_gene_df.iloc[0]["adipo"] - pred_gene_df.iloc[0]["adipo"]) +
            numpy.abs(true_gene_df.iloc[0]["other"] - pred_gene_df.iloc[0]["other"])
        )

        # Getting the L1 loss for lipo by adipo
        numerical_stab_term = 1e-20
        pred_lipo_adipo = pred_gene_df.iloc[0]["lipo"] / (pred_gene_df.iloc[0]["adipo"] + numerical_stab_term)
        true_lipo_adipo = true_gene_df.iloc[0]["lipo"] / (true_gene_df.iloc[0]["adipo"] + numerical_stab_term)
        l1_lipo_adipo = numpy.abs(true_lipo_adipo - pred_lipo_adipo)

        # Getting the average error
        average_l1 = 0.75 * l1_three + 0.25 * l1_lipo_adipo
        all_l1_loss_list.append(average_l1)

    # Getting the overall average over all the gene perturbation
    l1_loss = numpy.mean(all_l1_loss_list)
    return float(l1_loss)


def _as_numpy_array(x: Any) -> NDArray[numpy.float64]:
    if not isinstance(x, numpy.ndarray):
        x = x.toarray()

    return x


def _find_metric_by_name(
    target_and_metrics: List[Tuple[crunch.api.Target, List[crunch.api.Metric]]],
    name: str
) -> crunch.api.Metric:
    for target, metrics in target_and_metrics:
        if target.name != name:
            continue

        for metric in metrics:
            if metric.name == name:
                return metric

    raise ValueError(f"metric {name} not found")
