import os
import statistics
from typing import Any, List, Tuple

import crunch
import crunch.utils
import numpy
import pandas
import scanpy
import scipy.stats
from crunch.scoring import ScoredMetric
from crunch.unstructured.utils import delta_message
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
    with tracer.log("Ensure files"):
        difference = delta_message(
            {
                PREDICTION_FILE_NAME,
                PROGRAM_PROPORTION_FILE_NAME,
            },
            set(os.listdir(prediction_directory_path)),
        )

        if difference:
            raise ParticipantVisibleError(f"Prediction files are not valid: {difference}")

    with tracer.log("Load ground truth: TF150"):
        gtruth_adata = scanpy.read_h5ad(os.path.join(data_directory_path, "tf150_gtruth.h5ad"))

    with tracer.log("Load prediction: prediction"):
        prediction = scanpy.read_h5ad(os.path.join(prediction_directory_path, PREDICTION_FILE_NAME))

        with tracer.log("Validating .var_names"):
            expected_column_count = gtruth_adata.uns["high_var_gene_mask"].sum()

            if len(prediction.var_names) != expected_column_count:
                raise ParticipantVisibleError("There is an invalid number of columns (`.var_names`). Perhaps the wrong ones were predicted?")

        with tracer.log("Validating .obs"):
            with tracer.log("Validating columns"):
                difference = delta_message(
                    {"gene"},
                    set(prediction.obs.columns),
                )

                if difference:
                    raise ParticipantVisibleError(f"Invalid column in .obs: {difference}")

            with tracer.log("Validating lines"):
                expected_genes = set(gtruth_adata.obs["gene"].unique())
                got_genes = set(prediction.obs["gene"].unique())

                print("expected", sorted(expected_genes))
                print("got     ", sorted(got_genes))

                if expected_genes != got_genes:
                    raise ParticipantVisibleError("There is an invalid number of genes (`.obs`). Perhaps the wrong ones were predicted?")

                line_count_per_gene = prediction.obs.value_counts()\
                    .reset_index()\
                    .set_index("gene", drop=True)["count"]\
                    .to_dict()

                print("line_count_per_gene", line_count_per_gene)

                for gene in expected_genes:
                    if line_count_per_gene[gene] != 100:
                        raise ParticipantVisibleError("There is an invalid number of lines (`.obs`) for a gene. Perhaps the wrong ones were predicted?")

    with tracer.log("Validating predict_program_proportion"):
        data_columns = ["adipo", "pre_adipo", "lipo", "other"]

        with tracer.log("Load prediction"):
            proportions = pandas.read_csv(os.path.join(prediction_directory_path, PROGRAM_PROPORTION_FILE_NAME))

        with tracer.log("Load expected genes"):
            expected_genes = _load_predict_perturbations(data_directory_path)

        with tracer.log("Check for required columns"):
            difference = delta_message(
                {"gene"} | set(data_columns),
                set(proportions.columns),
            )

            if difference:
                raise ParticipantVisibleError(f"predict_program_proportion's columns do not match: {difference}")

        with tracer.log("Check for rows"):
            expected_genes_count = len(expected_genes)
            got_rows = len(proportions)

            if expected_genes_count != got_rows:
                raise ParticipantVisibleError(f"predict_program_proportion: Row count is incorrect, expected {expected_genes_count} but got {got_rows}")

        with tracer.log("Check for NaN values"):
            if proportions.isnull().values.any():
                raise ParticipantVisibleError(f"predict_program_proportion: Found NaN value(s)")

        with tracer.log("Check for infinity values"):
            proportions = proportions.replace([-numpy.inf, numpy.inf], numpy.nan)

            if proportions.isnull().values.any():
                raise ParticipantVisibleError(f"predict_program_proportion: Found Inf value(s)")

        with tracer.log("Check for genes"):
            difference = delta_message(
                set(expected_genes),
                set(proportions["gene"].unique()),
            )

            if difference:
                raise ParticipantVisibleError(f"predict_program_proportion: Genes do not match: {difference}")

        for column_name in tracer.loop(data_columns, lambda x: f"Check data types in the '{x}' values"):
            if not pandas.api.types.is_numeric_dtype(proportions[column_name].values):
                raise ParticipantVisibleError(f"predict_program_proportion: Found non-numeric values for column `{column_name}`")


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
        gtruth_adata = scanpy.read_h5ad(os.path.join(data_directory_path, "tf150_gtruth.h5ad"))

    with tracer.log("Load ground truth: predict_program_proportion"):
        ground_truth_proportion = pandas.read_csv(os.path.join(data_directory_path, "program_proportion_gtruth.csv"))

    with tracer.log("Load prediction: prediction"):
        prediction_adata = scanpy.read_h5ad(os.path.join(prediction_directory_path, PREDICTION_FILE_NAME))

    with tracer.log("Load prediction: predict_program_proportion"):
        pred_proportion = pandas.read_csv(os.path.join(prediction_directory_path, PROGRAM_PROPORTION_FILE_NAME))

    with tracer.log("Extract unstructured annotation"):
        hvg_mask = gtruth_adata.uns["high_var_gene_mask"]
        perturbed_centroid = gtruth_adata.uns["perturbed_centroid_train"][hvg_mask]

    person_values = []
    mmd_values = []

    perturbations = gtruth_adata.obs["gene"].cat.categories.tolist()
    for perturbation in tracer.loop(perturbations, "Scoring gene: {value}"):

        with tracer.log("Filter the slice"):
            hvg_mask = gtruth_adata.uns["high_var_gene_mask"]
            gtruth_mask = gtruth_adata.obs["gene"] == perturbation
            prediction_mask = prediction_adata.obs["gene"] == perturbation

            gtruth_X = _as_numpy_array(gtruth_adata[gtruth_mask, hvg_mask].X)
            pred_X = _as_numpy_array(prediction_adata[prediction_mask].X)

        with tracer.log("Compute pearson(X)"):
            person_value = _pearson(
                gtruth_X,
                pred_X,
                perturbed_centroid,
            )

            person_values.append(person_value)

        with tracer.log("Compute MMD(X)"):
            mmd_value = _mmd(
                gtruth_X,
                pred_X,
            )

            mmd_values.append(mmd_value)

    with tracer.log("Compute averages"):
        person_value = statistics.mean(person_values)
        mmd_value = statistics.mean(mmd_values)

    with tracer.log("Compute L1_DISTANCE(X)"):
        l1_distance_value = _l1_distance(
            ground_truth_proportion,
            pred_proportion,
            gene_column_name="gene",
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
    ground_truth_X: NDArray[numpy.float64],
    prediction_X: NDArray[numpy.float64],
    perturbed_centroid: NDArray[numpy.float64],
) -> float:
    ground_truth_X_target = ground_truth_X.mean(axis=0)
    prediction_X_target = prediction_X.mean(axis=0)

    return scipy.stats.pearsonr(
        ground_truth_X_target - perturbed_centroid,
        prediction_X_target - perturbed_centroid,
    ).statistic


def _mmd(
    ground_truth_X: NDArray[numpy.float64],
    prediction_X: NDArray[numpy.float64],
) -> float:
    def _gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
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

    def _compute_mmd_batch(X_batch, Y_batch, kernel_mul, kernel_num, fix_sigma, kernel_func):
        num_batch_element = X_batch.shape[0]

        kernels = kernel_func(X_batch, Y_batch, kernel_mul, kernel_num, fix_sigma)
        XX = kernels[:num_batch_element, :num_batch_element]
        YY = kernels[num_batch_element:, num_batch_element:]
        XY = kernels[:num_batch_element, num_batch_element:]
        YX = kernels[num_batch_element:, :num_batch_element]
        mmd_val = numpy.sum(XX + YY - XY - YX)

        return mmd_val, num_batch_element ** 2

    def balance_source_target_sample_per_perturbation(gtruth_X_tgt, pred_X_tgt):
        num_gtruth = gtruth_X_tgt.shape[0]
        num_pred = pred_X_tgt.shape[0]
        min_sample = min(num_gtruth, num_pred)

        return gtruth_X_tgt[0:min_sample, :], pred_X_tgt[0:min_sample, :]

    kernel_mul = 2.0
    kernel_num = 5
    fix_sigma = 2326
    batch_size = 100

    # Balancing the samples to compute mmd using equal number of samples
    gtruth_X, pred_X = balance_source_target_sample_per_perturbation(ground_truth_X, prediction_X)

    # Sharding the X into smaller batches
    num_batches = gtruth_X.shape[0] // batch_size if gtruth_X.shape[0] % batch_size == 0 else (gtruth_X.shape[0] // batch_size) + 1

    # Do not compute MMD if only one sample
    if gtruth_X[(num_batches - 1) * batch_size:num_batches * batch_size, :].shape[0] < 2:
        num_batches = num_batches - 1

    results = [
        _compute_mmd_batch(
            gtruth_X[bidx * batch_size:(bidx + 1) * batch_size, :],
            pred_X[bidx * batch_size:(bidx + 1) * batch_size, :],
            kernel_mul,
            kernel_num,
            fix_sigma,
            _gaussian_kernel
        )
        for bidx in tracer.loop(range(num_batches), lambda bidx: f"Processing batch {bidx + 1} of {num_batches}")
    ]

    mmd_dist_sum = sum(r[0] for r in results)
    num_element = sum(r[1] for r in results)

    mmd_dist = mmd_dist_sum / num_element
    return mmd_dist


def _l1_distance(
    ground_truth: pandas.DataFrame,
    prediction: pandas.DataFrame,
    gene_column_name: str,
) -> float:
    unique_perturb_genes = list(ground_truth[gene_column_name].unique())

    all_l1_loss_list = []
    for gene in unique_perturb_genes:
        ground_truth_at_gene = ground_truth[ground_truth[gene_column_name] == gene]
        prediction_at_gene = prediction[prediction[gene_column_name] == gene]

        assert ground_truth_at_gene.shape[0] == 1 and prediction_at_gene.shape[0] == 1, "More than one prediction for state"

        # Getting the L1 loss for main  pre, adipo and other
        l1_three = (
            numpy.abs(ground_truth_at_gene.iloc[0]["pre_adipo"] - prediction_at_gene.iloc[0]["pre_adipo"]) +
            numpy.abs(ground_truth_at_gene.iloc[0]["adipo"] - prediction_at_gene.iloc[0]["adipo"]) +
            numpy.abs(ground_truth_at_gene.iloc[0]["other"] - prediction_at_gene.iloc[0]["other"])
        )

        # Getting the L1 loss for lipo by adipo
        numerical_stab_term = 1e-20
        pred_lipo_adipo = prediction_at_gene.iloc[0]["lipo"] / (prediction_at_gene.iloc[0]["adipo"] + numerical_stab_term)
        true_lipo_adipo = ground_truth_at_gene.iloc[0]["lipo"] / (ground_truth_at_gene.iloc[0]["adipo"] + numerical_stab_term)
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


def _load_predict_perturbations(
    data_directory_path: str,
) -> List[str]:
    genes = []

    with open(os.path.join(data_directory_path, "predict_perturbations.txt"), "r") as fd:
        for line in fd.readlines():
            line = line.strip()

            if line:
                genes.append(line)

    return genes
