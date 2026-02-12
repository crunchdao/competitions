import os
import random
from typing import List, Tuple

import anndata
import numpy
import pandas
import scanpy
from crunch.api import Metric, Target
from crunch.scoring import ScoredMetric
from crunch.unstructured.utils import delta_message, truncate
from crunch.utils import Tracer
from tqdm import tqdm


class ParticipantVisibleError(Exception):
    """Custom exception for errors related to participant visibility."""
    pass


tracer = Tracer()


def check(
    prediction_directory_path: str,
    data_directory_path: str
):
    with tracer.log("Load prediction file"):
        prediction_path = os.path.join(prediction_directory_path, "prediction.parquet")

        if not os.path.exists(prediction_path):
            raise ParticipantVisibleError(f"Prediction file not found")

        prediction = pandas.read_parquet(prediction_path)

    with tracer.log("Check for the `Gene Name` column"):
        difference = delta_message(
            {"Gene Name"},
            set(prediction.columns),
        )

        if difference:
            raise ParticipantVisibleError(f"Columns do not match: {difference}")

    with tracer.log("Loading expected columns"):
        gene_list_path = os.path.join(data_directory_path, "Crunch3_gene_list.csv")

        expected_genes = pandas.read_csv(gene_list_path)
        expected_genes = set(expected_genes["gene_symbols"])

    with tracer.log("Ensure the DataFrame is sorted by its index before checking"):
        prediction.sort_index(inplace=True)

    with tracer.log("Check that the index runs from 1 to the total number of expected genes"):
        expected_index = numpy.arange(1, len(expected_genes) + 1)

        if not numpy.array_equal(prediction.index.values, expected_index):
            truncated = truncate(prediction.index.values)
            raise ParticipantVisibleError(f"Rank (as a DataFrame.index) must be from 1 to {len(expected_genes)}, but got [{truncated}]")

    with tracer.log("Check that the predicted genes match the expected genes exactly"):
        difference = delta_message(
            expected_genes,
            set(prediction["Gene Name"]),
        )

        if difference:
            raise ParticipantVisibleError(f"Gene names do not match: {difference}")


def score(
    prediction_directory_path: str,
    data_directory_path: str,
    target_and_metrics: List[Tuple[Target, List[Metric]]],
):
    list_size = 50

    with tracer.log("Load prediction file"):
        prediction_path = os.path.join(prediction_directory_path, "prediction.parquet")
        prediction = pandas.read_parquet(prediction_path)

    with tracer.log("Read in experimental data"):
        adata = scanpy.read_h5ad(os.path.join(data_directory_path, "UC9-norm-minimum.h5ad"))
        xenium_panel = set(adata.var.index)

    with tracer.log("Convert old gene symbol to latest symbol used in the expeirmental data"):
        gene_symbol_map_df = pandas.read_csv(os.path.join(data_directory_path, "Xenium-panel-symbol-mapping.tsv"), sep="\t", header=0, index_col=None)

        gene_symbol_map = {}
        for _, row in gene_symbol_map_df.iterrows():
            gene_symbol_map[row["Gene"]] = row["Latest Symbol"]

        gene_symbol_map_df[gene_symbol_map_df["Gene"] != gene_symbol_map_df["Latest Symbol"]]

    with tracer.log("Filter and map gene names"):
        full_gene_list = xenium_panel & {
            gene_symbol_map.get(item, item)
            for item in prediction["Gene Name"].head(list_size)
        }

    with tracer.log("Compute score"):
        n_simulation = 1000
        k = 3

        gene_panel = xenium_panel

        full_gene_list = list(full_gene_list)
        all_sim_accuracies = []

        current_list_size = min(len(full_gene_list), list_size)
        num_for_filling = 0
        gene_panel_remaining = list(gene_panel - set(full_gene_list))
        if len(full_gene_list) < list_size:
            num_for_filling = list_size - len(full_gene_list)
            if len(gene_panel_remaining) < num_for_filling:
                print('error: not enough gene from the panel to fill in')

        # for index in tqdm(range(n_simulation)):
        for index in tracer.loop(range(n_simulation), action=lambda x: f"Simulate random gene lists: {x + 1}/{n_simulation}"):
            seed = 42 + index
            randomizer = random.Random(seed)

            # Randomly sample 'current_list_size' genes from the current set
            random_genes = randomizer.sample(full_gene_list, current_list_size)

            # Randomly fill in genes from the Gene panel
            random_genes_fill = randomizer.sample(gene_panel_remaining, num_for_filling)
            random_genes = random_genes + random_genes_fill

            # Get the k accuracies for this random sample
            k_accuracies = _get_gene_list_accuracies(adata, random_genes, seed, k=k)

            # Store them (flattening the k-folds into the total list)
            all_sim_accuracies.extend(k_accuracies)

    score_mean = numpy.mean(all_sim_accuracies)
    score_median = numpy.median(all_sim_accuracies)
    score_max = max(all_sim_accuracies)

    print(f"Mean accuracy: {score_mean:.4f}")
    print(f"Median accuracy: {score_median:.4f}")
    print(f"Max accuracy: {score_max:.4f}")

    metric = target_and_metrics[0][1][0]

    return {
        metric.id: ScoredMetric(
            value=score_mean,
        ),
    }


def _get_gene_list_accuracies(
    adata: anndata.AnnData,
    gene_list: List[str],
    seed: int,
    k=3
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    # Filter genes that actually exist in the adata
    valid_genes = [g for g in gene_list if g in adata.var.index]

    # Extract data for the specific genes
    # Handle both dense and sparse matrices
    if hasattr(adata.X, "toarray"):
        X = adata[:, valid_genes].X.toarray()
    else:
        X = adata[:, valid_genes].X

    # Encode labels: 'dysplasia' vs 'non-dysplasia' -> 0 and 1
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['dysplasia_region'])

    # Initialize Logistic Regression
    # We use 'liblinear' for smaller datasets or 'lbfgs' for larger ones
    clf = LogisticRegression(max_iter=200)  # previous 100

    # Perform k-fold cross validation
    # This returns an array of k scores
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    accuracies = cross_val_score(clf, X, y, cv=kf, pre_dispatch=1, n_jobs=1)

    return accuracies
