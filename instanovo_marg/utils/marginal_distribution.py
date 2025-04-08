from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import re


def get_marginal_distribution(sdf, sequence_length):
    """
    Compute the marginal distribution of amino acids of sdf of a given sequence length
    """
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    sequences = sdf.to_pandas()["sequence"].tolist()

    # Dictionary: length → Counter of AA frequencies
    lengthwise_counts = defaultdict(Counter)

    for seq in sequences:
        clean_seq = re.sub(r'[^A-Za-z]', '', seq)
        filtered_seq = [aa for aa in clean_seq if aa in valid_aas]
        length = len(filtered_seq)
        if length > 0:
            lengthwise_counts[length].update(filtered_seq)

    # Now build a dictionary: length → amino acid distribution (as pandas Series)
    lengthwise_distributions = {}

    for length, aa_counter in lengthwise_counts.items():
        total = sum(aa_counter.values())
        prob_dict = {aa: count / total for aa, count in aa_counter.items()}
        
        # Optional: make sure all 20 amino acids are represented (fill missing with 0)
        for aa in valid_aas:
            prob_dict.setdefault(aa, 0.0)
        
        lengthwise_distributions[length] = pd.Series(prob_dict).sort_index()

    if sequence_length in lengthwise_distributions:
        log_probs = np.log(lengthwise_distributions[length].values)
        tiled_log_probs = np.tile(log_probs, (len(log_probs), 1))
        return log_probs
    else:
        raise ValueError(f"No sequences of length {sequence_length} found.")
