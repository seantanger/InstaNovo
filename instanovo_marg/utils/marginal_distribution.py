import re
from collections import Counter

import numpy as np
import pandas as pd


def get_marginal_distribution(sdf=None):
    # Define vocab (you already provided this)
    vocab = {
        0: "[PAD]",
        1: "[SOS]",
        2: "[EOS]",
        3: "G",
        4: "A",
        5: "S",
        6: "P",
        7: "V",
        8: "T",
        9: "C",
        10: "L",
        11: "I",
        12: "N",
        13: "D",
        14: "Q",
        15: "K",
        16: "E",
        17: "M",
        18: "H",
        19: "F",
        20: "R",
        21: "Y",
        22: "W",
        23: "M(+15.99)",
        24: "C(+57.02)",
        25: "N(+.98)",
        26: "Q(+.98)",
    }

    # Reverse vocab for token lookup
    aa_to_idx = {aa: idx for idx, aa in vocab.items()}

    # Valid amino acids (must match vocab, excluding SOS)
    valid_aas = set(vocab.values()) - {"[PAD]", "[SOS]", "[EOS]"}

    # Regex to match standard AAs and modified ones like M(+15.99)
    aa_pattern = re.compile(r"[A-Z](?:\(\+\d+(?:\.\d+)?\))?")

    # Initialize counts: position (0-39) → Counter of amino acid indices
    positional_counts = [Counter() for _ in range(40)]

    # Process each sequence
    if sdf is None:
        from instanovo.utils import SpectrumDataFrame

        sdf = SpectrumDataFrame.from_huggingface(
            "InstaDeepAI/ms_ninespecies_benchmark",
            is_annotated=True,
            shuffle=False,
            split="test[:10%]",  
        )
    sequences = sdf.to_pandas()["sequence"].tolist()

    for seq in sequences:
        # Extract amino acids using regex pattern
        aa_seq = [aa for aa in aa_pattern.findall(seq) if aa in valid_aas]

        # Append EOS if it fits, truncate if longer than 39 before EOS
        aa_seq = aa_seq[:39]  # max 39 tokens to allow EOS at end
        aa_seq.append("[EOS]")

        # Pad to length 40 if needed
        while len(aa_seq) < 40:
            aa_seq.append("[PAD]")

        # Count each token at each position
        for i, aa in enumerate(aa_seq):
            if aa in aa_to_idx:
                idx = aa_to_idx[aa]
                positional_counts[i][idx] += 1

    # Now convert to a DataFrame: each row = position, each column = vocab index
    positional_df = pd.DataFrame(positional_counts).fillna(0).astype(int)
    positional_df.index.name = "position"

    # Ensure all indices from 0 to 26 are present (fill missing with 0)
    full_index = pd.RangeIndex(0, 27)
    data_filled = positional_df.sum(axis=0).reindex(full_index, fill_value=0)

    # Convert to column vector
    column_vector = data_filled.to_frame(name="count")  # shape (27, 1)

    np.save("instanovo_marg/configs/amino_acid_distribution.npy", column_vector.values)
