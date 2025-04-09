import re
from collections import Counter

import numpy as np
import pandas as pd


def get_marginal_distribution(sdf: Optional[SpectrumDataFrame] = None, 
                              vocab: Optional[Dict] = None, 
                              pretrain: bool = False) -> np.ndarray:
    """
        Compute the marginal amino acid distribution from an InstaNovo SpectrumDataFrame.
        Saves the result as a 27x1 numpy array of token counts.
    """
    if pretrain:
        # Define vocab (amino acids and modifications)
        vocab1 = {
            "M(ox)": "M[UNIMOD:35]",
            "M(+15.99)": "M[UNIMOD:35]",
            "S(p)": "S[UNIMOD:21]",
            "T(p)": "T[UNIMOD:21]",
            "Y(p)": "Y[UNIMOD:21]",
            "S(+79.97)": "S[UNIMOD:21]",
            "T(+79.97)": "T[UNIMOD:21]",
            "Y(+79.97)": "Y[UNIMOD:21]",
            "Q(+0.98)": "Q[UNIMOD:7]",
            "N(+0.98)": "N[UNIMOD:7]",
            "Q(+.98)": "Q[UNIMOD:7]",
            "N(+.98)": "N[UNIMOD:7]",
            "C(+57.02)": "C[UNIMOD:4]",  
            "(+42.01)": "[UNIMOD:1]", 
            "(+43.01)": "[UNIMOD:5]", 
            "(-17.03)": "[UNIMOD:385]",
        }

        # Define vocab2 (standard amino acids and special tokens)
        vocab2 = {
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
        }

        # Combine vocab and vocab2
        vocab = set(vocab2.values()) | set(vocab1.keys())

        residues_keys = {
    0: '[PAD]', 1: '[SOS]', 2: '[EOS]', 3: 'G', 4: 'A', 5: 'S', 6: 'P', 7: 'V', 8: 'T', 9: 'C',
    10: 'L', 11: 'I', 12: 'N', 13: 'D', 14: 'Q', 15: 'K', 16: 'E', 17: 'M', 18: 'H', 19: 'F',
    20: 'R', 21: 'Y', 22: 'W', 23: 'M[UNIMOD:35]', 24: 'C[UNIMOD:4]', 25: 'N[UNIMOD:7]',
    26: 'Q[UNIMOD:7]', 27: 'S[UNIMOD:21]', 28: 'T[UNIMOD:21]', 29: 'Y[UNIMOD:21]',
    30: '[UNIMOD:1]', 31: '[UNIMOD:5]', 32: '[UNIMOD:385]'
}

        # Reverse vocab for token lookup (to map from amino acid to index)
        aa_to_idx = {aa: idx for idx, aa in enumerate(vocab)}

        # Define valid amino acids (including modified amino acids)
        valid_aas = vocab

        # Regex pattern to match standard AAs and modified ones like M(+15.99)
        aa_pattern = re.compile(r"[A-Z](?:\(\+\d+(?:\.\d+)?\))?")

        # Initialize Counter to track amino acid frequencies (across the entire sequence)
        aa_counts = Counter()

        if sdf is None:
            from instanovo.utils import SpectrumDataFrame

            sdf = SpectrumDataFrame.from_huggingface(
                "InstaDeepAI/ms_ninespecies_benchmark",
                is_annotated=True,
                shuffle=False,
                split="test[:100%]", 
            )

        # Process each sequence (here, I assume you already have the sequences in `sdf`)
        sequences = sdf.to_pandas()["sequence"]

        # Process sequences and count amino acids
        for seq in sequences:
            # Extract amino acids using regex pattern

            aa_seq = [aa for aa in aa_pattern.findall(seq) if aa in valid_aas]
            aa_seq = aa_seq[:39]  # max 39 tokens to allow EOS at end
            aa_seq.append("[EOS]")

            # Pad to length 40 if needed
            while len(aa_seq) < 40:
                aa_seq.append("[PAD]")
            # Count occurrences of each amino acid in the sequence (ignoring positions)
            aa_counts.update(aa_seq)

        # Ensure all valid amino acids are included, even those with count 0
        full_counts = {aa: aa_counts.get(aa, 0) for aa in valid_aas}

        # Convert the full counts to a DataFrame
        aa_counts_df = pd.DataFrame(full_counts.items(), columns=["Amino Acid", "Count"])

        aa_counts_df["Converted Amino Acid"] = (
            aa_counts_df["Amino Acid"].map(vocab1).fillna(aa_counts_df["Amino Acid"])
        )
        aa_counts_df["Converted Amino Acid"] = pd.Categorical(
            aa_counts_df["Converted Amino Acid"], 
            categories=residues_keys, ordered=True
        )
        aa_counts_df.drop(columns=["Amino Acid"], inplace=True)

        aa_counts_df = aa_counts_df.groupby("Converted Amino Acid").sum().reset_index()

        distributions = np.zeros(len(residues_keys))
        for idx, aa_name in residues_keys.items():
            if aa_name in aa_counts_df["Converted Amino Acid"].values:
                # Get the count for this amino acid
                count = aa_counts_df[aa_counts_df["Converted Amino Acid"] == aa_name][
                    "Count"
                ].values[0]
                # Assign the count to the appropriate index in the distributions array
                distributions[idx] = count

        distributions += 1e-6
        distributions /= distributions.sum()
        np.save("instanovo_marg/configs/amino_acid_distribution.npy", np.log(distributions))
    else:
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
                split="train[:100%]",
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
        column_vector += 1e-6
        column_vector /= column_vector.sum()

        np.save("instanovo_marg/configs/amino_acid_distribution.npy", column_vector.values)
