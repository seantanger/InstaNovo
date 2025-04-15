from instanovo.utils import SpectrumDataFrame

sdf_train = SpectrumDataFrame.from_huggingface(
    "InstaDeepAI/ms_proteometools",
    is_annotated=True,
    shuffle=False,
    split="train",  # Let's only use a subset of the test data for faster inference in this notebook
)


sdf_train.write_ipc("data/ms_proteometools/train.ipc")


# Validation
sdf_valid = SpectrumDataFrame.from_huggingface(
    "InstaDeepAI/ms_proteometools",
    is_annotated=True,
    shuffle=False,
    split="validation",  # Let's only use a subset of the test data for faster inference in this notebook
)
sdf_valid.write_ipc("data/ms_proteometools/valid.ipc")

# Test
sdf_test = SpectrumDataFrame.from_huggingface(
    "InstaDeepAI/ms_proteometools",
    is_annotated=True,
    shuffle=False,
    split="test[:20%]",  # Let's only use a subset of the test data for faster inference in this notebook
)
sdf_test.write_ipc("data/ms_proteometools/test.ipc")