from instanovo.utils import SpectrumDataFrame

sdf_train = SpectrumDataFrame.from_huggingface(
    "InstaDeepAI/ms_ninespecies_benchmark",
    is_annotated=True,
    shuffle=False,
    split="train",  # Let's only use a subset of the test data for faster inference in this notebook
)


sdf_train.write_ipc("data/new_schema/train.ipc")


# Validation
sdf_valid = SpectrumDataFrame.from_huggingface(
    "InstaDeepAI/ms_ninespecies_benchmark",
    is_annotated=True,
    shuffle=False,
    split="validation",  # Let's only use a subset of the test data for faster inference in this notebook
)
sdf_valid.write_ipc("data/new_schema/valid.ipc")

# Test
# Validation
sdf_test = SpectrumDataFrame.from_huggingface(
    "InstaDeepAI/ms_ninespecies_benchmark",
    is_annotated=True,
    shuffle=False,
    split="test",  # Let's only use a subset of the test data for faster inference in this notebook
)
sdf_test.write_ipc("data/new_schema/test.ipc")