import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel
from multiprocessing import Pool, cpu_count
import pickle

project = "jdt"


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        code_ids,
        nl_ids,
        url,
    ):
        self.code_ids = code_ids
        self.nl_ids = nl_ids
        self.url = url


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, inputs):
        return self.encoder(inputs, attention_mask=inputs.ne(1))[1]


print("Loading model...")
model = RobertaModel.from_pretrained("neulab/codebert-java")
model = Model(model)
model.to(torch.device("cpu"))
model.eval()


def compute_nl_vec(feature):
    """Computes the NL vector for a single feature."""

    inputs = torch.tensor([feature.nl_ids]).to(torch.device("cpu"))
    with torch.no_grad():
        vec = model(inputs).cpu().numpy()
    return vec, feature.url


def compute_code_vec(feature):
    """Computes the code vector for a single feature."""

    inputs = torch.tensor([feature.code_ids]).to(torch.device("cpu"))
    with torch.no_grad():
        vec = model(inputs).cpu().numpy()
    del inputs
    return vec, feature.url


with open(f"../joblib_memmap_{project}_flim/nl_features.pkl", "rb") as f:
    nl_features = pickle.load(f)

print("Computing report features...")
nl_vecs_with_url = []
with Pool(cpu_count()) as p:
    nl_vecs_with_url = p.map(compute_nl_vec, nl_features)

nl_vecs = [item[0] for item in nl_vecs_with_url]
nl_urls = [item[1] for item in nl_vecs_with_url]
nl_vecs = np.concatenate(nl_vecs, axis=0)

np.save(f"../joblib_memmap_{project}_flim/nl_vecs.npy", nl_vecs)
np.save(f"../joblib_memmap_{project}_flim/nl_urls.npy", nl_urls)

with open(f"../joblib_memmap_{project}_flim/code_features.pkl", "rb") as f:
    code_features = pickle.load(f)

print("Computing code features...")
with Pool(cpu_count()) as p:
    code_vecs_with_url = p.map(compute_code_vec, code_features)

code_vecs = [item[0] for item in code_vecs_with_url]
code_urls = [item[1] for item in code_vecs_with_url]
code_vecs = np.concatenate(code_vecs, axis=0)


np.save(f"../joblib_memmap_{project}_flim/code_vecs.npy", code_vecs)
np.save(f"../joblib_memmap_{project}_flim/code_urls.npy", code_urls)

print("Feature computation and saving complete.")
