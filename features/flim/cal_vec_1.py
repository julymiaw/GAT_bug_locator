import json
from transformers import RobertaTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

project = "jdt"
USE_TQDM = False


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


def convert_example_to_features(args):
    """Convert example to features (used in each process)."""
    js, tokenizer = args

    class Args(object):
        """A single set of features of data."""

        def __init__(self):
            self.code_length = 512
            self.nl_length = 128

    args = Args()
    # code
    code = js["code_tokens"]
    code_tokens = tokenizer.tokenize(code)[: args.code_length - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = js["docstring_tokens"]
    nl_tokens = tokenizer.tokenize(nl)[: args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_ids, nl_ids, js["url"])


print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-java")


def load_and_process_data(file_path, process_func, tokenizer):
    """Loads data from a JSONL file and processes it in parallel."""
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]

    features = []
    chunksize = 100
    with Pool(cpu_count()) as p:
        if USE_TQDM:
            iterable = tqdm(
                p.imap(
                    process_func, [(js, tokenizer) for js in data], chunksize=chunksize
                ),
                total=len(data),
                desc="Processing data",
            )
        else:
            iterable = p.map(
                process_func, [(js, tokenizer) for js in data], chunksize=chunksize
            )
        for feature in iterable:
            features.append(feature)
    return features


print("Loading and preprocessing report data...")
query_file_name = "../joblib_memmap_" + project + "_flim/report.jsonl"
nl_features = load_and_process_data(
    query_file_name, convert_example_to_features, tokenizer
)

print("Loading and preprocessing code data...")
code_file_name = "../joblib_memmap_" + project + "_flim/code.jsonl"
code_features = load_and_process_data(
    code_file_name, convert_example_to_features, tokenizer
)

# Save preprocessed features
import pickle

with open(f"../joblib_memmap_{project}_flim/nl_features.pkl", "wb") as f:
    pickle.dump(nl_features, f)
with open(f"../joblib_memmap_{project}_flim/code_features.pkl", "wb") as f:
    pickle.dump(code_features, f)

print("Preprocessing complete. Features saved.")
