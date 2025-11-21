import sys
import os
import json
import numpy as np
from datasets import load_dataset as hg_load_dataset
import ast

project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_query_local(args, version="", verbose=False):
    query_data = {}

    # split_config_file = 'default_splits/{}.txt'.format(args.splits)

    split_config_file = os.path.join(
        project_root_path,
        "chinatravel",
        "evaluation",
        "default_splits",
        "{}.txt".format(args.splits),
    )

    print("config file for testing split: {}".format(split_config_file))

    query_id_list = []
    with open(split_config_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            query_id_list.append(line)

    if verbose:
        print(query_id_list)

    data_dir = os.path.join(project_root_path, "chinatravel", "data")

    dir_list = os.listdir(data_dir)
    for dir_i in dir_list:
        dir_ii = os.path.join(data_dir, dir_i)
        if os.path.isdir(dir_ii):
            file_list = os.listdir(dir_ii)

            for file_i in file_list:
                query_id = file_i.split(".")[0]
                if query_id in query_id_list:
                    data_i = json.load(
                        open(os.path.join(dir_ii, file_i), encoding="utf-8")
                    )

                    if hasattr(args, 'oracle_translation') and not args.oracle_translation:
                        if "hard_logic" in data_i:
                            del data_i["hard_logic"]
                        if "hard_logic_py" in data_i:
                            del data_i["hard_logic_py"]
                        if "hard_logic_nl" in data_i:
                            del data_i["hard_logic_nl"]

                    query_data[query_id] = data_i

    # print(query_data)

    if verbose:
        for query_id in query_id_list:
            print(query_id, query_data[query_id])

    return query_id_list, query_data


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(json_data, file_path):
    # Create directory if it doesn't exist
    import os
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w", encoding="utf8") as dump_f:
        json.dump(json_data, dump_f, ensure_ascii=False, indent=4, cls=NpEncoder)



def load_query(args):
    # Parse split name to check if it includes slicing (e.g., easy_100, easy_200, easy_300)
    split_parts = args.splits.split('_')
    base_split = split_parts[0]  # e.g., "easy"
    slice_index = None

    if len(split_parts) == 2 and split_parts[1].isdigit():
        slice_index = int(split_parts[1])  # e.g., 100, 200, 300

    # Special handling for human1000 (from test config - minimal data)
    if args.splits == "human1000":
        config_name = "test"
        actual_split = "human1000"
        slice_index = None
        is_minimal_data = True  # Flag to indicate minimal data format
    # Check if base_split is a standard split
    elif base_split not in ["easy", "medium", "human", "preference",
                          "preference0", "preference1", "preference2",
                          "preference3", "preference4", "preference5"]:
        # If not standard, try loading from local
        if not args.splits in ["preference_base50",
                               "preference0_base50", "preference1_base50", "preference2_base50",
                               "preference3_base50", "preference4_base50", "preference5_base50"]:
            return load_query_local(args)
        is_minimal_data = False
    else:
        is_minimal_data = False
        # Determine config name
        config_name = "default"
        actual_split = base_split  # The actual split name to load from HuggingFace

        if base_split.startswith("preference"):
            config_name = "preference"
            if base_split != "preference":
                actual_split = base_split  # preference0, preference1, etc.

        # Special handling for preference_base50 format
        if args.splits in ["preference_base50", "preference0_base50", "preference1_base50",
                           "preference2_base50", "preference3_base50", "preference4_base50",
                           "preference5_base50"]:
            config_name = "preference"
            actual_split = args.splits
            slice_index = None  # Don't slice these

    # Load data from HuggingFace
    query_data = hg_load_dataset("LAMDA-NeSy/ChinaTravel", name=config_name)[actual_split].to_list()

    # Apply slicing if specified
    if slice_index is not None:
        total_samples = len(query_data)
        if slice_index == 100:
            # First 100 samples (0-99)
            query_data = query_data[:100]
            print(f"[Data Slicing] Loading first 100 samples (0-99) from {base_split}")
        elif slice_index == 200:
            # Middle 100 samples (100-199)
            query_data = query_data[100:200]
            print(f"[Data Slicing] Loading middle 100 samples (100-199) from {base_split}")
        elif slice_index == 300:
            # Last 100 samples (200-299)
            query_data = query_data[200:300]
            print(f"[Data Slicing] Loading last 100 samples (200-299) from {base_split}")
        else:
            print(f"[Warning] Unrecognized slice index: {slice_index}. Loading full dataset.")

    for data_i in query_data:
        if "hard_logic_py" in data_i:
            data_i["hard_logic_py"] = ast.literal_eval(data_i["hard_logic_py"])

    query_id_list = [data_i["uid"] for data_i in query_data]
    data_dict = {}
    for data_i in query_data:
        # For minimal data format (only uid and nature_language),
        # agent will extract all information from natural language
        if is_minimal_data:
            # Ensure minimal required fields exist
            if "nature_language" not in data_i:
                raise ValueError(f"Minimal data must contain 'nature_language' field. Got: {list(data_i.keys())}")
            # Set oracle_translation to False for minimal data to force NL2SL translation
            # Don't include hard_logic fields since we don't have them
        else:
            # For full data format, handle oracle_translation flag
            if not args.oracle_translation:
                if "hard_logic" in data_i:
                    del data_i["hard_logic"]
                if "hard_logic_py" in data_i:
                    del data_i["hard_logic_py"]
                if "hard_logic_nl" in data_i:
                    del data_i["hard_logic_nl"]

        data_dict[data_i["uid"]] = data_i

    return query_id_list, data_dict


import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--splits", type=str, default="easy")

if __name__ == "__main__":


    # from datasets import load_dataset as hg_load_dataset

    # # Login using e.g. `huggingface-cli login` to access this dataset
    # ds = hg_load_dataset("LAMDA-NeSy/ChinaTravel")
    # print(ds)
    # print(ds["easy"].to_list())

    # exit(0)
    args = argparser.parse_args()
    query_id_list, query_data = load_query(args)
    # print(query_id_list)
    # print(query_data)

    for uid in query_id_list:
        if uid in query_data:
            print(uid, query_data[uid])
        else:
            raise ValueError(f"{uid} not in query_data")
    