import pandas as pd
from pprint import pprint
import random

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import normalize_answer

PARQUET_PATH = "~/data/searchR1_processed_direct/test.parquet"


def inspect_parquet(path, check_frame: bool = False, count_data_source: bool = False, sample_n_check_gt: int = None):
    df = pd.read_parquet(path)
    if check_frame:
        print(f"Total samples in parquet: {len(df)}")
        sample_dict = df.iloc[0].to_dict()
        pprint(sample_dict)

    if count_data_source:
        print("Data source counts:")
        print(df["data_source"].value_counts())
        print("Data source proportions:")
        print(df["data_source"].value_counts(normalize=True))

    if sample_n_check_gt is not None:
        sample_n = min(sample_n_check_gt, len(df))
        sample_indices = random.sample(range(len(df)), sample_n)
        for idx in sample_indices:
            gt = df.iloc[idx]["env_kwargs"]["ground_truth"]["target"][0]
            ngt = normalize_answer(gt)
            print(f"Index: {idx:<6}, Ground Truth: {gt:<40} Normalized Ground Truth: {ngt}")



if __name__ == "__main__":
    inspect_parquet(PARQUET_PATH, check_frame=False, count_data_source=False, sample_n_check_gt=None)