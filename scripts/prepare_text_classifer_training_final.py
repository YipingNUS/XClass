""" Select the most confident examples for each category using a distance threshold.
    The (raw) data of selected documents and labels is written to a separate folder.
"""

import argparse
from pathlib import Path
import pandas as pd


def load_dataset(input_path):
    with open(input_path/"classes.txt", "r") as f:
        classes = [c.strip() for c in f.readlines()]
    id2label = {str(id):label for id, label in enumerate(classes)}
    print(id2label)

    with open(input_path/"dataset.txt", "r") as f:
        texts = [str(t.strip()) for t in f.readlines()]

    with open(input_path/"labels.txt", "r") as f:
        labels = [id2label[l.strip()] for l in f.readlines()]

    assert len(texts) == len(labels)
    return texts, labels


def main(dataset_name, suffix, confidence_threshold, input_path, output_path, output_file):
    input_col = "preprocessed_text"
    if dataset_name == "sbic" or dataset_name == "sbic-fine-grained":
        target_col = "targetMinority"
        raw_col = "post"
        p_out = Path(output_path)/dataset_name
    elif dataset_name == "waseem":
        target_col = "label"
        raw_col = "text"
        p_out = Path(output_path)/"waseem_dataset/share"
    elif dataset_name == "waseem-sbic-cross-domain":
        target_col = "label"
        raw_col = "text"
        p_out = Path(output_path)/dataset_name
    elif dataset_name == "sbic-waseem-cross-domain":
        target_col = "label"
        raw_col = "text"
        p_out = Path(output_path)/dataset_name
    else:
        raise Exception(f"Unknown dataset: {dataset_name}.")

    input_path = Path(input_path)/f"{dataset_name}_{suffix}.{confidence_threshold}"
    texts, labels = load_dataset(input_path=input_path)
    df = pd.DataFrame({raw_col: ["dummy"]*len(texts), input_col: texts, target_col: labels})
    df.to_csv(str(p_out/output_file), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--input_path", type=str, default="/Users/admin/hate_speech/XClass/data/datasets")
    parser.add_argument("--suffix", type=str, default="pca64.clusgmm.bbu-12.mixture-100.42")
    parser.add_argument("--confidence_threshold", default=0.5)
    parser.add_argument("--output_path", type=str, default="/Users/admin/hate_speech/datasets")
    parser.add_argument("--output_file", type=str, default="pseudo.trn.csv")
    args = parser.parse_args()
    print(vars(args))
    main(args.dataset_name, args.suffix, args.confidence_threshold, args.input_path, args.output_path, args.output_file)
