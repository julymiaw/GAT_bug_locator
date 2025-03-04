import csv
import torch


def generate_class_id_mapping(csv_file, output_path):
    class_names = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["className"])

    class_names = sorted(list(set(class_names)))
    class_idx_dict = {name: idx for idx, name in enumerate(class_names)}
    torch.save(class_idx_dict, output_path)


def main():
    csv_file = "class_info.csv"
    output_path = "model/dataset/class_idx_dict.pt"
    generate_class_id_mapping(csv_file, output_path)


if __name__ == "__main__":
    main()
