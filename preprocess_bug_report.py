import json
import csv
import torch

class_idx_dict = torch.load("model/dataset/class_idx_dict.pt", weights_only=True)


def filter_bug_reports(file_path, class_names):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    total_errors = len(data)
    missing_description_count = 0
    removed_reports_count = 0
    filtered_data = {}
    bug_class_dict = {}

    for bug_id, bug_data in data.items():
        description = bug_data["bug_report"]["description"]
        if not description:
            missing_description_count += 1
            continue

        valid_report = True
        updated_result_names = []
        updated_result_indexes = []
        for class_name in bug_data["bug_report"]["result"]:
            if class_name.endswith(".java"):
                class_name = class_name[:-5]
            if class_name not in class_names:
                valid_report = False
                break
            updated_result_names.append(class_name)
            updated_result_indexes.append(class_idx_dict[class_name])

        if valid_report:
            bug_data["bug_report"]["result"] = updated_result_names
            filtered_data[bug_id] = bug_data
            bug_class_dict[int(bug_id)] = updated_result_indexes  # changed
        else:
            removed_reports_count += 1

    return (
        total_errors,
        missing_description_count,
        removed_reports_count,
        filtered_data,
        bug_class_dict,
    )


def load_class_names(file_path):
    class_names = set()
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_names.add(row["className"])
    return class_names


def main():
    report_file_path = "EclipseBugRepository.json"
    class_info_file_path = "class_info.csv"
    filtered_report_file_path = "FilteredEclipseBugRepository.json"
    bug_class_dict_file_path = "model/dataset/bug_class_dict.pt"

    class_names = load_class_names(class_info_file_path)

    (
        total_errors,
        missing_description_count,
        removed_reports_count,
        filtered_data,
        bug_class_dict,
    ) = filter_bug_reports(report_file_path, class_names)

    with open(filtered_report_file_path, "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)

    torch.save(bug_class_dict, bug_class_dict_file_path)

    print(f"Total Errors: {total_errors}")
    print(f"Errors Missing Description: {missing_description_count}")
    print(f"Removed Reports: {removed_reports_count}")
    print(f"Filtered Reports: {len(filtered_data)}")

    if removed_reports_count > 0:
        print(
            f"Removed {removed_reports_count} error reports due to missing valid classes."
        )
    else:
        print("No error reports were removed due to missing valid classes.")


if __name__ == "__main__":
    main()
