import json
import time
import sys
from timeit import default_timer
from tqdm import tqdm
import bugzilla
import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    bug_reports_without_description_file_path = sys.argv[1]
    print(
        "bug report without description file path",
        bug_reports_without_description_file_path,
    )
    bug_reports_with_description_file_path = sys.argv[2]
    print("output bug report file path", bug_reports_with_description_file_path)
    api_key = sys.argv[3]
    url = sys.argv[4]  # 'https://bugs.eclipse.org/bugs/rest/'

    add_missing_descriptions(
        bug_reports_without_description_file_path,
        bug_reports_with_description_file_path,
        api_key,
        url,
    )

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time ", total)


def add_missing_descriptions(
    in_reports, out_reports, api_key, url, max_retries=20, max_workers=8
):
    with open(in_reports) as bug_report_file:
        bug_reports = json.load(bug_report_file)

    empty_descriptions_keys = [
        key
        for key in bug_reports
        if bug_reports[key]["bug_report"]["description"] is None
    ]
    print("Missing bug report keys size", len(empty_descriptions_keys))

    new_bug_reports = {}
    b = bugzilla.Bugzilla(url=url, api_key=api_key)

    def fetch_description(key):
        current_bug_id = bug_reports[key]["bug_report"]["bug_id"]
        retries = 0
        while retries < max_retries:
            try:
                comments = b.get_comments(current_bug_id)
                description = find_description(
                    comments["bugs"][current_bug_id]["comments"]
                )
                return key, description
            except (
                json.decoder.JSONDecodeError,
                requests.exceptions.ConnectionError,
            ) as e:
                retries += 1
                print(
                    f"Error fetching comments for bug {current_bug_id}, retrying {retries}/{max_retries}"
                )
                time.sleep(2**retries)  # Exponential backoff
                if retries == max_retries:
                    print(
                        f"Failed to fetch comments for bug {current_bug_id} after {max_retries} retries"
                    )
                    return key, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_description, key): key
            for key in empty_descriptions_keys
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            key, description = future.result()
            if description is not None:
                bug_reports[key]["bug_report"]["description"] = description
                new_bug_reports[key] = bug_reports[key]

    with open(out_reports, "w") as bug_report_out_file:
        json.dump(new_bug_reports, bug_report_out_file)


def find_description(comments):
    for comment in comments:
        if comment["count"] == 0:
            return comment["text"]
    return ""


if __name__ == "__main__":
    main()
