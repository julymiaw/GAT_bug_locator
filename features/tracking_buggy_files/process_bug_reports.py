#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.xml> <repository_path> <bug_reports.json>
"""

import subprocess
import sys
import xml.etree.ElementTree as ET

from json import dump
from tqdm import tqdm


def main():
    bug_reports_file = sys.argv[1]
    repository = sys.argv[2]
    json_file_name = sys.argv[3]
    dataset = load_dataset(bug_reports_file, repository)
    with open(json_file_name, "w") as f:
        dump(dataset, f)


def load_dataset(bug_reports_file, repository):
    dataset = {}
    tree = ET.parse(bug_reports_file)
    root = tree.getroot()
    for database in root.findall("database"):
        for bug_report in tqdm(database.findall("table")):
            fixing_commit = bug_report.find("column[@name='commit']").text
            bug_report_content = {
                "commit": fixing_commit,
                "id": bug_report.find("column[@name='id']").text,
                "bug_id": bug_report.find("column[@name='bug_id']").text,
                "summary": bug_report.find("column[@name='summary']").text,
                "timestamp": bug_report.find("column[@name='report_timestamp']").text,
                "status": bug_report.find("column[@name='status']").text,
                "result": bug_report.find("column[@name='result']").text,
                "description": bug_report.find("column[@name='description']").text,
            }

            commit_content = retrieve_commit(
                repository, fixing_commit, bug_report_content["bug_id"]
            )

            dataset[fixing_commit] = {
                "bug_report": bug_report_content,
                "commit": commit_content,
            }

    return dataset


def retrieve_commit(repository, commit, bug_id, ext=".java"):
    metadata, commit = retrieve_metadata(repository, commit, bug_id)
    diff = retrieve_diff(repository, commit, ext=ext)
    commit_content = {"metadata": metadata, "diff": diff}
    return commit_content


def retrieve_metadata(repository, commit, bug_id):
    full_sha = None
    author = None
    date = None
    message = ""
    cmd = "git -C " + repository + " show -s " + commit
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    show_lines = stdout.decode("latin-1").split("\n")
    error_message = stderr.decode("utf-8")

    if error_message:
        # 处理歧义错误
        if "存在歧义" in error_message:
            error_lines = error_message.split("\n")
            candidates = []
            for line in error_lines:
                if line.startswith("提示：  "):
                    candidate_info = line.split()
                    if len(candidate_info) > 2 and candidate_info[2] == "commit":
                        candidates.append(candidate_info[1])

            # 尝试找到包含 bug_id 的候选 commit
            for candidate in candidates:
                cmd = "git -C " + repository + " show -s " + candidate
                process = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                show_lines = stdout.decode("latin-1").split("\n")
                error_message = stderr.decode("utf-8")

                if error_message:
                    print(f"Error: {error_message}", file=sys.stderr)
                    continue

                temp_message = ""
                for index, line in enumerate(show_lines):
                    if index == 0:
                        full_sha = line
                    elif index == 1:
                        author = line
                    elif index == 2:
                        date = line
                    else:
                        temp_message += line

                if bug_id in temp_message:
                    message = temp_message
                    commit = candidate
                    break

    if not message:
        for index, line in enumerate(show_lines):
            if index == 0:
                full_sha = line
            elif index == 1:
                author = line
            elif index == 2:
                date = line
            else:
                message += line

    metadata = {"sha": full_sha, "author": author, "date": date, "message": message}
    return metadata, commit


def retrieve_diff(repository, commit, ext=".java"):
    cmd = "git -C " + repository + " diff-tree --no-commit-id --name-only -r " + commit
    #    print cmd
    files = {}
    diff_tree_lines = (
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        .stdout.read()
        .decode("latin-1")
        .split("\n")
    )
    for line in iter(diff_tree_lines):
        filepath = line.rstrip()
        if filepath != "" and filepath.endswith(ext):
            files[filepath] = retrieve_diff_on_filepath(repository, commit, filepath)
    return files


def retrieve_diff_on_filepath(repository, commit, filepath):
    cmd = (
        "git -C "
        + repository
        + " diff --unified=0 --no-prefix "
        + commit
        + "^ "
        + commit
        + " -- "
        + filepath
    )
    diff_lines = (
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        .stdout.read()
        .decode("latin-1")
    )
    return diff_lines


if __name__ == "__main__":
    main()
