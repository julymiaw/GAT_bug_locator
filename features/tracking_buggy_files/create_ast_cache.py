#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <repository_path> <bug_reports.json> <data_prefix>
"""

import datetime
import json
import pickle
import subprocess
import sys

from joblib import Parallel, delayed

from date_utils import convert_commit_date
from multiprocessing import Pool
from operator import itemgetter
from timeit import default_timer
from tqdm import tqdm
from unqlite import UnQLite

# 控制是否使用 tqdm 进度条
USE_TQDM = True  # 设置为 False 以使用普通循环
num_threads = 4


def check_git_installed():
    try:
        # 尝试运行 git --version 命令
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Git is installed.")
    except subprocess.CalledProcessError:
        # 如果命令执行失败，提示用户安装 git
        print(
            "Error: Git is not installed. Please install Git before running this script."
        )
        sys.exit(1)


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()

    # 检查 git 是否安装
    check_git_installed()

    repository_path = sys.argv[1]
    print("repository path", repository_path)
    bug_report_file_path = sys.argv[2]
    print("bug report file path", bug_report_file_path)
    data_prefix = sys.argv[3]
    print("data prefix", data_prefix)

    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        process(bug_reports, repository_path, data_prefix)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time ", total)


def process(bug_reports, repository_path, data_prefix):
    ast_cache = prepare_ast_cache(repository_path)

    ast_cache_collection_db = UnQLite(data_prefix + "_ast_cache_collection_db")

    before = default_timer()
    for k, v in ast_cache.items():
        # k：java 文件的 sha
        # v：类信息字典
        # commits
        # - classname：主类和静态内部类的全限定名
        # - superclassNames：extends 继承的父类的全限定名
        # - interfaceNames：implements 实现的接口的全限定名
        # - methodNames：所有方法的名称
        # - variableNames：所有的变量名，包含类字段，以及局部变量字段等
        # - methodContent：所有的方法
        # - methodVariableTypes：与刚刚的顺序一致，每个方法的参数类型
        # - commentContent：所有的注释
        # - rawSourceContent：原始代码
        # tokenized_counters：
        # - classname：主类和静态内部类的全限定名
        # - tokenizedClassNames：类名拆分成词，并计数
        # - superclassNames：extends 继承的父类的全限定名
        # - tokenizedSuperclassNames：将父类名拆分成词，并计数
        # - interfaceNames：implements 实现的接口的全限定名
        # - tokenizedInterfaceNames：接口名拆分成词，并计数
        # - tokenizedMethodNames：将方法名拆分成词，并计数
        # - tokenizedMethods：将方法内容拆分成词，并计数
        # - tokenizedVariableNames：将每个变量名分别拆分成词，并计数
        # - methodVariableTypes：与刚刚的顺序一致，每个方法的参数类型
        # - tokenizedComments：注释拆分成词，并计数
        # - tokenizedSource：将源代码拆分成词，并计数

        ast_cache_collection_db[k] = pickle.dumps(v, -1)
    after = default_timer()
    total = after - before
    print("total ast cache saving time ", total)

    bug_report_files = prepare_bug_report_files(repository_path, bug_reports, ast_cache)

    before = default_timer()

    bug_report_files_collection_db = UnQLite(
        data_prefix + "_bug_report_files_collection_db"
    )
    for k, v in bug_report_files.items():
        # k：commit 对应的 sha
        # v：字典：
        # - shas：本次提交时，所有与 ast cache 匹配的文件 sha
        # - class_name_to_sha：文件中所有类名（全限定名）到 sha 的映射
        # - sha_to_file_name：sha 到文件名称的映射
        bug_report_files_collection_db[k] = pickle.dumps(v, -1)

    after = default_timer()
    total = after - before
    print("total bug report files saving time ", total)


def list_notes(repository_path, refs="refs/notes/commits"):
    cmd = " ".join(["git", "-C", repository_path, "notes", "--ref", refs, "list"])
    notes_lines = (
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        .stdout.read()
        .decode("latin-1")
        .split("\n")
    )
    return notes_lines


def cat_file_blob(repository_path, sha, encoding="latin-1"):
    # 根据 sha 值获取信息。
    cmd = " ".join(["git", "-C", repository_path, "cat-file", "blob", sha])
    cat_file_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result = cat_file_process.stdout.read().decode(encoding)
    return result


def ls_tree(repository_path, sha):
    cmd = " ".join(["git", "-C", repository_path, "ls-tree", "-r", sha])
    ls_results = (
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        .stdout.read()
        .decode("latin-1")
        .split("\n")
    )
    return ls_results[:-1]


def _process_notes(note, repository_path):
    (note_content_sha, note_object_sha) = note.split(" ")
    note_content = cat_file_blob(repository_path, note_content_sha)
    ast_extraction_result = json.loads(note_content)
    return note_object_sha, ast_extraction_result


def _f(args):
    return _process_notes(args[0], args[1])


def prepare_ast_cache(repository_path):
    tokenized_refs = "refs/notes/tokenized_counters"
    ast_notes = list_notes(repository_path, refs=tokenized_refs)
    print("existing tokenized notes ", len(ast_notes))

    before = default_timer()

    work = []
    for note in ast_notes:
        if note != "":
            work.append((note, repository_path))

    with Pool(num_threads) as pool:  # 统一使用进程池
        iterator = pool.imap(_f, work)

        if USE_TQDM:
            iterator = tqdm(iterator, total=len(work))  # 条件式进度条包装

        ast_cache = dict(iterator)  # 统一处理结果

    after = default_timer()
    total = after - before

    print("total ast cache retrieval time ", total)
    print("size of ast cache ", sys.getsizeof(ast_cache))
    return ast_cache


def sort_bug_reports_by_commit_date(bug_reports):
    commit_dates = []
    if USE_TQDM:
        iterable = tqdm(bug_reports)
    else:
        iterable = bug_reports
    for commit in iterable:
        sha = (
            bug_reports[commit]["commit"]["metadata"]["sha"]
            .replace("commit ", "")
            .strip()
        )
        commit_date = convert_commit_date(
            bug_reports[commit]["commit"]["metadata"]["date"]
            .replace("Date:", "")
            .strip()
        )
        commit_dates.append((sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [commit_date[0] for commit_date in sorted_commit_dates]
    return sorted_commits


def _load_parent_commit_files(repository_path, commit, ast_cache):
    parent = commit + "^"

    class_name_to_sha = {}
    sha_to_file_name = {}
    shas = []
    # 获取特定 commit 时，整个项目的所有 java 文件，并获取其中 sha 值符合的，将他们的类名映射到文件的 sha 值
    for ls_entry in ls_tree(repository_path, parent):
        (file_sha_part, file_name) = ls_entry.split("\t")
        file_sha = file_sha_part.split(" ")[2]
        # file_sha = intern(file_sha)
        # file_name = intern(file_name)
        if file_name.endswith(".java") and file_sha in ast_cache:
            # shas.append(intern(file_sha))
            file_sha_ascii = file_sha
            shas.append(file_sha_ascii)
            class_names = ast_cache[file_sha]["classNames"]
            for class_name in class_names:
                class_name_ascii = class_name
                class_name_to_sha[class_name_ascii] = file_sha_ascii
            sha_to_file_name[file_sha_ascii] = file_name

    f_lookup = {
        "shas": shas,
        "class_name_to_sha": class_name_to_sha,
        "sha_to_file_name": sha_to_file_name,
    }
    return commit.encode("ascii", "ignore"), f_lookup


def prepare_bug_report_files(repository_path, bug_reports, ast_cache):
    sorted_commits = sort_bug_reports_by_commit_date(bug_reports)

    before = default_timer()

    if USE_TQDM:
        r = Parallel(n_jobs=6 * 12, backend="threading")(
            delayed(_load_parent_commit_files)(repository_path, commit, ast_cache)
            for commit in tqdm(sorted_commits)
        )
    else:
        r = Parallel(n_jobs=6 * 12, backend="threading")(
            delayed(_load_parent_commit_files)(repository_path, commit, ast_cache)
            for commit in sorted_commits
        )
    bug_report_files = dict(r)

    after = default_timer()
    total = after - before
    print("total bug report files retrieval time ", total)
    return bug_report_files


if __name__ == "__main__":
    main()
