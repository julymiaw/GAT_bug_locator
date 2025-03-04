import re
import pandas as pd
import json
from unqlite import UnQLite
import pickle
from multiprocessing import Pool, cpu_count
import gc

project = "jdt"
fold_number = 12

# project='eclipse_platform_ui'
# fold_number=12

# project = "tomcat"
# fold_number = 2

# project = "aspectj"
# fold_number = 1

# project = "swt"
# fold_number = 8

# project='birt'
# fold_number=8

# import sys

# project = sys.argv[1]
# fold_number = sys.args[2]

test_fold_all = pd.concat(
    [
        pd.read_pickle(f"../{project}/{project}_normalized_testing_fold_{k}")
        for k in range(fold_number + 1)
    ]
)

report_idx = test_fold_all.index.get_level_values(0).unique()
code_idx = test_fold_all.index.get_level_values(1).unique()

print(f"Total bug reports: {len(report_idx)}")
print(f"Total code files: {len(code_idx)}")

del test_fold_all
gc.collect()


def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports


bug_report_file_path = "../" + project + "/" + project + ".json"
bug_reports = load_bug_reports(bug_report_file_path)

bid_list = list(report_idx)
summarys = []
descriptions = []
for bug_report_id in bid_list:
    current_bug_report = bug_reports[bug_report_id]["bug_report"]
    summarys.append(current_bug_report["summary"])
    descriptions.append(current_bug_report["description"])

report_dataFrame = pd.DataFrame(
    {"summary": summarys, "description": descriptions}, index=bid_list
)
report_dataFrame.index.names = ["bid"]


def remove_twoHeadWord(string):
    contents = string.split(" ")[2:]
    return " ".join(contents)


report_dataFrame["summary"] = report_dataFrame["summary"].apply(remove_twoHeadWord)
report_dataFrame.fillna("", inplace=True)
report_dataFrame.head()


def clean_string_report(string):
    outtmp = re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]', string)
    outtmp = [token for token in outtmp if token.isalnum()]
    return " ".join(outtmp)


with open(
    "../joblib_memmap_" + project + "_flim/report.jsonl",
    "w",
    encoding="utf-8",
) as f_out:
    for row in report_dataFrame.iterrows():
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = clean_string_report(
            row[1].summary + " " + row[1].description
        )
        task1["code_tokens"] = ""
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)

print("完成报告处理")


def clean_code_names(ast_file):
    names = (
        ast_file["classNames"]
        + ast_file["superclassNames"]
        + ast_file["interfaceNames"]
        + ast_file["methodNames"]
    )
    names = [" ".join(name.split(".")) for name in names]
    return " ".join(names)


def convert_dict2string_set(dict_list):
    counter_list = set()
    for dict_item in dict_list:
        for k, v in dict_item.items():
            if len(k) <= 2:
                continue
            counter_list.add(k)
    return " ".join(counter_list)


def clean_string_code(string):
    m = re.compile(r"/\*.*?\*/", re.S)
    outstring = re.sub(m, "", string)
    m = re.compile(r"package*", re.S)
    outstring = re.sub(m, "", outstring)
    m = re.compile(r"import*", re.S)
    outstring = re.sub(m, "", outstring)
    m = re.compile(r"//.*")
    outtmp = re.sub(m, "", outstring)
    for char in ["\r\n", "\r", "\n"]:
        outtmp = outtmp.replace(char, " ")
    outtmp = " ".join(outtmp.split())
    outtmp = re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]', outtmp)
    return " ".join(outtmp)


def get_method_top_k(choosed_methods: list, k=50):
    method_list = []
    for string_bef in choosed_methods:
        string = clean_string_code(string_bef)
        if "{" in string and "}" in string:
            method_list.append(string)
    method_list = sorted(
        method_list, key=lambda string: len(string.split()), reverse=True
    )
    if len(method_list) == 0:
        for string_bef in choosed_methods:
            string = clean_string(string_bef)
            method_list.append(string)
        method_list = sorted(
            method_list, key=lambda string: len(string.split()), reverse=True
        )
        return method_list[:k]
    return method_list[:k]


def clean_string(string):
    outstring_list = re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]', string)
    return " ".join(outstring_list)


ast_cache_collection_db = UnQLite(
    "../" + project + "/" + project + "_ast_cache_collection_db",
    flags=0x00000100 | 0x00000002,
)


all_ast_index = list(code_idx)


def process_ast_file(ast_index):
    ast_file = pickle.loads(ast_cache_collection_db[ast_index])

    tokenizedMethods = convert_dict2string_set(ast_file["tokenizedMethods"])
    tokenizedClassNames = convert_dict2string_set(ast_file["tokenizedClassNames"])
    tokenizedMethodNames = convert_dict2string_set(ast_file["tokenizedMethodNames"])
    tokenizedVariableNames = convert_dict2string_set(ast_file["tokenizedVariableNames"])
    tokenizedComments = convert_dict2string_set(ast_file["tokenizedComments"])

    source = clean_string_code(ast_file["rawSourceContent"])
    names = clean_code_names(ast_file)
    top_k_method = get_method_top_k(ast_file["methodContent"], 10)

    return (
        ast_index,
        top_k_method,
        tokenizedMethods,
        tokenizedClassNames,
        tokenizedMethodNames,
        tokenizedVariableNames,
        tokenizedComments,
        source,
        names,
    )


with Pool(cpu_count()) as p:
    results = list(p.map(process_ast_file, all_ast_index))

(
    all_ast_index,
    all_ast_file_methods,
    all_ast_file_tokenizedMethods,
    all_ast_file_tokenizedClassNames,
    all_ast_file_tokenizedMethodNames,
    all_ast_file_tokenizedVariableNames,
    all_ast_file_tokenizedComments,
    all_ast_file_source,
    all_ast_file_names,
) = zip(*results)

all_ast_index_dataframe = pd.DataFrame(
    {
        "all_ast_file_methods": all_ast_file_methods,
        "tokenizedMethods": all_ast_file_tokenizedMethods,
        "tokenizedClassNames": all_ast_file_tokenizedClassNames,
        "tokenizedMethodNames": all_ast_file_tokenizedMethodNames,
        "tokenizedVariableNames": all_ast_file_tokenizedVariableNames,
        "tokenizedComments": all_ast_file_tokenizedComments,
        "source": all_ast_file_source,
        "names": all_ast_file_names,
    },
    index=all_ast_index,
)

with open(
    "../joblib_memmap_" + project + "_flim/code.jsonl",
    "w",
    encoding="utf-8",
) as f_out:
    for row in all_ast_index_dataframe.iterrows():
        all_methods = row[1].all_ast_file_methods
        for method in all_methods:
            task1 = dict()
            task1["url"] = row[0]
            task1["docstring_tokens"] = ""
            task1["code_tokens"] = method
            out_line = json.dumps(task1, ensure_ascii=False) + "\n"
            f_out.write(out_line)
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = ""
        task1["code_tokens"] = row[1].names
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = ""
        task1["code_tokens"] = row[1].source
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)
        # 写tokenizedMethods开始
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = ""
        task1["code_tokens"] = row[1].tokenizedMethods
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)
        # 写tokenizedMethods结束
        # 写tokenizedClassNames开始
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = ""
        task1["code_tokens"] = row[1].tokenizedClassNames
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)
        # 写tokenizedClassNames结束
        # 写tokenizedMethodNames开始
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = ""
        task1["code_tokens"] = row[1].tokenizedMethodNames
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)
        # 写tokenizedMethodNames结束
        # 写tokenizedVariableNames开始
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = ""
        task1["code_tokens"] = row[1].tokenizedVariableNames
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)
        # 写tokenizedVariableNames结束

        # 写tokenizedComments开始
        task1 = dict()
        task1["url"] = row[0]
        task1["docstring_tokens"] = ""
        task1["code_tokens"] = row[1].tokenizedComments
        out_line = json.dumps(task1, ensure_ascii=False) + "\n"
        f_out.write(out_line)
        # 写tokenizedComments结束
