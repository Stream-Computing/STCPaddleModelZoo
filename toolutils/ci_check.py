# ci_check.py
import fire
import pandas as pd
import os, json
import numpy as np

def get_commit_id(file_path):
    commit_file = os.path.join(file_path, "reports/commit_id.txt")
    ddk_commit_id = "cannot find"
    tb_commit_id = "cannot find"
    if os.path.exists(commit_file):
        with open(commit_file, "r") as f:
            lines = f.readlines()
            for data in lines:
                if data.find("stc_ddk commit_id :") != -1:
                    ddk_commit_id = data.split(":")[-1]
                if data.find("tensorturbo commit_id :") != -1:
                    tb_commit_id = data.split(":")[-1]
    return tb_commit_id, ddk_commit_id


def ci_check(base_path = "./"):
    base_path = os.path.abspath(base_path)
    df_base = pd.read_csv(base_path + "/reports/STC/Overall.csv")
    df_ci = pd.DataFrame()
    for a, b, c in os.walk(base_path +"/workloads"):
        for cell in sorted(c):
            model_name = cell.split("/")[-1].split(".")[0]
            if not os.path.exists(os.path.join(base_path,"reports/LOG",model_name+".log")):
                continue
            with open(a + "/" + cell) as f:
                data = json.load(f)
                if model_name.upper() in df_base["Model"].values:
                    df_ci = df_ci.append({"model": model_name, "state": "Pass"}, ignore_index=True)
                    continue
            with open(os.path.join(base_path,"reports/LOG",model_name+".log"), "r") as f:
                data = f.readlines()
                data = " ".join(data)

                if data.find("magic_num == 0xfafa") != -1:
                    df_ci = df_ci.append({"model": model_name, "state": "AIC NoPass"}, ignore_index=True)
                    continue
                compile_file_path = os.path.join(base_path, "engines/STC/mix_tmp/",model_name)
                if not os.path.exists(compile_file_path):
                    df_ci = df_ci.append({"model": model_name, "state": "AIC NoPass"}, ignore_index=True)
                    continue

            df_ci = df_ci.append({"model": model_name, "state": "AIE NoPass"}, ignore_index=True)


    tb, ddk = get_commit_id(base_path)
    new_state_name = f'state: tb_commit_id: {tb}, stc_ddk_commit_id: {ddk}'
    df_ci.rename(columns = {"state": new_state_name}, inplace=True)

    df_base.rename(columns = {'Model':'model'}, inplace = True)
    # 创建一个remap，解决大小写风险。
    name_table = {}
    for a in list(df_ci["model"].unique()):
        name_table[a.lower()] = a

    df_base['model'] = df_base['model'].str.lower()
    df_ci['model'] = df_ci['model'].str.lower()
    df_ci = pd.merge(left = df_ci, right = df_base, how = 'left', left_on = ['model'], right_on = ['model']) 

    df_ci.to_csv(os.path.join(base_path, "reports", "ci_log.csv"),encoding="utf_8_sig", index=None)


    # cal performance and accuracy vs base_line if too fluctuating, save in df_issue
    df_issue = pd.DataFrame()
    model_names = list(df_ci["model"].unique())
    for name in model_names:
        temp = df_ci[df_ci["model"] == name]
        if temp[new_state_name].iloc[0] != "Pass":
            print("model covert error")
            df_issue = df_issue.append({"model": name, "state": temp[new_state_name].iloc[0]}, ignore_index=True)
            continue
        temp_dict = dict(temp[temp["batch_mode"] == "batch_normal"].reset_index(drop=True).iloc[0])
        workload_name = name_table[name]
        
        with open(os.path.join(base_path, "workloads", workload_name+".json")) as f:
            workload_json = json.load(f)

        if "base_line" not in workload_json:
            df_issue = df_issue.append({"model": name, "state": "[base_line] not in workload_json"}, ignore_index=True)
            continue

        if "performance" not in workload_json["base_line"]:
            df_issue = df_issue.append({"model": name, "state": "[performance] not in workload_json['base_line']"}, ignore_index=True)
            continue

        if "accuracy" not in workload_json["base_line"]:
            df_issue = df_issue.append({"model": name, "state": "[accuracy] not in workload_json['base_line']"}, ignore_index=True)
            continue

        for key, val in workload_json["base_line"]["performance"].items():
            if key not in temp_dict:
                print("error: do not has performance key match")
                error_name = f"performance key set error: {key} not in outputs"
            else:
                if (abs(float(temp_dict[key]) - val) / (val + np.finfo(float).eps)) > 0.03:
                    error_name = f"performance {key} too fluctuating. this run is [{temp_dict[key]}], but baseline is [{val}]"
                else:
                    continue
            df_issue = df_issue.append({"model": name, "state": error_name}, ignore_index=True)
                
        for key, val in workload_json["base_line"]["accuracy"].items():
            if key not in temp_dict:
                print("error: do not has accuracy match")
                error_name = f"accuracy key set error: {key} not in outputs"
                df_issue = df_issue.append({"model": name, "state": error_name}, ignore_index=True)
            else:
                if (abs(float(temp_dict[key]) - val) / (val + np.finfo(float).eps)) > 0.03:
                    error_name = f"accuracy {key} too fluctuating. this run is [{temp_dict[key]}], but baseline is [{val}]"
                else:
                    continue
            df_issue = df_issue.append({"model": name, "state": error_name}, ignore_index=True)

    df_issue.rename(columns = {"state": new_state_name}, inplace=True)
    df_issue.to_csv(os.path.join(base_path, "reports", "ci_issue.csv"),encoding="utf_8_sig", index=None)

if __name__ == '__main__':
    fire.Fire(ci_check)
