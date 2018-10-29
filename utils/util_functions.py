import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
from settings import *
#%matplotlib inline
plt.style.use('ggplot')
#%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')


def decode_count(count):
    out = count[0]
    base = list(count[1:])
    out_comment = str(out) + "アウト"
    if out_comment == "3アウト":
        return out_comment + "チェンジ"
    if base == ["0", "0", "0"]:
        return out_comment + "ランナーなし"
    elif base == ["1", "1", "1"]:
        return out_comment + "満塁"
    base = [elem=="1" for elem in base]
    base_comment = ""
    for i in range(3):
        if base[i]:
            base_comment += str(i+1)
    return out_comment + base_comment + "塁"

def bar_and_df(triple_transfer_list, ad_before, dis_before, after):
    ad = pd.DataFrame(sorted([[elem[0][0], elem[0][1], elem[0][2], elem[1]] for elem in list(Counter([elem for elem in triple_transfer_list if elem[0]==ad_before and elem[1]==after]).items())], key=lambda x: -x[-1]))
    dis = pd.DataFrame(sorted([[elem[0][0], elem[0][1], elem[0][2], elem[1]] for elem in list(Counter([elem for elem in triple_transfer_list if elem[0]==dis_before and elem[1]==after]).items())], key=lambda x: -x[-1]))
    ad[4] = ad[3] / sum(ad[3])
    dis[4] = dis[3] / sum(dis[3])
    con = pd.concat([ad.set_index(2), dis.set_index(2)], axis=1, join="outer").fillna(0)[4]
    con.columns = ["ad_rate", "dis_rate"]
    con = con.sort_values(by="ad_rate")

    plt.figure(figsize=(18, 6))
    plt.bar(list(range(len(con.ad_rate))), con.ad_rate, tick_label=[decode_count(elem) for elem in con.index], width=0.35, align="center")
    plt.bar(np.array(list(range(len(con.dis_rate))))+0.35, con.dis_rate, tick_label=[decode_count(elem) for elem in con.index], width=0.35, align="center")
    plt.xticks(rotation=60, fontsize=20)
    plt.legend([decode_count(ad[0][0]), decode_count(dis[0][0])], fontsize=20)
    #plt.title(decode_count(ad[1][0]) + "時の結果", fontsize=30)
    plt.title(decode_count(ad_before)+"→"+decode_count(ad[1][0]) + "時の結果", fontsize=30)
    plt.xlabel("プレーの結果", fontsize=20)
    plt.ylabel("割合", fontsize=20)
    plt.show()

    con = pd.concat([ad.set_index(2), dis.set_index(2)], axis=1, join="outer").fillna(0)[[3, 4]]
    con.columns = ["ad_num", "dis_num", "ad_rate", "dis_rate"]
    con["ad_num"] = con["ad_num"].astype(int)
    con["dis_num"] = con["dis_num"].astype(int)
    con["pvalue"] = [stats.ttest_ind(d1, d2).pvalue for d1, d2 in zip([[0] * (con["ad_num"].sum() - con["ad_num"][idx]) + [1] * con["ad_num"][idx] for idx in con.index], [[0] * (con["dis_num"].sum() - con["dis_num"][idx]) + [1] * con["dis_num"][idx] for idx in con.index])]
    con.index = [decode_count(elem) for elem in con.index]
    display(con.sort_values(by="pvalue"))

def filename_to_datetime(filename):
    year = filename[:4]
    month = filename[4:6]
    day = filename[6:8]
    return pd.to_datetime(year+month+day)

def make_df(start, end):
    case_index = 0
    inning_index = 1
    offense_team_index = 6
    defense_team_index = 2
    event_list = []
    file_list = os.listdir(DATA_DIR)[1:]

    date_series = pd.Series(file_list).apply(filename_to_datetime)
    target_file_list = list(pd.Series(file_list)[(date_series<pd.to_datetime(end)+pd.offsets.timedelta(1)) & (date_series>=pd.to_datetime(start))])

    columns = pd.read_csv(os.path.join(DATA_DIR, file_list[0]), encoding="cp932", index_col=0, dtype="object").columns
    length = len(columns)

    for file in target_file_list:
        curr_event_list = pd.read_csv(os.path.join(DATA_DIR, file), encoding="cp932", index_col=0, dtype="object").values.tolist()
        new_event_list = []
        for i in range(len(curr_event_list)):
            new_event_list.append(curr_event_list[i])
            if i < len(curr_event_list)-1 and curr_event_list[i][case_index][0] == "2" and curr_event_list[i+1][case_index][0] == "1":
                new_event_list.append(["3000", curr_event_list[i][inning_index]]+[np.nan for _ in range(length-2)])
                new_event_list.append(["0000", curr_event_list[i+1][inning_index]]+[np.nan for _ in range(length-2)])
        event_list.extend(new_event_list+[["GAMESET"]+[np.nan for _ in range(length-1)]])

    for i in range(len(event_list)):
        #print(event_list[i][inning_index])
        if event_list[i][inning_index] is np.nan:
            if event_list[i][case_index] != "GAMESET":
                event_list[i][inning_index] = event_list[i-1][inning_index]
                event_list[i][offense_team_index] = event_list[i-1][offense_team_index]
                event_list[i][defense_team_index] = event_list[i-1][defense_team_index]

    return pd.DataFrame(event_list, columns=columns)

def make_inning_list(event_df):
    case_index = 0
    inning_index = 1

    inning_list = []
    for inning in range(10):
        curr_inning_list = []
        for i in range(len(event_df)):
            if event_df.iloc[i, case_index] == "GAMESET":
                continue
            if int(event_df.iloc[i, inning_index].split("回")[0]) == inning + 1:
                curr_inning_list.append(event_df.iloc[i, case_index])
        inning_list.append(curr_inning_list)
    return inning_list

def make_inning_triple(inning_list):
    inning_triple_list = []
    for inning in range(len(inning_list)):
        curr_triple = []
        for i in range(2, len(inning_list[inning])):
            before, curr, after = inning_list[inning][i-2], inning_list[inning][i-1], inning_list[inning][i]
            if "GAMESET" in [before, curr, after]:
                continue
            if before == "3000" or curr == "3000":
                continue
            curr_triple.append((before, curr, after))
        inning_triple_list.append(curr_triple)
    return inning_triple_list

def make_flattened_list(nested_list):
    flattened_list = []
    for elem_list in nested_list:
        flattened_list += elem_list
    return flattened_list

def to_index(lst, dictionary):
    return [dictionary[elem] for elem in lst]

def ks_test(event_df, kind, title=""):
    """
    kind : "statistic" or "pvalue"
    """
    inning_list = make_inning_list(event_df)
    triple_list = make_inning_triple(inning_list)

    triple_set = set(make_flattened_list(triple_list))

    # 状況とインデックスの対応dict
    triple_dict = dict()
    for i, triple in enumerate(triple_set):
        triple_dict[triple] = i

    inning_triple_index_list = []
    for lst in triple_list:
        inning_triple_index_list.append(to_index(lst, triple_dict))

    # KS検定の結果
    df_ks = pd.DataFrame(index=[_ for _ in range(1, 10)], columns=[_ for _ in range(1, 10)]).astype(float)
    for i in range(9):
        for j in range(9):
            if kind == "statistic":
                df_ks.iloc[i, j] = stats.ks_2samp(inning_triple_index_list[i], inning_triple_index_list[j]).statistic
            elif kind == "pvalue":
                df_ks.iloc[i, j] = stats.ks_2samp(inning_triple_index_list[i], inning_triple_index_list[j]).pvalue

    plt.figure(figsize=(10, 5))
    sns.heatmap(df_ks, cmap="Blues", annot=True)
    plt.xlabel("イニング")
    plt.ylabel("イニング")
    plt.title(title+" "+str(df_ks.sum().sum()))
    plt.show()
