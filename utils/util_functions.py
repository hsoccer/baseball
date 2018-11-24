import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
from settings import *
#""%matplotlib inline
plt.style.use('ggplot')
plt.style.use('seaborn-deep')
#%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')

central = [
    "広島",
    "ヤクルト",
    "巨人",
    "ＤｅＮＡ",
    "中日",
    "阪神",
]

pacific = [
    "西武",
    "ソフトバンク",
    "日本ハム",
    "オリックス",
    "ロッテ",
    "楽天",
]


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

def bar_and_df(triple_transfer_list, ad_before, dis_before, after, title="", show_df=True):
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
    plt.title(decode_count(ad[1][0]) + "時の結果" + "({})".format(title), fontsize=30)
    plt.xlabel("プレーの結果", fontsize=20)
    plt.ylabel("割合", fontsize=20)
    plt.show()

    if show_df:
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

def make_df(start=None, end=None, data_dir=DETAIL_DATA_DIR):
    event_list = []
    file_list = os.listdir(data_dir)[1:]
    
    if data_dir == DETAIL_DATA_DIR:
        case_index = 0
        inning_index = 1
        offense_team_index = 6
        defense_team_index = 2
        if start:
            date_series = pd.Series(file_list).apply(filename_to_datetime)
            target_file_list = list(pd.Series(file_list)[(date_series<pd.to_datetime(end)+pd.offsets.timedelta(1)) & (date_series>=pd.to_datetime(start))])
        else:
            target_file_list = file_list
        
    elif data_dir == DETAIL_DATA_DIR_MLB:
        case_index = 0
        inning_index = 7
        offense_team_index = 9
        defense_team_index = 4
        if start or end:
            if not start:
                start = 0
            elif not end:
                end = float("inf")
            date_series = pd.Series(file_list).apply(lambda x: int(x.split(".")[0]))
            target_file_list = list(pd.Series(file_list)[(date_series<end) & (date_series>=start)])
        else:
            target_file_list = file_list

    columns = pd.read_csv(os.path.join(data_dir, file_list[0]), encoding="cp932", index_col=0, dtype="object").columns
    length = len(columns)

    for file in target_file_list:
        curr_event_list = pd.read_csv(os.path.join(data_dir, file), encoding="cp932", index_col=0, dtype="object").values.tolist()
        new_event_list = []
        for i in range(len(curr_event_list)):
            new_event_list.append(curr_event_list[i])
            if i < len(curr_event_list)-1 and curr_event_list[i][case_index][0] == "2" and curr_event_list[i+1][case_index][0] == "1":
                new_event_list.append(["3000", curr_event_list[i][inning_index]]+[np.nan for _ in range(length-2)])
                new_event_list.append(["0000", curr_event_list[i+1][inning_index]]+[np.nan for _ in range(length-2)])
            if i < len(curr_event_list)-1 and curr_event_list[i][case_index][0] == "2" and curr_event_list[i+1][case_index][0] == "0":
                new_event_list.append(["3000", curr_event_list[i][inning_index]]+[np.nan for _ in range(length-2)])
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
    columns = list(event_df.columns)
    case_index = columns.index("状況")
    try:
        inning_index = columns.index("イニング")
    except:
        inning_index = columns.index("回")

    inning_list = []
    for inning in range(18):
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

def ks_test(event_df, kind, title="", return_df=False):
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

    if return_df:
        return df_ks
    else:
        plt.figure(figsize=(10, 5))
        sns.heatmap(df_ks, cmap="Blues", annot=True)
        plt.xlabel("イニング")
        plt.ylabel("イニング")
        plt.title(title+" "+str(df_ks.sum().sum()))
        plt.show()

def ks_statistics_distance(df_ks_1, df_ks_2):
    return ((df_ks_1 - df_ks_2)**2).sum().sum() / sum(df_ks_1.shape)

def similar_teams(target_team, team_ks_dict, central=central, pacific=pacific):
    if target_team in central:
        league = central
    else:
        league = pacific
    res = []
    for team in league:
        res.append((ks_statistics_distance(team_ks_dict[team], team_ks_dict[target_team]), team))
    return sorted(res)

def extract_case(event_df, before_2, before_1):
    con_event_df = pd.concat([event_df, event_df[["状況"]].shift(1).rename({"状況": "状況-1"}, axis=1), event_df[["状況"]].shift(2).rename({"状況": "状況-2"}, axis=1)], axis=1)
    return con_event_df[(con_event_df["状況-1"]==before_1) & (con_event_df["状況-2"]==before_2)]

def make_triple_from_case(con_event_df):
    return [list(reversed(elem)) for elem in con_event_df[["状況", "状況-1", "状況-2"]].values.tolist()]

name_dict = dict([(team, (lambda x: x[0] if x!="阪神" else x[1])(team)) for team in pacific+central])

def make_score_df(team):
    res = []
    columns = ["相手", "得点", "安打", "失点", "被安打"]
    files = os.listdir(SCORE_DATA_DIR)
    for file in files:
        score_df = pd.read_csv(os.path.join(SCORE_DATA_DIR, file), encoding="cp932", index_col=0)
        if not name_dict[team] in score_df.index:
            continue
        opponent = score_df.index[0] if score_df.index[1] == name_dict[team] else score_df.index[1]
        team_score = score_df.loc[name_dict[team], "計"]
        team_hits = score_df.loc[name_dict[team], "安"]
        opponent_score = score_df.loc[opponent, "計"]
        opponent_hits = score_df.loc[opponent, "安"]
        res.append([opponent, team_score, team_hits, opponent_score, opponent_hits])
    df = pd.DataFrame(res, columns=columns)
    df["得点率"] = df["得点"] / df["安打"]
    df["失点率"] = df["失点"] / df["被安打"]
    return df

# エントロピーの計算
# 観測データの１次元配列がインプット
# pyotlibの結果と完全に一致
def entropy(data):
    count = pd.Series([tuple(elem) for elem in data]).value_counts()
    prob = count / sum(count)
    return - sum(prob * np.log(prob))

def cond_entropy(target_data, given_data):
    assert len(given_data.shape) == 2
    combined_data = [(target_elem, tuple(given_elem)) for target_elem, given_elem in zip(target_data, given_data)]
    combined_entropy = entropy(combined_data)
    single_entropy = entropy(given_data)
    return combined_entropy - single_entropy

def is_improved(before, after):
    if before[0] == after[0]:
        if int(before[::-1][:-1]) <= int(after[::-1][:-1]):
            return True
    return False

def make_score_df_mlb(team, year="both"):
    num = 1944417
    res = []
    columns = ["相手", "得点", "失点"]
    df = pd.read_csv(os.path.join(SCORE_DATA_DIR_MLB, "score_mlb.csv"), encoding="cp932", index_col=0)
    if year == "both":
        df_top = df[df["表チーム"]==team]
        df_bot = df[df["裏チーム"]==team]
    elif int(year) == 2018:
        df_top = df[(df["表チーム"]==team) & (df["試合ID"].apply(lambda x: int(x.split(".")[0]))>num)]
        df_bot = df[(df["裏チーム"]==team) & (df["試合ID"].apply(lambda x: int(x.split(".")[0]))>num)]
    elif int(year) == 2017:
        df_top = df[(df["表チーム"]==team) & (df["試合ID"].apply(lambda x: int(x.split(".")[0]))<=num)]
        df_bot = df[(df["裏チーム"]==team) & (df["試合ID"].apply(lambda x: int(x.split(".")[0]))<=num)]
    for elem in df_top.values:
        res.append([elem[2], elem[3], elem[4]])
    for elem in df_bot.values:
        res.append([elem[1], elem[4], elem[3]])
    return pd.DataFrame(res, columns=columns)

def is_improved(before, after):
    if before[0] == after[0]:
        if int(before[::-1][:-1]) <= int(after[::-1][:-1]):
            return True
    return False

def is_deteriorated(before, after):
    if int(before[0]) < int(after):
        if int(before[::-1][:-1]) >= int(after[::-1][:-1]):
            return True
    return False