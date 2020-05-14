import scipy.stats as st
import scikit_posthocs as hocs
import pandas as pd
from math import sqrt

def calcstats(file, name, type):
    dat = pd.read_excel(file)
    df1 = dat[["Anger", "Fear", "Joy", "Sadness"]]
    friedman = st.friedmanchisquare(*df1.values)
    print(friedman)
    p_values = hocs.posthoc_nemenyi_friedman(df1.T)
    p_values.to_excel(r"Final1\results_" + type + "_" + name + "_p_values.xlsx")
    ranks = pd.DataFrame(columns=df1.keys())
    for key in df1.keys():
        ranks[key] = df1[key].rank(ascending=False)
    df1["Totals"] = df1.sum(axis=1) / 4
    df1["Ranks"] = ranks.mean(axis=1)
    df1 = df1.sort_values(by=["Ranks"])
    df1 = df1.reset_index()
    R_1 = df1["Ranks"].iloc[0]
    df1["P-values"] = df1.apply(lambda row: min(test_pairs(4, len(df1.index), R_1, row["Ranks"])*(row.name+1), 1), axis=1)
    for index, row in df1.iterrows():
        if index != len(df1.index)-1:
            df1["P-values"].iloc[index] = max(df1["P-values"].iloc[index+1:].max(), df1["P-values"].iloc[index])
    df1 = df1.sort_values(by=["index"])
    df1 = df1.reset_index(drop=True)
    df1.to_excel(r"Final1\results_" + type + "_" + name + ".xlsx")
calcstats(r"D:\School\Thesis\FinalCode\Final Output\results_transformations_FRNN.xlsx", "FRNN","Trans")

def test_pairs(N,k,R_1,R_2):
    z = (R_1-R_2)/sqrt(k*(k+1)/(6*N))
    return st.norm.cdf(z)
