import pandas as pd
import numpy as np
df = pd.read_excel("conversion_table.xlsx")

# PCL-5 sum scores: (0,80)
# PCL-C/M sum scores: (17,85)
# DTS sum scores: (0,68)
# MPSS sum scores: (0,51)
# pcl_5_ind = [0,2,4]
# pcl= [1,3,5]
# dts_ind = [6,7,8]
# mpss_ind = [9,10,11]

pcl_5_range = (0,80)
pcl_c_range = (17,85)
dts_range = (0,68)
mpss_range = (0,51)

pcl_5 = pd.DataFrame(columns=["PCL-C","PCL-M","DTS","MPSS"],index=np.arange(pcl_5_range[1] + 1))
pcl_c = pd.DataFrame(columns=["PCL-5","PCL-M","DTS","MPSS"],index=np.arange(pcl_c_range[0],pcl_c_range[1] + 1))
pcl_m = pd.DataFrame(columns=["PCL-5","PCL-C","DTS","MPSS"],index=np.arange(pcl_c_range[0],pcl_c_range[1] + 1))
dts = pd.DataFrame(columns=["PCL-5","PCL-C","PCL-M","MPSS"],index=np.arange(dts_range[1] + 1))
mpss = pd.DataFrame(columns=["PCL-5","PCL-C","PCL-M","DTS"],index=np.arange(mpss_range[1] + 1))

def adjust_score(score,_range):
    return int(np.floor(np.clip(score,_range[0],_range[1])))

for i in pcl_5.index:
    pcl_5.loc[i,"PCL-C"] = adjust_score(df.loc[0,'a0'] + i*df.loc[0,'β1'] + i**2 * df.loc[0,'β2'],pcl_c_range)
    pcl_5.loc[i,"PCL-M"] = adjust_score(df.loc[0,'a0'] + i*df.loc[0,'β1'] + i**2 * df.loc[0,'β2'],pcl_c_range)
    pcl_5.loc[i,"DTS"] = adjust_score(df.loc[4,'a0'] + i*df.loc[4,'β1'] + i**2 * df.loc[4,'β2'],dts_range)
    pcl_5.loc[i,"MPSS"] = adjust_score(df.loc[2,'a0'] + i*df.loc[2,'β1'] + i**2 * df.loc[2,'β2'],mpss_range)

for i in pcl_c.index:
    pcl_c.loc[i,"PCL-5"] = adjust_score(df.loc[1,'a0'] + i*df.loc[1,'β1'] + i**2 * df.loc[1,'β2'],pcl_5_range)
    pcl_c.loc[i,"PCL-M"] = i
    pcl_c.loc[i,"DTS"] = adjust_score(df.loc[3,'a0'] + i*df.loc[3,'β1'] + i**2 * df.loc[3,'β2'],dts_range)
    pcl_c.loc[i,"MPSS"] = adjust_score(df.loc[5,'a0'] + i*df.loc[5,'β1'] + i**2 * df.loc[5,'β2'],mpss_range)

for i in pcl_m.index:
    pcl_m.loc[i,"PCL-5"] = adjust_score(df.loc[1,'a0'] + i*df.loc[1,'β1'] + i**2 * df.loc[1,'β2'],pcl_5_range)
    pcl_m.loc[i,"PCL-C"] = i
    pcl_m.loc[i,"DTS"] = adjust_score(df.loc[3,'a0'] + i*df.loc[3,'β1'] + i**2 * df.loc[3,'β2'],dts_range)
    pcl_m.loc[i,"MPSS"] = adjust_score(df.loc[5,'a0'] + i*df.loc[5,'β1'] + i**2 * df.loc[5,'β2'],mpss_range)

for i in dts.index:
    dts.loc[i,"PCL-5"] = adjust_score(df.loc[6,'a0'] + i*df.loc[6,'β1'] + i**2 * df.loc[6,'β2'],pcl_5_range)
    dts.loc[i,"PCL-C"] = adjust_score(df.loc[7,'a0'] + i*df.loc[7,'β1'] + i**2 * df.loc[7,'β2'],pcl_c_range)
    dts.loc[i,"PCL-M"] = adjust_score(df.loc[7,'a0'] + i*df.loc[7,'β1'] + i**2 * df.loc[7,'β2'],pcl_c_range)
    dts.loc[i,"MPSS"] = adjust_score(df.loc[8,'a0'] + i*df.loc[8,'β1'] + i**2 * df.loc[8,'β2'],mpss_range)

for i in mpss.index:
    mpss.loc[i,"PCL-5"] = adjust_score(df.loc[9,'a0'] + i*df.loc[9,'β1'] + i**2 * df.loc[9,'β2'],pcl_5_range)
    mpss.loc[i,"PCL-C"] = adjust_score(df.loc[10,'a0'] + i*df.loc[10,'β1'] + i**2 * df.loc[10,'β2'],pcl_c_range)
    mpss.loc[i,"PCL-M"] = adjust_score(df.loc[10,'a0'] + i*df.loc[10,'β1'] + i**2 * df.loc[10,'β2'],pcl_c_range)
    mpss.loc[i,"DTS"] = adjust_score(df.loc[11,'a0'] + i*df.loc[11,'β1'] + i**2 * df.loc[11,'β2'],dts_range)

pcl_5.to_csv("./csv_files/pcl-5.csv",index=True)
pcl_c.to_csv("./csv_files/pcl-c.csv",index=True)
pcl_m.to_csv("./csv_files/pcl-m.csv",index=True)
dts.to_csv("./csv_files/dts.csv",index=True)
mpss.to_csv("./csv_files/mpss.csv",index=True)