import io
from typing import List
import os
from pathlib import Path
from shiny import App,reactive, render, ui,Session
import pandas as pd
import numpy as np

css_path = "styles.css"
app_ui = ui.page_fluid(
#    ui.panel_title("Hello Shiny!"),
    {'class' : 'container'},
    ui.include_css(css_path),
    # shinyswatch.theme.darkly(),
    ui.tags.h3("Conversion Calculator", class_="app-heading"),
    ui.tags.div(
        {"class" : "file"},
        ui.input_file("file1", "Choose a file to upload for conversion. Please see template for column names.", accept=[".csv"],multiple=False),
        ui.tags.div(
        {"class" : "buttons"},
        ui.tags.div(
        {"class" : "switch"},
        ui.span("Original"),
        ui.input_switch("newtable","Converted",False),
        ),
        ui.download_button("download_template", "Download Template"),
        ui.download_button("download_conversion", "Download Conversion"),
        # ui.input_action_button(id="exit", label="Close App")
        ),
    ),
    ui.tags.div(
    {"class" : "table_parent"},
    ui.output_table("file_content")
    )
    
)

path = './csv_files'
def server(input, output, session:Session):
    MAX_SIZE = 50000

    @session.download(filename="verbal-learning-template.xlsx")
    def download_template():
        return os.path.join(path,'verbal-learning-template.xlsx')

    # @reactive.Effect
    # # @reactive.event(input.exit)
    # async def _():
    #     start = time.time() 
    #     end = 0
    #     while True:
    #         end = reactive.Value(time.time() - start)
    #         if end.get() > 900:
    #             await session.close()
    #             break

    @output
    @render.table
    def file_content():
        file_infos = input.file1()
        if not file_infos:
            return

        if len(file_infos) > 1:
            return

        return parse_input_csv(file_infos[0]["datapath"]) if not input.newtable() else conversion(parse_input_csv(file_infos[0]['datapath']))

    @reactive.Calc
    def run_conversion():
        file_infos = input.file1()
        if not file_infos:
            return
        return conversion(parse_input_csv(file_infos[0]["datapath"]))

    @session.download(filename="converted.csv")
    def download_conversion():
        file_infos = input.file1()
        if not file_infos:
            ui.notification_show(f"No file uploaded for conversion ",duration=5,close_button=True,type="error")
            return

        try:
            # pass
            return io.BytesIO(dataframe_to_csv(run_conversion()).encode("utf-8"))
        except Exception as e:
            return io.BytesIO(str(e).encode("utf-8"))

app = App(app_ui, server)

def dataframe_to_csv(dataframe: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a CSV string."""
    return dataframe.to_csv(index=False)

def parse_input_csv(input: str) -> pd.DataFrame:
    """Parse the input CSV into a pandas DataFrame."""
    
    if Path(input).suffix == ".csv":
        return pd.read_csv(input)
    elif Path(input).suffix == ".xlsx":
        return pd.read_excel(input,sheet_name=0)

# Conversion function indepedent of classes

cvlt_cols = ['cvlt_imfr_t1_c','cvlt_imfr_t2_c','cvlt_imfr_t3_c','cvlt_imfr_t4_c','cvlt_imfr_t5_c','cvlt_imfr_t15_total','cvlt_imfr_b_c','cvlt_sdfr_c','cvlt_ldfr_c',
            #  'cvlt_recog_hits','cvlt_recog_fp',
             ]

ravlt_cols = ['ravlt_imfr_t1_c','ravlt_imfr_t2_c','ravlt_imfr_t3_c','ravlt_imfr_t4_c','ravlt_imfr_t5_c','ravlt_imfr_t15_total','ravlt_imfr_b_c','ravlt_sdfr_c','ravlt_ldfr_c',
            #   'ravlt_recog_hits','ravlt_recog_fp',
              ]

hvlt_cols = ['hvlt_imfr_t1_c','hvlt_imfr_t2_c','hvlt_imfr_t3_c','hvlt_imfr_t13_total','hvlt_dr_c'
            #  'hvlt_recog_hits','hvlt_recog_fp'
             ]

max_values = {
    "cvlt_imfr_t1_c": 16,
    "cvlt_imfr_t2_c": 16,
    "cvlt_imfr_t3_c": 16,
    "cvlt_imfr_t4_c": 16,
    "cvlt_imfr_t5_c": 16,
    "cvlt_imfr_t15_total": 80,
    "cvlt_imfr_b_c": 16,
    "cvlt_sdfr_c": 16,
    "cvlt_ldfr_c": 16,
    "cvlt_recog_hits": 16,
    "cvlt_recog_fp": 16,
    "ravlt_imfr_t1_c": 15,
    "ravlt_imfr_t2_c": 15,
    "ravlt_imfr_t3_c": 15,
    "ravlt_imfr_t4_c": 15,
    "ravlt_imfr_t5_c": 15,
    "ravlt_imfr_t15_total": 75,
    "ravlt_imfr_b_c": 15,
    "ravlt_sdfr_c": 15,
    "ravlt_ldfr_c": 15,
    "ravlt_recog_hits": 15,
    "ravlt_recog_fp": 15,
    "hvlt_imfr_t1_c": 12,
    "hvlt_imfr_t2_c": 12,
    "hvlt_imfr_t3_c": 12,
    "hvlt_imfr_t13_total": 36,
    "hvlt_dr_c": 12,
    "hvlt_recog_hits": 12,
    "hvlt_recog_fp": 12,
}

def replace_in_csv(input_data,measures_data,input_col,index,target_col,target_col_num):
    if not np.isnan(input_data.loc[index,input_col]):
        if (input_data.loc[index,input_col] <= max_values[input_col] and input_data.loc[index,input_col] >= 0):
            input_data.loc[index,target_col] = str(measures_data.iloc[int(input_data.loc[index,input_col]),target_col_num])

def conversion(input_data: pd.DataFrame) -> pd.DataFrame:
    # error_flag = 0
    all_cols = set(cvlt_cols + ravlt_cols + hvlt_cols)
    input_cols = set(input_data.columns)

    if (len(all_cols & input_cols) == 0):
        ui.notification_show(f"Couldn't find correct column names, please use template for data upload ",duration=8,close_button=True,type="error")
        return
    for col in cvlt_cols + ravlt_cols + hvlt_cols:
        if col not in input_data.columns:
            input_data[col] = np.nan
            # if error_flag == 0:
                # ui.notification_show(f"Couldn't find {col} in data ",duration=5,close_button=True,type="warning")
            # error_flag+=1
            
    
    notnull_cvlt_sum = input_data[cvlt_cols].notnull().any().sum()
    notnull_ravlt_sum = input_data[ravlt_cols].notnull().any().sum()
    notnull_hvlt_sum = input_data[hvlt_cols].notnull().any().sum()

    input_col_list = ["cvlt","ravlt","hvlt"]
    max_ind = np.array([notnull_cvlt_sum, notnull_ravlt_sum,notnull_hvlt_sum]).argmax()
    input_col = input_col_list[max_ind]

    if input_col == "cvlt":
        cvlt_t1 = parse_input_csv(os.path.join(path,'cvlt_t1_cw.csv'))
        cvlt_t15 = parse_input_csv(os.path.join(path,'cvlt_t15_cw.csv'))
        cvlt_sdfr = parse_input_csv(os.path.join(path,'cvlt_sdfr_cw.csv'))
        cvlt_ldfr = parse_input_csv(os.path.join(path,'cvlt_ldfr_cw.csv'))

        input_data.loc[:,ravlt_cols + hvlt_cols] = input_data.loc[:,ravlt_cols + hvlt_cols].astype('object')

        for i in input_data.index:
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t1_c',i,'ravlt_imfr_t1_c',1)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t1_c',i,'hvlt_imfr_t1_c',2)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t2_c',i,'ravlt_imfr_t2_c',1)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t2_c',i,'hvlt_imfr_t2_c',2)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t3_c',i,'ravlt_imfr_t3_c',1)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t3_c',i,'hvlt_imfr_t3_c',2)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t4_c',i,'ravlt_imfr_t4_c',1)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_t5_c',i,'ravlt_imfr_t5_c',1)
            replace_in_csv(input_data,cvlt_t15,'cvlt_imfr_t15_total',i,'ravlt_imfr_t15_total',1)
            replace_in_csv(input_data,cvlt_t15,'cvlt_imfr_t15_total',i,'hvlt_imfr_t13_total',2)
            replace_in_csv(input_data,cvlt_t1,'cvlt_imfr_b_c',i,'ravlt_imfr_b_c',1)
            replace_in_csv(input_data,cvlt_sdfr,'cvlt_sdfr_c',i,'ravlt_sdfr_c',1)
            replace_in_csv(input_data,cvlt_ldfr,'cvlt_ldfr_c',i,'ravlt_ldfr_c',1)
            replace_in_csv(input_data,cvlt_ldfr,'cvlt_ldfr_c',i,'hvlt_dr_c',2)
            # replace_in_csv(input_data,cvlt_t1,'cvlt_recog_hits',i,'ravlt_recog_hits',1)
            # replace_in_csv(input_data,cvlt_t1,'cvlt_recog_hits',i,'hvlt_recog_hits',2)
            # replace_in_csv(input_data,cvlt_t1,'cvlt_recog_fp',i,'ravlt_recog_fp',1)
            # replace_in_csv(input_data,cvlt_t1,'cvlt_recog_fp',i,'hvlt_recog_fp',2)

        for col in ravlt_cols + hvlt_cols:            
            cond = input_data[col].str.contains(r'\+',na=False)
            if input_data.loc[cond,col].unique().size > 0:
                plus_value = input_data.loc[cond,col].unique()[0]
                plus_value = int(plus_value.split("+")[0])
                range_choices = np.arange(plus_value,max_values[col] + 1)
                input_data.loc[cond,col] = np.random.choice(range_choices,
                                                            size=len(input_data.loc[cond,col]),
                                                            p=np.ones(len(range_choices))/len(range_choices))
            not_nans = input_data[col].notna()
            input_data.loc[not_nans, col] =input_data.loc[not_nans, col].astype(int)
            
        ravlt_trials = ['ravlt_imfr_t1_c','ravlt_imfr_t2_c','ravlt_imfr_t3_c','ravlt_imfr_t4_c','ravlt_imfr_t5_c']
        ravlt_sum = input_data.loc[:,ravlt_trials].sum(axis=1)
        hvlt_trials = ['hvlt_imfr_t1_c','hvlt_imfr_t2_c','hvlt_imfr_t3_c']
        hvlt_sum = input_data.loc[:,hvlt_trials].sum(axis=1)
        ravlt_diff = input_data['ravlt_imfr_t15_total'] - ravlt_sum
        hvlt_diff = input_data['hvlt_imfr_t13_total'] - hvlt_sum
        # ravlt_unit = np.sign(ravlt_diff)
        # While subtracting, if any value went below zero ?
        # Adjusting for RAVLT data
        for index in input_data.index:
            if (input_data.loc[index,ravlt_trials].isna().any()) or (ravlt_diff.values[index] == 0):
                continue
            ravlt_index_sum = input_data.loc[index,ravlt_trials].sum()
            ravlt_index_diff = input_data.loc[index,'ravlt_imfr_t15_total'] - ravlt_index_sum
            ravlt_index_unit = np.sign(ravlt_index_diff)
            while ravlt_index_diff != 0:
                for ravlt_trial in ravlt_trials:
                    input_data.loc[index,ravlt_trial] = input_data.loc[index,ravlt_trial] + ravlt_index_unit
                    ravlt_index_diff-= ravlt_index_unit
                    if ravlt_index_diff == 0:
                        break
        # Adjusting for HVLT data
        for index in input_data.index:
            if (input_data.loc[index,hvlt_trials].isna().any()) or (hvlt_diff.values[index] == 0):
                continue
            hvlt_index_sum = input_data.loc[index,hvlt_trials].sum()
            hvlt_index_diff = input_data.loc[index,'hvlt_imfr_t13_total'] - hvlt_index_sum
            hvlt_index_unit = np.sign(hvlt_index_diff)

            while hvlt_index_diff != 0:
                for hvlt_trial in hvlt_trials:
                    input_data.loc[index,hvlt_trial] = input_data.loc[index,hvlt_trial] + hvlt_index_unit
                    hvlt_index_diff-= hvlt_index_unit
                    if hvlt_index_diff == 0:
                        break

        # while ravlt_diff.all() != 0:
        #     for ravlt_trial in ravlt_trials:
        #         input_data[ravlt_trial] = input_data[ravlt_trial] + ravlt_unit
        #         ravlt_diff-= ravlt_unit
        #         if ravlt_diff.all() == 0:
        #             break
        # hvlt_unit = np.sign(hvlt_diff)
        # while hvlt_diff.all() != 0:
        #     for hvlt_trial in hvlt_trials:
        #         input_data[hvlt_trial] = input_data[hvlt_trial] + hvlt_unit
        #         hvlt_diff-= hvlt_unit
        #         if hvlt_diff.all() == 0:
        #             break
        
        ui.notification_show("Converted CVLT !",duration=3,close_button=True,type="message")

            

# 76+ => random sample from 76 - 80 probability even
    elif input_col == "ravlt":
        ravlt_t1 = parse_input_csv(os.path.join(path,'ravlt_t1_cw.csv'))
        ravlt_t15 = parse_input_csv(os.path.join(path,'ravlt_t15_cw.csv'))
        ravlt_sdfr = parse_input_csv(os.path.join(path,'ravlt_sdfr_cw.csv'))
        ravlt_ldfr = parse_input_csv(os.path.join(path,'ravlt_ldfr_cw.csv'))

        input_data.loc[:,hvlt_cols + cvlt_cols] = input_data.loc[:,hvlt_cols + cvlt_cols].astype('object')
        for i in input_data.index:
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t1_c',i,'hvlt_imfr_t1_c',1)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t1_c',i,'cvlt_imfr_t1_c',2)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t2_c',i,'hvlt_imfr_t2_c',1)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t2_c',i,'cvlt_imfr_t2_c',2)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t3_c',i,'hvlt_imfr_t3_c',1)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t3_c',i,'cvlt_imfr_t3_c',2)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t4_c',i,'cvlt_imfr_t4_c',2)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_t5_c',i,'cvlt_imfr_t5_c',2)
            replace_in_csv(input_data,ravlt_t15,'ravlt_imfr_t15_total',i,'hvlt_imfr_t13_total',1)
            replace_in_csv(input_data,ravlt_t15,'ravlt_imfr_t15_total',i,'cvlt_imfr_t15_total',2)
            replace_in_csv(input_data,ravlt_t1,'ravlt_imfr_b_c',i,'cvlt_imfr_b_c',2)
            replace_in_csv(input_data,ravlt_sdfr,'ravlt_sdfr_c',i,'cvlt_sdfr_c',1)
            replace_in_csv(input_data,ravlt_ldfr,'ravlt_ldfr_c',i,'hvlt_dr_c',1)
            replace_in_csv(input_data,ravlt_ldfr,'ravlt_ldfr_c',i,'cvlt_ldfr_c',2)
            # replace_in_csv(input_data,ravlt_t1,'ravlt_recog_hits',i,'hvlt_recog_hits',1)
            # replace_in_csv(input_data,ravlt_t1,'ravlt_recog_hits',i,'cvlt_recog_hits',2)
            # replace_in_csv(input_data,ravlt_t1,'ravlt_recog_fp',i,'hvlt_recog_fp',1)
            # replace_in_csv(input_data,ravlt_t1,'ravlt_recog_fp',i,'cvlt_recog_fp',2)

        for col in hvlt_cols + cvlt_cols:
            cond = input_data[col].str.contains(r'\+',na=False)
            if input_data.loc[cond,col].unique().size > 0:
                plus_value = input_data.loc[cond,col].unique()[0]
                plus_value = int(plus_value.split("+")[0])
                range_choices = np.arange(plus_value,max_values[col] + 1)
                input_data.loc[cond,col] = np.random.choice(range_choices,
                                                            size=len(input_data.loc[cond,col]),
                                                            p=np.ones(len(range_choices))/len(range_choices))
            not_nans = input_data[col].notna()
            input_data.loc[not_nans, col] =input_data.loc[not_nans, col].astype(int)
        
        cvlt_trials = ['cvlt_imfr_t1_c','cvlt_imfr_t2_c','cvlt_imfr_t3_c','cvlt_imfr_t4_c','cvlt_imfr_t5_c']
        cvlt_sum = input_data.loc[:,cvlt_trials].sum(axis=1)
        hvlt_trials = ['hvlt_imfr_t1_c','hvlt_imfr_t2_c','hvlt_imfr_t3_c']
        hvlt_sum = input_data.loc[:,hvlt_trials].sum(axis=1)
        cvlt_diff = input_data['cvlt_imfr_t15_total'] - cvlt_sum
        hvlt_diff = input_data['hvlt_imfr_t13_total'] - hvlt_sum
        # ravlt_unit = np.sign(ravlt_diff)
        # While subtracting, if any value went below zero ?
        # Adjusting for CVLT data
        for index in input_data.index:
            if (input_data.loc[index,cvlt_trials].isna().any()) or (cvlt_diff.values[index] == 0):
                continue
            cvlt_index_sum = input_data.loc[index,cvlt_trials].sum()
            cvlt_index_diff = input_data.loc[index,'cvlt_imfr_t15_total'] - cvlt_index_sum
            cvlt_index_unit = np.sign(cvlt_index_diff)
            while cvlt_index_diff != 0:
                for cvlt_trial in cvlt_trials:
                    input_data.loc[index,cvlt_trial] = input_data.loc[index,cvlt_trial] + cvlt_index_unit
                    cvlt_index_diff-= cvlt_index_unit
                    if cvlt_index_diff == 0:
                        break
        # Adjusting for HVLT data
        for index in input_data.index:
            if (input_data.loc[index,hvlt_trials].isna().any()) or (hvlt_diff.values[index] == 0):
                continue
            hvlt_index_sum = input_data.loc[index,hvlt_trials].sum()
            hvlt_index_diff = input_data.loc[index,'hvlt_imfr_t13_total'] - hvlt_index_sum
            hvlt_index_unit = np.sign(hvlt_index_diff)

            while hvlt_index_diff != 0:
                for hvlt_trial in hvlt_trials:
                    input_data.loc[index,hvlt_trial] = input_data.loc[index,hvlt_trial] + hvlt_index_unit
                    hvlt_index_diff-= hvlt_index_unit
                    if hvlt_index_diff == 0:
                        break
        
        ui.notification_show("Converted RAVLT !",duration=3,close_button=True,type="message")
    
    elif input_col == "hvlt":
        hvlt_t1 = parse_input_csv(os.path.join(path,'hvlt_t1_cw.csv'))
        hvlt_t15 = parse_input_csv(os.path.join(path,'hvlt_t15_cw.csv'))
        hvlt_ldfr = parse_input_csv(os.path.join(path,'hvlt_ldrf_cw.csv'))

        input_data.loc[:,cvlt_cols + ravlt_cols] = input_data.loc[:,cvlt_cols + ravlt_cols].astype('object')
        for i in input_data.index:
            replace_in_csv(input_data,hvlt_t1,'hvlt_imfr_t1_c',i,'cvlt_imfr_t1_c',1)
            replace_in_csv(input_data,hvlt_t1,'hvlt_imfr_t1_c',i,'ravlt_imfr_t1_c',2)
            replace_in_csv(input_data,hvlt_t1,'hvlt_imfr_t2_c',i,'cvlt_imfr_t2_c',1)
            replace_in_csv(input_data,hvlt_t1,'hvlt_imfr_t2_c',i,'ravlt_imfr_t2_c',2)
            replace_in_csv(input_data,hvlt_t1,'hvlt_imfr_t3_c',i,'cvlt_imfr_t3_c',1)
            replace_in_csv(input_data,hvlt_t1,'hvlt_imfr_t3_c',i,'ravlt_imfr_t3_c',2)
            replace_in_csv(input_data,hvlt_t15,'hvlt_imfr_t13_total',i,'cvlt_imfr_t15_total',1)
            replace_in_csv(input_data,hvlt_t15,'hvlt_imfr_t13_total',i,'ravlt_imfr_t15_total',2)
            replace_in_csv(input_data,hvlt_ldfr,'hvlt_dr_c',i,'cvlt_ldfr_c',1)
            replace_in_csv(input_data,hvlt_ldfr,'hvlt_dr_c',i,'ravlt_ldfr_c',2)
            # replace_in_csv(input_data,hvlt_t1,'hvlt_recog_hits',i,'cvlt_recog_hits',1)
            # replace_in_csv(input_data,hvlt_t1,'hvlt_recog_hits',i,'ravlt_recog_hits',2)
            # replace_in_csv(input_data,hvlt_t1,'hvlt_recog_fp',i,'cvlt_recog_fp',1)
            # replace_in_csv(input_data,hvlt_t1,'hvlt_recog_fp',i,'ravlt_recog_fp',2)
        for col in cvlt_cols + ravlt_cols:
            cond = input_data[col].str.contains(r'\+',na=False)
            if input_data.loc[cond,col].unique().size > 0:
                plus_value = input_data.loc[cond,col].unique()[0]
                plus_value = int(plus_value.split("+")[0])
                range_choices = np.arange(plus_value,max_values[col] + 1)
                input_data.loc[cond,col] = np.random.choice(range_choices,
                                                            size=len(input_data.loc[cond,col]),
                                                            p=np.ones(len(range_choices))/len(range_choices))
            not_nans = input_data[col].notna()
            input_data.loc[not_nans, col] =input_data.loc[not_nans, col].astype(int)
        
        ravlt_trials = ['ravlt_imfr_t1_c','ravlt_imfr_t2_c','ravlt_imfr_t3_c','ravlt_imfr_t4_c','ravlt_imfr_t5_c']
        ravlt_sum = input_data.loc[:,ravlt_trials].sum(axis=1)
        cvlt_trials = ['cvlt_imfr_t1_c','cvlt_imfr_t2_c','cvlt_imfr_t3_c','cvlt_imfr_t4_c','cvlt_imfr_t5_c']
        cvlt_sum = input_data.loc[:,cvlt_trials].sum(axis=1)
        ravlt_diff = input_data['ravlt_imfr_t15_total'] - ravlt_sum
        cvlt_diff = input_data['cvlt_imfr_t15_total'] - cvlt_sum

        # Adjusting for RAVLT data
        for index in input_data.index:
            if (input_data.loc[index,ravlt_trials].isna().any()) or (ravlt_diff.values[index] == 0):
                continue
            ravlt_index_sum = input_data.loc[index,ravlt_trials].sum()
            ravlt_index_diff = input_data.loc[index,'ravlt_imfr_t15_total'] - ravlt_index_sum
            ravlt_index_unit = np.sign(ravlt_index_diff)
            while ravlt_index_diff != 0:
                for ravlt_trial in ravlt_trials:
                    input_data.loc[index,ravlt_trial] = input_data.loc[index,ravlt_trial] + ravlt_index_unit
                    ravlt_index_diff-= ravlt_index_unit
                    if ravlt_index_diff == 0:
                        break
        
        # Adjusting for CVLT data
        for index in input_data.index:
            if (input_data.loc[index,cvlt_trials].isna().any()) or (cvlt_diff.values[index] == 0):
                continue
            cvlt_index_sum = input_data.loc[index,cvlt_trials].sum()
            cvlt_index_diff = input_data.loc[index,'cvlt_imfr_t15_total'] - cvlt_index_sum
            cvlt_index_unit = np.sign(cvlt_index_diff)
            while cvlt_index_diff != 0:
                for cvlt_trial in cvlt_trials:
                    input_data.loc[index,cvlt_trial] = input_data.loc[index,cvlt_trial] + cvlt_index_unit
                    cvlt_index_diff-= cvlt_index_unit
                    if cvlt_index_diff == 0:
                        break                

        ui.notification_show("Converted HVLT !",duration=3,close_button=True,type="message")

    return input_data