import io
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
    ui.tags.h3("PTSD Severity Screener Conversion Tool", class_="app-heading"),
    ui.tags.div(
        {"class" : "file"},
        ui.input_file("file1", "Choose a file to upload for conversion. Please see template for column names.", accept=[".csv",".xlsx"],multiple=False),
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

    @session.download(filename="ptsd-template.xlsx")
    def download_template():
        return os.path.join(path,'ptsd-template.xlsx')

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
cols = ['PCL5_total','PCLC_total','PCLM_total','DTS_total','MPSS_total']

min_values = {
    "PCL5_total": 0,
    "PCLC_total": 17,
    "PCLM_total": 17,
    "DTS_total": 0,
    "MPSS_total": 0,
}
max_values = {
    "PCL5_total": 80,
    "PCLC_total": 85,
    "PCLM_total": 85,
    "DTS_total": 68,
    "MPSS_total": 51,
}

def replace_in_csv(input_data,measures_data,input_col,index,target_col,target_col_str):
    if not np.isnan(input_data.loc[index,input_col]):
        if (input_data.loc[index,input_col] <= max_values[input_col] and input_data.loc[index,input_col] >= min_values[input_col]):
            input_data.loc[index,target_col] = measures_data.loc[int(input_data.loc[index,input_col]),target_col_str]

def conversion(input_data: pd.DataFrame) -> pd.DataFrame:
    input_cols = set(input_data.columns)
    # Only runs when intersection is 0.
    if (len(set(cols) & input_cols) == 0):
        ui.notification_show(f"Couldn't find correct column names, please use template for data upload ",duration=8,close_button=True,type="error")
        return
    for col in cols:
        if col not in input_data.columns:
            input_data[col] = np.nan

    notnull_pcl5_sum = input_data['PCL5_total'].notnull().sum()
    notnull_pclc_sum = input_data['PCLC_total'].notnull().sum()
    notnull_pclm_sum = input_data['PCLM_total'].notnull().sum()
    notnull_dts_sum = input_data['DTS_total'].notnull().sum()
    notnull_mpss_sum = input_data['MPSS_total'].notnull().sum()

    input_col_list = ["PCL5_total","PCLC_total","PCLM_total","DTS_total","MPSS_total"]
    # input_col_list = ['PCL5_total','PCLC_total', 'PCLM_total','DTS_total','MPSS_total']
    max_ind = np.array([notnull_pcl5_sum, notnull_pclc_sum,notnull_pclm_sum,notnull_dts_sum,notnull_mpss_sum]).argmax()
    input_col = input_col_list[max_ind]
    if input_col not in input_cols:
        ui.notification_show(f"Couldn't find {input_col}, please use template for data upload ",duration=8,close_button=True,type="error")
        return

    if input_col == "PCL5_total":
        pcl_5 = pd.read_csv(os.path.join(path,'pcl-5.csv'),index_col=[0])
        convert_cols = [x for x in cols if x != 'PCL5_total']
        input_data.loc[:,convert_cols] = input_data.loc[:,convert_cols].astype('object')
        for i in input_data.index:
            replace_in_csv(input_data,pcl_5,'PCL5_total',i,'PCLC_total','PCL-C')
            replace_in_csv(input_data,pcl_5,'PCL5_total',i,'PCLM_total','PCL-M')
            replace_in_csv(input_data,pcl_5,'PCL5_total',i,'DTS_total','DTS')
            replace_in_csv(input_data,pcl_5,'PCL5_total',i,'MPSS_total','MPSS')

        ui.notification_show("Converted PCL5!",duration=3,close_button=True,type="message")

    elif input_col == "PCLC_total":
        pcl_c = pd.read_csv(os.path.join(path,'pcl-c.csv'),index_col=[0])
        convert_cols = [x for x in cols if x != 'PCLC_total']
        input_data.loc[:,convert_cols] = input_data.loc[:,convert_cols].astype('object')
        for i in input_data.index:
            replace_in_csv(input_data,pcl_c,'PCLC_total',i,'PCL5_total','PCL-5')
            replace_in_csv(input_data,pcl_c,'PCLC_total',i,'PCLM_total','PCL-M')
            replace_in_csv(input_data,pcl_c,'PCLC_total',i,'DTS_total','DTS')
            replace_in_csv(input_data,pcl_c,'PCLC_total',i,'MPSS_total','MPSS')

        ui.notification_show("Converted PCLC!",duration=3,close_button=True,type="message")
    
    elif input_col == "PCLM_total":
        pcl_m = pd.read_csv(os.path.join(path,'pcl-m.csv'),index_col=[0])
        convert_cols = [x for x in cols if x != 'PCLM_total']
        input_data.loc[:,convert_cols] = input_data.loc[:,convert_cols].astype('object')
        for i in input_data.index:
            replace_in_csv(input_data,pcl_m,'PCLM_total',i,'PCL5_total','PCL-5')
            replace_in_csv(input_data,pcl_m,'PCLM_total',i,'PCLC_total','PCL-C')
            replace_in_csv(input_data,pcl_m,'PCLM_total',i,'DTS_total','DTS')
            replace_in_csv(input_data,pcl_m,'PCLM_total',i,'MPSS_total','MPSS')

        ui.notification_show("Converted PCLM!",duration=3,close_button=True,type="message")
    elif input_col == "DTS_total":
        dts = pd.read_csv(os.path.join(path,'dts.csv'),index_col=[0])
        convert_cols = [x for x in cols if x != 'DTS_total']
        input_data.loc[:,convert_cols] = input_data.loc[:,convert_cols].astype('object')
        for i in input_data.index:
            replace_in_csv(input_data,dts,'DTS_total',i,'PCL5_total','PCL-5')
            replace_in_csv(input_data,dts,'DTS_total',i,'PCLC_total','PCL-C')
            replace_in_csv(input_data,dts,'DTS_total',i,'PCLM_total','PCL-M')
            replace_in_csv(input_data,dts,'DTS_total',i,'MPSS_total','MPSS')

        ui.notification_show("Converted DTS!",duration=3,close_button=True,type="message")
    elif input_col == "MPSS_total":
        mpss = pd.read_csv(os.path.join(path,'mpss.csv'),index_col=[0])
        convert_cols = [x for x in cols if x != 'MPSS_total']
        input_data.loc[:,convert_cols] = input_data.loc[:,convert_cols].astype('object')
        for i in input_data.index:
            replace_in_csv(input_data,mpss,'MPSS_total',i,'PCL5_total','PCL-5')
            replace_in_csv(input_data,mpss,'MPSS_total',i,'PCLC_total','PCL-C')
            replace_in_csv(input_data,mpss,'MPSS_total',i,'PCLM_total','PCL-M')
            replace_in_csv(input_data,mpss,'MPSS_total',i,'DTS_total','DTS')

        ui.notification_show("Converted MPSS!",duration=3,close_button=True,type="message")

    return input_data