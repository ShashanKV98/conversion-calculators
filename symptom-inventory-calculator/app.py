import io
from typing import List
import os
from pathlib import Path
from shiny import App,reactive, render, ui,Session
import pandas as pd
import numpy as np
from crosswalk_symptom_inventories import set_crosswalk_files,crosswalk_scores
from collections import OrderedDict

css_path = "styles.css"
app_ui = ui.page_fluid(
#    ui.panel_title("Hello Shiny!"),
    {'class' : 'container'},
    ui.include_css(css_path),
    # shinyswatch.theme.darkly(),
    ui.tags.h3("Symptom inventories Calculator", class_="app-heading"),
    ui.tags.div(
        {"class" : "file"},
        ui.tags.div(
       ui.input_select(
           "input_name",
           "Symptom Inventory Input",
           ["BSI","RPQ","SCL","NSI"],
       ),
       class_="select_dropdown"
        ),
    ui.tags.div(
       ui.input_select(
           "output_name",
           "Symptom Inventory Output",
           ["RPQ","BSI","SCL","NSI"],
           
       ),
       class_ = "select_dropdown"
    ),
        ui.input_action_button("convert","Convert table"),
        ui.download_button("download_conversion", "Download Conversion"),
    ),
    ui.tags.div(
        {"class" : "tables"},
    
        ui.tags.div(
        {"class" : "table_parent"},
        ui.output_ui("input_table"),
        ),
        ui.tags.div(
        {"class" : "table_parent"},
        ui.output_ui("output_table")
        ),
        # ui.tags.div(
        # {"class" : "table_parent"},
        # ui.output_table("output_table")
        # ),
    )
    
)

def server(input, output, session):
    num_rows = reactive.Value(0)
    @output
    @render.ui
    @reactive.event(input.input_name)
    def input_table():
        inv_input = input.input_name()
        if not inv_input:
            return
        input_rows = get_input_table(inv_input)
        count = 0
        table_rows = []
        for category, group in input_rows.items():
            table_rows.append(ui.tags.tr(ui.tags.td(category),class_="table_category"))
            for index, row in enumerate(group): 
                row_cells = [
                    ui.tags.td(row[1]),
                    ui.tags.td(
                        ui.input_numeric(f"number_{count+1}",label="",value="",min=0,max=4),
                        class_= "row_numbers"
                    )
                ]
                count+= 1

                table_rows.append(ui.tags.tr(*row_cells,class_="table_row"))
        num_rows.set(count)
        return ui.tags.table(
            ui.tags.legend(
                ui.tags.span(f"{inv_input} table",class_="table_title"),
                ui.tags.br(class_="br"),
                ui.tags.span("Enter integer values between 0 and 4",class_="table_instr")
                ,class_="table_heading"
            ),
            ui.tags.tbody(
                *table_rows,
            )
        )

    @output
    @render.ui
    @reactive.event(input.convert)
    def output_table():
        inv_input = input.input_name()
        inv_output = input.output_name()

        scores = [input[f"number_{i+1}"]() for i in range(num_rows.get())]

        sum_score_str = ''
        if inv_output == "BSI":
            sum_score_str = "Raw sum / 72"
        elif inv_output == "RPQ":
            sum_score_str = "Raw sum / 64"
        elif inv_output == "NSI":
            sum_score_str = "Raw sum / 88"
        elif inv_output == "SCL":
            sum_score_str = "Raw sum / 360"
        # Checking the input numbers are within the range (0,4)
        for score in scores:
            if score is None:
                ui.notification_show(f"Please fill out scores for all symptoms",duration=5,close_button=True,type="error")
                return
            if score < 0 or score > 4:
                ui.notification_show(f"Please enter scores between 0 and 4 ",duration=5,close_button=True,type="error")
                return
        
        # Checking if the output measure is different from input measure
        if inv_input == inv_output:
            ui.notification_show(f"Please select a different output measure",duration=5,close_button=True,type="error")
            return
        
        outdict,scores_sum = convert(inv_input,inv_output,scores)
        table_rows = []
        for category, group in outdict.items():
            table_rows.append(ui.tags.tr(ui.tags.td(category),class_="table_category"))
            for score, title in group: 
                row_cells = [
                    ui.tags.tr(
                    ui.tags.td(title),
                    ui.tags.td(score),
                    class_="output_table_row"
                    )
                ]

                table_rows.append(ui.tags.tr(*row_cells,class_="table_row"))
        table_rows.append(ui.tags.tr(
            ui.tags.td(f"{sum_score_str}"),
            ui.tags.td(scores_sum),
            class_="total_score_row"
        ))
        return ui.tags.table(
            ui.tags.legend(
                ui.tags.span(f"{inv_output} table",class_="table_title"),
                ui.tags.br(),
                class_="table_heading"
                ),
                ui.tags.div(
                ui.tags.span("Output symptoms",class_="table_instr"),
                ui.tags.span("Estimated scores",class_="table_instr"),
                class_="output_captions"
                ),
                ui.tags.br(class_="br"),
                ui.tags.tbody(
                *table_rows,
                )
            ),

    @session.download(filename="converted_table.csv")
    def download_conversion():
        inv_input = input.input_name()
        inv_output = input.output_name()
        scores = [input[f"number_{i+1}"]() for i in range(num_rows.get())]
        sum_score_str = ''
        if inv_output == "BSI":
            sum_score_str = "Raw sum / 72"
        elif inv_output == "RPQ":
            sum_score_str = "Raw sum / 64"
        elif inv_output == "NSI":
            sum_score_str = "Raw sum / 88"
        elif inv_output == "SCL":
            sum_score_str = "Raw sum / 360"
        # Checking the input numbers are within the range (0,4)
        for score in scores:
            if score is None:
                ui.notification_show(f"Please fill out scores for all symptoms",duration=5,close_button=True,type="error")
                return
            if score < 0 or score > 4:
                ui.notification_show(f"Please enter scores between 0 and 4 ",duration=5,close_button=True,type="error")
                return
        
        # Checking if the output measure is different from input measure
        if inv_input == inv_output:
            ui.notification_show(f"Please select a different output measure",duration=5,close_button=True,type="error")
            return
        
        outdict,scores_sum = convert(inv_input,inv_output,scores)
        if not outdict:
            ui.notification_show(f"No conversion to download ",duration=5,close_button=True,type="error")
            return
        df = pd.DataFrame()
        for category, group in outdict.items():
            for score, title in group:
                df.loc['scores',title] = score
        df.loc['scores',f'{sum_score_str}'] = scores_sum
        csv_file = df.to_csv()

        try:
            return io.BytesIO(csv_file.encode("utf-8"))
        except Exception as e:
            return io.BytesIO(str(e).encode("utf-8"))

app = App(app_ui, server)

def convert(inv_input,inv_output,scores):

    input_rows = np.asarray([])
    output_rows = np.asarray([])

    input_rows,output_rows = get_table(inv_input,inv_output)
    output_zip = []
    scores = list(map(int,scores))
    
    A = score_conversion(inv_input,inv_output)

    input_text_rows = A.text_dict[inv_input]
    output_text_rows = A.text_dict[inv_output]
    input_row_values = []
    output_row_values = []
    for key in input_rows.keys():
        input_row_values.extend(input_rows[key])
    for key in output_rows.keys():
        output_row_values.extend(output_rows[key])

    order = []
    for i in range(len(input_row_values)):
        order.append(input_text_rows.index(input_row_values[i][1]))
    reorder_scores = [x for _,x in sorted(zip(order,scores))]
    predicted_scores = crosswalk_scores(
                        input_scores = reorder_scores,
                        score_dict = A.score_dict,
                        text_dict = A.text_dict,
                        hist_dict = A.hist_dict,
                        simil_arr = A.simil_arr,
                        empirical_shift_down = False,
                        inv_in = inv_input,
                        inv_out = inv_output,
                        verbose= False,
                        link_hists=True,
                        random_seed = 42,
                        )
    order= []
    for i in range(len(output_text_rows)):
        order.append(output_row_values.index(output_text_rows[i]))
    
    predicted_scores = [int(i) for i in predicted_scores]
    final_scores = [x for _,x in sorted(zip(order,predicted_scores))]
    count = 0
    outdict = output_rows.copy()
    for key in outdict.keys():
        outdict[key] = list(zip(final_scores[count:len(outdict[key])+count],outdict[key]))
        count+= len(outdict[key])
    scores_sum = int(np.sum(predicted_scores))
    return outdict,scores_sum

def get_input_table(input):
    groups_path = "groups.p"
    groups = pd.read_pickle(groups_path)
    groups = OrderedDict(sorted(groups.items()))
    input_titles ={}
    inp_count=1
    for key in groups.keys():
        if input in key:
            count_arr = [int(i) for i in range(inp_count,len(groups[key])+inp_count)]
            input_titles[key[1]] = list(zip(count_arr,groups[key]))
            inp_count+=len(groups[key])
    return input_titles

def get_table(input,output):
    groups_path = "groups.p"
    groups = pd.read_pickle(groups_path)
    groups = OrderedDict(sorted(groups.items()))
    input_titles ={}
    output_titles={}
    inp_count=1
    for key in groups.keys():
        if input in key:
            count_arr = [int(i) for i in range(inp_count,len(groups[key])+inp_count)]
            input_titles[key[1]] = list(zip(count_arr,groups[key]))
            inp_count+=len(groups[key])
        if output in key:
            # output_titles.append((key[1],groups[key]))
            output_titles[key[1]] = groups[key]

    return input_titles,output_titles

def score_conversion(inventory_in,inventory_out):
    score_dict_path = "score_dict.p"
    text_dict_path = "text_dict.p"
    hist_dict_path = "hist_dict.p"
    A = set_crosswalk_files( 
                            score_file = score_dict_path,
                            text_file = text_dict_path,
                            hist_file = hist_dict_path,
                            inv_in = inventory_in,
                            inv_out = inventory_out,
                            )
                            
    return A
