import copy
import os

from utils import generate_color_map, return_generated_figure, get_legend_box, save_as_pdf_allbox, generate_dataset
from utils import  generate_dataset_umap, save_as_pdf_hbox
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import pickle
import numpy as np
import copy

data_dir = Path("/Users/dadua2/projects")
# data_dir = Path("/data/CARD/projects/ImagingBasedProgressionGWAS/projects")
scale = 4
combined_best_parameters = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16'],
    # 'PPMI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16'],
    'ADNI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], 'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16'],
    # 'ALVEOLAR_metacelltype':  [[0, -1.25, -1.25], 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=05;min_dist=0.10;num_cores=16'],
    'COVID19Proteomics': [[1.25, -1.25, -1.25], 'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=05;min_dist=0.01;num_cores=16'],
}

time_ids = ['t000', 't010', 't020', 't030', 't040', 't050']
def f(x):
            for e, i in enumerate(time_ids):
                if e == 0:
                    continue
                if i > x:
                    return time_ids[e-1]
            return time_ids[-1]

all_rows = [ 'R0', 'R1', 'R2', 'R3', 'R5', 'R6' ]
all_rows = ['R0']
ALLPLOTS = {
    # Done
    'R0': {
            'C0': ['PPMI_FOR_ALIGNED_TIME_SERIES', 'umap', 'metric=euclidean;n_neighbors=10;min_dist=0.01;num_cores=16', [0, 0, 0]],
            'C1': ['PPMI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [0, -1.25, -1.25] ],
            'C2': ['PPMI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [-1.25, 1.25, 1.25] ],
    },
    # Done
    'R1': {
            'C0': ['ADNI_FOR_ALIGNED_TIME_SERIES', 'umap', 'metric=cosine;n_neighbors=05;min_dist=0.10;num_cores=16', [0, 0, 0]],
            'C1': ['ADNI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [0, -1, -1] ],
            'C2': ['ADNI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [-1, 1, 1] ],
    },
    # Done
    'R2': {
            'C0': ['NORM-ALVEOLAR_metacelltype', 'umap', 'metric=cosine;n_neighbors=05;min_dist=0.10;num_cores=16', [0, 0, 0]],
            'C1': ['NORM-ALVEOLAR_metacelltype', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=05;min_dist=0.01;num_cores=16', [0, -1.25, -1.25] ],
            'C2': ['NORM-ALVEOLAR_metacelltype', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=05;min_dist=0.01;num_cores=16', [1.25, -1.25, 1.25] ],
    },
    # Done
    'R3': {
            'C0': ['IMAGENORM-PPMI-ADNI', 'umap', 'metric=cosine;n_neighbors=05;min_dist=0.10;num_cores=16', [0, 0, 0]],
            'C1': ['IMAGENORM-PPMI-ADNI', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [0, -1.25, 1.25] ],
            'C2': ['IMAGENORM-PPMI-ADNI', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [1.25, -1.25, 1.25] ],
    },
    # Done Camera Remaining
    'R4': {
            'C0': ['MINMAX_MIMIC_ALLSAMPLES', 'umap', 'metric=euclidean;n_neighbors=05;min_dist=0.01;num_cores=16', [0, 0, 0]],
            'C1': ['MINMAX_MIMIC_ALLSAMPLES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.01;num_cores=16', [0, -1.25, -1.25] ],
            'C2': ['MINMAX_MIMIC_ALLSAMPLES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.01;num_cores=16', [1.25, -1.25, 1.25] ],
    },
    # Done
    'R5': {
            'C0': ['AllN2_BioReactor_2D_3D_Report_protein', 'umap', 'metric=cosine;n_neighbors=10;min_dist=0.01;num_cores=16', [0, 0, 0]],
            'C1': ['AllN2_BioReactor_2D_3D_Report_protein', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=03;min_dist=0.01;num_cores=16', [0, -1.25, -1.25] ],
            'C2': ['AllN2_BioReactor_2D_3D_Report_protein', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=03;min_dist=0.01;num_cores=16', [1.25, -1.25, 1.25] ],
    },
    # Done
    'R6': {
            'C0': ['NORM-COVID19Proteomics', 'umap', 'metric=euclidean;n_neighbors=05;min_dist=0.01;num_cores=16', [0, 0, 0]],
            'C1': ['NORM-COVID19Proteomics', 'umap_aligned', 'metric=cosine;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [0, -1.25, 1.25] ],
            'C2': ['NORM-COVID19Proteomics', 'umap_aligned', 'metric=cosine;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [1.25, -1.25, 1.25] ],
    },

}

SPECS = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': {
        'dataset_name': "Parkinson's Disease Progression",
        'colorname': 'Subtypes',
        'num_subjects': 1,
        'selected_cats': ['PD_h', 'HC'],
        'mapping_legend': {'PD_h': 'High Parkinson Disease', 'HC': 'Healthy Controls'},
        'annotation_legend': {'PD_h': 'High PD', 'HC': 'Healthy Controls'},
        'time_mapping': ['Years from baseline', {'t0':'Y0', 't1':'Y1', 't2':'Y2', 't3':'Y3', 't4':'Y4', 't5':'Year5',
                                                 't00':'Y0', 't01':'Y1', 't02':'Y2', 't03':'Y3', 't04':'Y4', 't05':'Year5'}],
        'alpha': 1
    },
    'ADNI_FOR_ALIGNED_TIME_SERIES':  {
        'dataset_name': "Alzheimer's Disease Progression",
        'colorname': 'Subtypes',
        'num_subjects': 1,
        'selected_cats': ['Dementia', 'Control'],
        'mapping_legend': {'Control': 'Healthy Controls', 'Dementia': 'Dementia'},
        'annotation_legend': {'Dementia': 'Dementia', 'Control': 'Healthy Controls'},
        'time_mapping': ['Months from baseline', {'t0': 'M0', 't1': 'M6', 't2': 'M12', 't3': 'Month24',
                                                   'bl': 'M0', 'm06': 'M6', 'm12': 'M12', 'm24': 'Month24'}],
        'alpha': 1
    },
    'NORM-ALVEOLAR_metacelltype':  {
        'dataset_name': "whole lung scRNA-seq",
        'colorname': 'cell.type',
        'num_subjects': 1,
        'selected_cats': ['AT2 cells', 'Mesothelial cells', 'T-lymphocytes'],
        'mapping_legend': {},
        'annotation_legend': {},
        'time_mapping': ['Days after bleomycin injury',
                         {
                             't0':'d0', 't1':'d3', 't2':'d7', 't3':'d10', 't4':'d14', 't5':'d21', 't6': 'day28',
                            't00':'d0', 't01':'d3', 't02':'d3', 't03':'d7', 't04':'d10', 't05':'d14', 't06': 'day21'
                          }],
        'alpha': 1
    },
    'IMAGENORM-PPMI-ADNI':  {
        'dataset_name': "Brain T1-MRI imaging for AD and PD",
        'colorname': 'GENDER_DIAGNOSIS',
        # 'colorname': 'DIAGNOSIS',
        'num_subjects': 1,
        'selected_cats': ['M_Dementia', 'F_Dementia', 'M_PD', 'F_PD'],
        # 'selected_cats': ['Dementia', 'Control', 'PD'],
        # 'selected_cats': ['F_Dementia', 'F_Control', 'F_PD'],
        'mapping_legend': {},
        'annotation_legend': {},
        'time_mapping': ['Age', { }],
        'alpha': 1
    },
    # 'MINMAX_MIMIC_ALLSAMPLES':  {
    #     'dataset_name': "MIMIC III EHR data",
    #     'colorname': 'LAST_CAREUNIT',
    #     'num_subjects': 0.3,
    #     'selected_cats': ['MICU', 'CSRU'],
    #     'mapping_legend': {'MICU': 'Medical Intensive ICU (Female)', 'CSRU': 'Cardiac Surgery Recovery Unit (Female)'},
    #     'annotation_legend': {'MICU': 'MICU (F)', 'CSRU': 'CSRU (F)'},
    #     'time_mapping': ['Hours after ICU admission',
    #                      {
    #                          't0':'H0', 't1':'H12', 't2':'H24', 't3':'H36', 't4':'H48', 't5':'H60', 't6': 'H72',
    #                         'H0000':'H0', 'H0001':'H12', 'H0002':'H24', 'H0003':'H36', 'H0004':'H48', 'H0005':'H60', 'H0006': 'H72'
    #                      }
    #             ],
    #     'alpha': 1
    # },
    'MINMAX_MIMIC_ALLSAMPLES':  {
        'dataset_name': "MIMIC III EHR data",
        'colorname': 'LAST_CAREUNIT',
        'num_subjects': 0.3,
        'selected_cats': ['MICU', 'CSRU'],
        'mapping_legend': {'MICU': 'Medical Intensive ICU (Male)', 'CSRU': 'Cardiac Surgery Recovery Unit (Male)'},
        'annotation_legend': {'MICU': 'MICU (M)', 'CSRU': 'CSRU (M)'},
        'time_mapping': ['Hours after ICU admission',
                         {
                             't0':'H0', 't1':'H12', 't2':'H24', 't3':'H36', 't4':'H48', 't5':'H60', 't6': 'Hour72',
                            'H0000':'H0', 'H0001':'H12', 'H0002':'H24', 'H0003':'H36', 'H0004':'H48', 'H0005':'H60', 'H0006': 'Hour72'
                         }
                ],
        'alpha': 1
    },
    'NORM-COVID19Proteomics':  {
        'dataset_name': "Proteomics COVID19 data",
        'colorname': 'Acuity_max',
        'num_subjects': 1,
        'selected_cats': [], # 'Severe', 'Non-severe'
        'mapping_legend': {},# 2: 'Severe', 3: 'Non-severe'},
        'annotation_legend': {}, # 2: 'Severe', 3: 'Non-severe'},
        'manual_label_mapping': {1: 'Severe',2:'Severe',3:'Non-severe',4:'Non-severe',5:'Non-severe'},
        'time_mapping': ['Days after COVID Infection',
                         {
                             't0':'d0', 't1':'d3', 't2':'day7',
                             't0':'d0', 't3':'d3', 't7':'day7',
                         }
                ],
        'alpha': 1
    },
    'AllN2_BioReactor_2D_3D_Report_protein':  {
        'dataset_name': "Proteomics Stem Cell data (different culture)",
        'colorname': 'label',
        'num_subjects': 1,
        'selected_cats': [],
        'mapping_legend': {},
        'annotation_legend': {},
        'time_mapping': ['Days',
                         {
                            't0':'d0', 't1':'d3', 't2':'d7', 't3':'d14', 't4':'d21', 't5':'day28', 't6': 'day28',
                            't00':'d0', 't01':'d3', 't02':'d7', 't03':'d14', 't04':'d21', 't05':'day28', 't06': 'day28'
                         }
                ],
        'alpha': 1
    },
}







import plotly.graph_objects as go
import plotly
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from PIL import Image as pilim
from PIL import Image, ImageDraw
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from collections import defaultdict

def legend_title(annotation, colorlist, title1):
    # colorlist = ['red', 'blue', 'red', 'red']#, 'red', 'red', 'red', 'blue']
    # annotation = ['d0'] *  num
    fig = go.Figure()
    # colorlist = ['red'] * len(annotation)
    # Add traces
    # fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker_color=df['c'],marker_size=20, name='markers'))
    df = pd.DataFrame({'x': list(np.linspace(0,10,len(annotation))), 'y': [0]*len(annotation), 'annot': annotation})
    df  = df.replace({'A93': 'Age93'})
    fig = px.scatter(df, x="x", y="y", color='annot', template='plotly_white', text='annot', color_discrete_map=colorlist, opacity=1, range_x=[-0.5, 11.5])
    for i in list(df['x'])[:-1]:
        fig.add_annotation(
           x = i+df.iloc[1]['x'] - df.iloc[0]['x'] - 0.3, # arrows ' headn* ( len(colorlist) / 5)
           y = 0, # arrows ' head
           ax = i+0.3, # arrows ' tail  ( len(colorlist) / 5)
           ay = 0, # arrows ' tail
           xref = 'x',
           yref = 'y',
           axref = 'x',
           ayref = 'y',
           text = '', #
           arrowhead = 2,
           arrowsize = 2,
           arrowwidth = 1,
           arrowcolor = 'black'
        )
    # fig.add_annotation(x=6.5, y=0, text=title1, showarrow=False, yshift=25, xshift=0, font=dict(size=15))
    # fig.add_annotation(x=10, y=0, text=title2, showarrow=False, yshift=60, xshift=0)
    fig.update_traces(textposition='bottom center')
    fig.update_traces(marker={'size': 28})
    fig.update_layout(
        showlegend=False,
        # height=200, # 300
        # width=600, #520
        autosize=False,  # True if camera_setup_ppmi else False,
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        font=dict(
            size=24,
            color="black"
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return plotly.io.to_image(fig, format='png', scale=scale)

def get_concat_h(im1, im2, color=(255, 255, 255)):
    dst = pilim.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2, color=(255, 255, 255)):
    dst = pilim.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def fillinwhite(im, height, width, pastebottom=False):
    dst = pilim.new('RGB', (width, height), 'white')
    if pastebottom:
        dst.paste(im, (width-im.width, height-im.height))
    else:
        dst.paste(im, (0, 0))
    return dst

from utils import trim_image

MAINFIG = []

all_columns = ['C0' , 'C1', 'C2']
for row in all_rows:
    all_figs = []
    all_titles = []
    all_legends = []
    color_map_list = None
    for col in all_columns:
        input_name = ALLPLOTS[row][col]
        print (input_name, row, col)
        input_dataset_name = input_name[0]
        visualization_type = input_name[1]
        parameter_name = input_name[2]
        eye_view = input_name[3]
        colorname = SPECS[input_dataset_name]['colorname']
        selected_cats = SPECS[input_dataset_name]['selected_cats']
        mapping_legend = SPECS[input_dataset_name]['mapping_legend']
        annotation_legend = SPECS[input_dataset_name]['annotation_legend']
        num_subjects = SPECS[input_dataset_name]['num_subjects']
        alpha = SPECS[input_dataset_name]['alpha']
        time_mapping = SPECS[input_dataset_name]['time_mapping']
        dataset_name = SPECS[input_dataset_name]['dataset_name']
        manual_label_mapping = {}
        if 'manual_label_mapping' in SPECS[input_dataset_name]:
            manual_label_mapping = SPECS[input_dataset_name]['manual_label_mapping']




        if visualization_type == 'umap':
            fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap/generated_data/{input_dataset_name}_16.pickle"
            with open(fpath, 'rb') as handle:
                results_data = pickle.load(handle)
            if input_dataset_name == "IMAGENORM-PPMI-ADNI":
                results_data['complete_dataframe']['GENDER_DIAGNOSIS'] = results_data['complete_dataframe']['GENDER'] + '_' +  results_data['complete_dataframe']['DIAGNOSIS']
                results_data['complete_dataframe']['time_id'] = results_data['complete_dataframe']['time_id'].map(lambda x: f(x))
                x = results_data['complete_dataframe'].groupby('time_id').agg({'AGE': 'mean'}).round().astype(int)
                time_mapping[1].update(dict(zip(list(x.index), list(x['AGE'].map(lambda x: f'A{x}')))))
                time_mapping[1].update(dict(zip(list(map(lambda x: f't{x}', list(x.index.map(lambda x: int(x.split('t')[-1]))))), list(x['AGE'].map(lambda x: f'A{x}')))))
            if input_dataset_name == "MINMAX_MIMIC_ALLSAMPLES":
                results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER'].isin(['M'])]

            print (time_mapping)
            all_samples = len(results_data['complete_dataframe'].subject_id.unique())
            if len(selected_cats) > 0:
                results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
            else:
                pass
            results_data['complete_dataframe'][colorname] = results_data['complete_dataframe'][colorname].map(lambda x: manual_label_mapping.get(x, x))
            fig, (color_maps, nocolor_maps) = generate_dataset_umap(results_data, input_dataset_name, parameter_name, eye_view,
                                                                    num_subjects, selected_cats, colorname, scale=scale,
                                                                    alpha=alpha, transparency_effect=False, time_mapping=time_mapping
                                                                    , mapping_legend=annotation_legend)
            all_figs.append(fig)
            all_legends.append(Image.open(get_legend_box(nocolor_maps, [], scale=1, fontsize=15, mapping_legend=mapping_legend, ncols=4,type='circle')))
            all_titles.append(f'{input_dataset_name}')
            color_map_line = {}
            for key, value in color_maps.items():
                color_map_line[time_mapping[1].get(key, key)] = value
        else:
            fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
            with open(fpath, 'rb') as handle:
                results_data = pickle.load(handle)
            if input_dataset_name == "IMAGENORM-PPMI-ADNI":
                results_data['complete_dataframe']['GENDER_DIAGNOSIS'] = results_data['complete_dataframe']['GENDER'] + '_' +  results_data['complete_dataframe']['DIAGNOSIS']
            if input_dataset_name == "MINMAX_MIMIC_ALLSAMPLES":
                results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER'].isin(['M'])]
            if len(selected_cats) > 0:
                results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
            else:
                pass
            results_data['complete_dataframe'][colorname] = results_data['complete_dataframe'][colorname].map(lambda x: manual_label_mapping.get(x, x))
            print (time_mapping)
            # x = results_data['complete_dataframe'].groupby('time_id').agg({'AGE': 'mean'}).round().astype(int)
            # time_mapping[1] = dict(zip(list(x.index), list(x['AGE'].map(lambda x: f'A{x}'))))

            fig, (color_maps, nocolor_maps) = generate_dataset(results_data, input_dataset_name, parameter_name, eye_view , num_subjects, selected_cats, colorname, scale=scale, alpha=alpha, transparency_effect=True, time_mapping=time_mapping)
            all_figs.append(fig)
            all_legends.append(Image.open(get_legend_box(nocolor_maps, selected_cats, scale=scale, fontsize=18, mapping_legend=mapping_legend)))
            all_titles.append(f"{input_dataset_name}")


    x = legend_title(annotation=list(color_map_line.keys()), colorlist=color_map_line, title1=dataset_name)
    img1 = trim_image(x)
    basewidth = 400 * scale
    wpercent = (basewidth/float(img1.size[0]))
    hsize = int((float(img1.size[1])*float(wpercent)))
    img1 = img1.resize((basewidth,hsize), Image.ANTIALIAS)


    x = all_figs[0]
    img2 = trim_image(x)
    basewidth = 400 * scale
    wpercent = (basewidth/float(img2.size[0]))
    hsize = int((float(img2.size[1])*float(wpercent)))
    img2 = img2.resize((basewidth,hsize), Image.ANTIALIAS)

    x = all_figs[1]
    img3 = trim_image(x)
    basewidth = 400 * scale
    wpercent = (basewidth/float(img3.size[0]))
    hsize = int((float(img3.size[1])*float(wpercent)))
    img3 = img3.resize((basewidth,hsize), Image.ANTIALIAS)


    x = all_figs[2]
    img4 = trim_image(x)
    basewidth = 400 * scale
    wpercent = (basewidth/float(img4.size[0]))
    hsize = int((float(img4.size[1])*float(wpercent)))
    img4 = img4.resize((basewidth,hsize), Image.ANTIALIAS)


    import io
    img_byte_arr = io.BytesIO()
    all_legends[-1].save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img5 = trim_image(img_byte_arr)
    basewidth = 400 * scale
    wpercent = (basewidth/float(img5.size[0]))
    hsize = int((float(img5.size[1])*float(wpercent)))
    img5 = img5.resize((basewidth,hsize), Image.ANTIALIAS)




    os.makedirs(f'allimages/{row}', exist_ok=True)
    img1.save(f'allimages/{row}/img1.png')
    img2.save(f'allimages/{row}/img2.png')
    img3.save(f'allimages/{row}/img3.png')
    img4.save(f'allimages/{row}/img4.png')
    img5.save(f'allimages/{row}/img5.png')

    print (img1.size)
    print (img2.size)
    print (img3.size)
    print (img4.size)
    print (img5.size)

import sys; sys.exit()



img12 = Image.new('RGB', (200*scale, 100*scale), color = (255, 255, 255))
draw = ImageDraw.Draw(img12)
font = ImageFont.truetype("arial.ttf", 16 * scale)
draw.text((2*scale, 40*scale), "Sample Count = {}".format(all_samples), (0, 0, 0), font=font)




x = all_figs[1]
v = trim_image(x)
# img22 = fillinwhite(trim_image(x), 400*scale, 500*scale)
img22 = trim_image(x)
img22.resize((400*scale, 500*scale), Image.ANTIALIAS)

x = all_figs[2]
# img23 = fillinwhite(trim_image(x), 400*scale, 500*scale)
img23 = trim_image(x)
img23.resize((400*scale, 500*scale), Image.ANTIALIAS)


import io
img_byte_arr = io.BytesIO()
all_legends[-1].save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()
img13 = fillinwhite(trim_image(img_byte_arr), 100*scale, 800*scale, pastebottom=True)


print (img11.width, img11.height)
print (img12.width, img12.height)
print (img21.width, img21.height)
print (img22.width, img22.height)
print (img23.width, img23.height)
print (img13.width, img13.height)

from utils import concat_horizontal, concat_vertical
img1 = concat_horizontal([img11, img12, img13])
img2 = concat_horizontal([img21, img22, img23])
img = concat_vertical([img1, img2])
MAINFIG.append(img)


# MAINIMG = concat_vertical(MAINFIG)
# MAINIMG.save(f"FINALMAINIMAGE.pdf", quality=100,)
# img.show()
# import sys; sys.exit()
# c1 = save_as_pdf_hbox(all_figs, all_titles, f'../final_images/SingleView_alv.pdf', ncols=3, scale=3, all_legends=all_legends)


# figure_details = {
#     'R0': {
#         'columns':
#             {
#                 'C0': {
#                         'figure': c1,
#                         'legend': None,
#                         'title': 'UMAP'
#                     }
#                 },
#         'row_title': "R0",
#     }
# }
#
# from utils import trim_image
# x = trim_image(all_figs[0])
#
#
# ncols = 3
# nrows = 4
#
# total_rows = len(all_figs) // ncols if len(all_figs) % ncols == 0 else 1 + len(all_figs) // ncols
# Z = [[0 for i in range(ncols)] for j in range(total_rows)] # [[0]*ncols] * total_rows
#
# for i in tqdm(range(0, len(all_figs))):
#     Z [i//ncols][i % ncols] = trim_image(all_figs[i])
#
# num_pages = total_rows // nrows if total_rows % nrows == 0 else 1 + total_rows // nrows
# pagemapping = defaultdict(list)
# for i in range(len(Z)):
#         if all_legends is None:
#             concated_image = concat_horizontal(Z[i])
#         else:
#             img_byte_arr = io.BytesIO()
#             all_legends[i].save(img_byte_arr, format='PNG')
#             img_byte_arr = img_byte_arr.getvalue()
#             concated_image = concat_vertical([concat_horizontal(Z[i]), trim_image(img_byte_arr, whitespace=20)])
#         # concated_image = concat_horizontal(Z[i])
#         img_byte_arr = io.BytesIO()
#         concated_image.save(img_byte_arr, format='PNG')
#         img_byte_arr = img_byte_arr.getvalue()
#         pagemapping[i%num_pages].append(trim_image(img_byte_arr, text=row_titles[i], make_box=True, scale=scale ))
#     pagelist = [concat_vertical(value) for key, value in dict(pagemapping).items()]
#     pagelist[0].save(f"{fname}", save_all = True, quality=100, append_images = pagelist[1:])
#     return pagelist
#
#
# from reportlab.pdfgen import canvas
# from reportlab.pdfbase import pdfform
# from reportlab.lib.utils import ImageReader
# from PIL import Image
# from reportlab.lib.colors import magenta, pink, blue, green, black, white
#
# c = canvas.Canvas('hellomainimage.pdf', pagesize=(595.27, 841.69))
#
# index = 1
#
#
# def add_to_c(c, index, img1, img2, img3, img4, img5):
#     start_height = 190 * index
#     width = 195
#     height = 150
#     legend_height = 30
#     img1 = ImageReader(img1)
#     img2 = ImageReader(img2)
#     img3 = ImageReader(img3)
#     img4 = ImageReader(img4)
#     img5 = ImageReader(img5)
#     text_height = 18
#     c.drawImage(img2, 5, start_height, width=width, height=height, preserveAspectRatio=True, showBoundary=False, mask='auto')
#     c.drawImage(img3, width+5, start_height, width=width, height=height, preserveAspectRatio=True, showBoundary=False, mask='auto')
#     c.drawImage(img4, width * 2+5, start_height, width=width, height=height, preserveAspectRatio=True, showBoundary=False, mask='auto')
#     c.drawImage(img1, 5, start_height + height, width=140, height=18, preserveAspectRatio=True, showBoundary=False, mask='auto')
#     c.drawImage(img5, width*1.6,start_height +  height, width=195, height=30, preserveAspectRatio=True, showBoundary=False, mask='auto')
#     form = c.acroForm
#     form.textfield(name=f'fname1_{index}', tooltip='First Name', x=5, y=start_height + height+18, height=18, fontSize=8, width=150, borderColor=white, fillColor=white, textColor=black, forceBorder=False)
#     form.textfield(name=f'fname2_{index}', tooltip='Samples Name', x=160, y= start_height + height, height=18, fontSize=8, width=120, borderColor=white, fillColor=white, textColor=black, forceBorder=False)
#     form.textfield(name=f'fname3_{index}', tooltip='Samples Name', x=160, y=start_height + height+18, height=18, fontSize=8, width=120, borderColor=white, fillColor=white, textColor=black, forceBorder=False)
#
#
# img1_byte_arr = io.BytesIO()
# img1.save(img1_byte_arr, format='PNG')
# img1_byte_arr = img1_byte_arr.getvalue()
#
# img2_byte_arr = io.BytesIO()
# img2.save(img2_byte_arr, format='PNG')
# img2_byte_arr = img2_byte_arr.getvalue()
#
# img3_byte_arr = io.BytesIO()
# img3.save(img3_byte_arr, format='PNG')
# img3_byte_arr = img3_byte_arr.getvalue()
#
# img4_byte_arr = io.BytesIO()
# img4.save(img4_byte_arr, format='PNG')
# img4_byte_arr = img4_byte_arr.getvalue()
#
# img5_byte_arr = io.BytesIO()
# img5.save(img5_byte_arr, format='PNG')
# img5_byte_arr = img5_byte_arr.getvalue()
#
# add_to_c(c, 0, img1_byte_arr,  img2_byte_arr, img3_byte_arr, img4_byte_arr, img5_byte_arr)
# add_to_c(c, 1, img1_byte_arr,  img2_byte_arr, img3_byte_arr, img4_byte_arr, img5_byte_arr)
# c.save()