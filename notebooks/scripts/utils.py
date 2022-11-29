import pickle
import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import random
import plotly.express as px
import seaborn as sns
from tslearn.metrics import dtw
import plotly
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import plotly
import pickle
import random

from wand.image import Image
from wand.color import Color
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
import math
from PIL import Image, ImageOps
import io
from PIL import Image as pilim
from wand.image import Image as wandim

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from collections import defaultdict

from pathlib import Path
data_dir = Path("/Users/dadua2/projects")





def get_pil_image(image_binary):
    with wandim(blob=image_binary) as img:
        image = Image.open(io.BytesIO(img.make_blob("png"))).copy()
    return image


def trim_image(image_binary, text=None, make_box=False, scale=1, whitespace=20, resize=False):
    with Drawing() as draw:
        with wandim(blob=image_binary) as img:
            img.trim(Color("WHITE"))
            if resize:
                img.resize(400, 300)
            image = Image.open(io.BytesIO(img.make_blob("png"))).copy()
    img_with_border = ImageOps.expand(image, border=whitespace*scale, fill='white')
    if text is not None:
        draw = ImageDraw.Draw(img_with_border)
        font = ImageFont.truetype("arial.ttf", 12 * scale)
        draw.text((0, 0), text, (0, 0, 0), font=font)
        img_with_border = ImageOps.expand(img_with_border, border=5, fill='white')
    if (make_box == True) or (text is not None):
        img_with_border = ImageOps.expand(img_with_border, border=1, fill='black')
        img_with_border = ImageOps.expand(img_with_border, border=10, fill='white')
    return img_with_border.copy()


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


def concat_horizontal(imlist):
    imfinal = imlist[0]
    for im in imlist[1:]:
        if type(im) is int:
            continue
        imfinal = get_concat_h(imfinal, im)
    return imfinal


def concat_vertical(imlist):
    imfinal = imlist[0]
    for im in imlist[1:]:
        if type(im) is int:
            continue
        imfinal = get_concat_v(imfinal, im)
    return imfinal


def return_generated_figure(temp, colorname, agelist, color_map, title, camera_setup_ppmi=None, time_mapping=['time_axis', {}],
                            mode_str='lines', rever_color=False):
    temp = temp.dropna(subset=['x', 'y', 'z', colorname, 'subject_id'])
    print('UMAP ALL DATA USAGE:', len(temp), temp['subject_id'].unique().shape)
    fig = px.line_3d(temp, x="z", y="y", z="x", color=colorname, width=1200, height=900, line_group='subject_id',
                     markers=True if mode_str == "lines+markers" else False, color_discrete_map=color_map, )
    # error_x = 'error_x', error_y = 'error_x', error_z = 'error_x')
    grid_color = 'rgba(242, 242, 252, 1)'# 'ghostwhite'
    bg_color = 'rgba(248, 248, 255, 0.95)' # 'ghostwhite'
    num = len(list(range(len(agelist))))
    if num < 5:
        tickllist = np.arange(0, num, 1)
    else:
        tickllist = np.arange(0, num, num // 5)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectratio=dict(x=1, y=0.5, z=0.5),
            xaxis_title=time_mapping[0],
            zaxis_title="UMAP-X",
            yaxis_title="UMAP-Y",
            yaxis=dict(
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='black',
                zerolinewidth=1,
                showgrid=True,
                zeroline=False,  # thick line at x=0
                visible=True,  # numbers below
                dtick=3,
                ticks="outside",
            ),
            xaxis=dict(
                tickvals=tickllist,
                ticktext=[time_mapping[1].get(f"t{i}", f"t{i}") for i in tickllist],
                # [time_mapping.get( results_data['age_list'][xo], results_data['age_list'][xo]) for xo in tickllist], # list(map(lambda x: time_mapping.get(x, x), results_data['age_list'])),
                # results_data['age_list'],
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='white',
                zerolinewidth=1,
                showgrid=True,
                tickmode='array',
                showspikes=False,
                zeroline=False,
                visible=True,
                ticks="outside",
            ),
            zaxis=dict(
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='black',
                zerolinewidth=1,
                showgrid=True,
                zeroline=False,  # thick line at x=0
                visible=True,  # numbers below
                showspikes=False,
                dtick=3,
                ticks="outside",
            )
        ),
        scene_camera=camera_setup_ppmi,  # dict(eye=dict( x=0.5, y=0.8, z=0.75 )),
        autosize=False  # True if camera_setup_ppmi else False,
    )

    fig.update_traces(marker={'size': 1})
    #                   error_x=dict(
    #                       # type='constant',
    #                       # color='purple',
    #                       thickness=5,
    #                       width=3,
    #                   ),
    #                   error_y=dict(
    #                       # type='constant',
    #                       # color='purple',
    #                       thickness=5,
    #                       width=3,
    #                   ),
    #                   error_z=dict(
    #                       # type='constant',
    #                       # color='purple',
    #                       thickness=5,
    #                       width=3,
    #                   ),
    #                   )
    fig.update_layout(
        font=dict(
            size=14,
            color="black"
        )
    )
    fig.update_layout(showlegend=False)

    fig.update_xaxes(
        title_standoff=0
    )
    return fig



def return_generated_figure_umap(temp, colorname, agelist, color_map, title, camera_setup_ppmi=None, time_mapping=['time_axis', {}],
                            mode_str='lines', rever_color=False):
    temp = temp.dropna(subset=['x', 'y', colorname, 'time_id', 'subject_id'])
    print('UMAP ALL DATA USAGE:', len(temp), temp['subject_id'].unique().shape)
    fig = px.scatter(temp, x="x", y="y", color='time_id', width=900, height=900, color_discrete_map=color_map)
    K = temp.groupby([colorname]).mean()
    for i in range(len(K)):
        fig.add_annotation(x=K.iloc[i]['x'], y=K.iloc[i]['y'], text=f"{K.index[i]}", showarrow=False, yshift=0, xshift=0, bgcolor='yellow', height=18, font={'color': 'black', 'size': 15})

    grid_color = 'white'
    bg_color = 'white'

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="UMAP-X",
        yaxis_title="UMAP-Y",
        yaxis=dict(
            zerolinecolor='black',
            zerolinewidth=1,
            showgrid=True,
            zeroline=False,  # thick line at x=0
            visible=True,  # numbers below
            dtick=3,
            ticks="outside",
        ),
        xaxis=dict(
            zerolinecolor='black',
            zerolinewidth=1,
            showgrid=True,
            zeroline=False,  # thick line at x=0
            visible=True,  # numbers below
            showspikes=False,
            dtick=3,
            ticks="outside",
        ),
        autosize=False,  # True if camera_setup_ppmi else False,
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
    )

    fig.update_layout(
        font=dict(
            size=15,
            color="black"
        )
    )
    fig.update_layout(showlegend=False)

    fig.update_xaxes(
        title_standoff=10
    )
    fig.update_yaxes(
        title_standoff=10
    )
    # fig.update_xaxes(showgrid=False, gridwidth=0.1, gridcolor='rgb(220,220,220)')
    # fig.update_yaxes(showgrid=False, gridwidth=0.1, gridcolor='rgb(220,220,220)')

    return fig

import pickle
import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import random
import plotly.express as px
import seaborn as sns
import plotly

l = sns.color_palette("viridis", 91)
mycolorscale = list(l.as_hex())
from pathlib import Path

import seaborn as sns


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)] + [alpha])


def hex_to_rgba_tuple(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha])

def rgb_to_rgba(h, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    # import pdb; pdb.set_trace()
    return tuple( list( map ( lambda x: int(x.strip()) / 255, str(h)[4:-1].split(',') )) + [alpha])
    # return f"rgba{rgb_value[3:-1]}, {alpha})"

# def generate_color_map(category_list, alpha=1):
def generate_color_map(data, colorname, alpha=1, transparency_effect=False, sequential_scale=None):

    category_list = data[colorname]
    category_list_unique = category_list.unique()
    if len(category_list_unique) <= 24:
        try:
            if sequential_scale:
                import seaborn as sns
                l = sns.color_palette("Spectral", 8) # sns.diverging_palette(145, 300, s=60)
                temp1_mycolorscale = list(l.as_hex())
                # temp1_mycolorscale = px.colors.diverging.Picnic[:4] + px.colors.diverging.Picnic[6:]
                temp_mycolorscale = []
                for i in range(0, len(temp1_mycolorscale), len(temp1_mycolorscale) // len(category_list_unique)):
                    temp_mycolorscale.append(temp1_mycolorscale[i])


            else:
                dummy = float(list(category_list_unique)[0])
                temp1_mycolorscale = px.colors.sequential.Plasma_r
                temp_mycolorscale = []
                for i in range(0, len(temp1_mycolorscale), len(temp1_mycolorscale)//len(category_list_unique)):
                    temp_mycolorscale.append(temp1_mycolorscale[i])
            print ("Using Sequential Scale")
        except:
            temp_mycolorscale = px.colors.qualitative.Dark24
            print ("Using Discrete Scale")

        mycolorscale = []
        no_mycolorscale = []
        raw_colorscale = []
        for enm in temp_mycolorscale:
            raw_colorscale.append(enm)
            if not 'rgb' in enm:
                color = 'rgba' + str(hex_to_rgba(
                    h=enm,
                    alpha=alpha
                ))
                mycolorscale.append(color)
                nocolor = hex_to_rgba_tuple(
                    h=enm,
                    alpha=1
                )
                no_mycolorscale.append(nocolor)
            else:
                color = 'rgba' + str(rgb_to_rgba(
                    h=enm,
                    alpha=alpha
                ))
                # print (color)
                mycolorscale.append(color)
                nocolor = rgb_to_rgba(
                    h=enm,
                    alpha=1
                )
                no_mycolorscale.append(nocolor)

        color_map = {}
        nocolor_map = {}
        raw_colormap = {}
        for enm, i in enumerate(sorted(list(category_list_unique))):
            color_map[i] = mycolorscale[enm]
            nocolor_map[i] = no_mycolorscale[enm]
            raw_colormap[i] =  raw_colorscale[enm]

        subject_id_map = {}
        if transparency_effect:
            subject_colorname_mapping = dict(zip(list(data['subject_id']), list(data[colorname])))
            MEAN = data.groupby([colorname, 'z']).mean()
            subject_distance_mapping_x = {}
            subject_distance_mapping_y = {}
            g = list(data.groupby('subject_id'))

            for i in tqdm(range(len(g))):
                temp = g[i][1].groupby([colorname, 'z']).agg('mean')
                dtw_score_x = dtw(temp['x'], MEAN.loc[temp.index]['x'])
                dtw_score_y = dtw(temp['y'], MEAN.loc[temp.index]['y'])
                subject_distance_mapping_x[g[i][0]] = dtw_score_x
                subject_distance_mapping_y[g[i][0]] = dtw_score_y
                # subject_distance_mapping[g[i][0]] = (temp - MEAN.loc[temp.index]).abs().mean().mean()

            mean_val_x = np.mean(list(subject_distance_mapping_x.values()))
            std_val_x = np.std(list(subject_distance_mapping_x.values()))
            mean_val_y = np.mean(list(subject_distance_mapping_y.values()))
            std_val_y = np.std(list(subject_distance_mapping_y.values()))

            check = []
            for key, value in subject_distance_mapping_x.items():
                colorvalue = subject_colorname_mapping[key]
                xvalue = subject_distance_mapping_x[key]
                yvalue = subject_distance_mapping_y[key]
                if not 'rgb' in raw_colormap[colorvalue]:
                    f = hex_to_rgba
                else:
                    f = rgb_to_rgba

                if xvalue > (mean_val_x - 0.5 * std_val_x) and xvalue < (mean_val_x + 0.5 * std_val_x) and  yvalue > (mean_val_y - 0.5 * std_val_y) and yvalue < (mean_val_y + 0.5 * std_val_y):
                    newcolor = 'rgba' + str(f(
                        h=raw_colormap[colorvalue],
                        alpha=alpha
                    ))
                    subject_id_map[key] = newcolor
                    check.append( alpha )
                elif xvalue > (mean_val_x - 1 * std_val_x) and xvalue < (mean_val_x + 1 * std_val_x) and yvalue > (
                            mean_val_y - 1 * std_val_y) and yvalue < (mean_val_y + 1 * std_val_y):
                    newcolor = 'rgba' + str(f(
                        h=raw_colormap[colorvalue],
                        alpha=0.8 * alpha
                    ))
                    subject_id_map[key] = newcolor
                    check.append(0.8 * alpha)
                elif xvalue > (mean_val_x - 2 * std_val_x) and xvalue < (mean_val_x + 2 * std_val_x) and yvalue > (
                        mean_val_y - 2 * std_val_y) and yvalue < (mean_val_y + 2 * std_val_y):
                    newcolor = 'rgba' + str(f(
                        h=raw_colormap[colorvalue],
                        alpha=0.5 * alpha
                    ))
                    subject_id_map[key] = newcolor
                    check.append(0.5 * alpha)
                else:
                    newcolor = 'rgba' + str(f(
                        h=raw_colormap[colorvalue],
                        alpha = 0 * alpha # earlier 0.25
                    ))
                    subject_id_map[key] = newcolor
                    check.append( 0 * alpha )
            from collections import Counter
            print (Counter(check))
        return color_map, nocolor_map, subject_id_map # (new_id, subject_id_color)
    else:
        pass

def save_as_pdf_hbox(all_figs, row_titles, fname, all_legends=None, ncols=3, scale=1):
    nrows = 4
    total_rows = len(all_figs) // ncols if len(all_figs) % ncols == 0 else 1 + len(all_figs) // ncols
    Z = [[0 for i in range(ncols)] for j in range(total_rows)] # [[0]*ncols] * total_rows
    for i in tqdm(range(0, len(all_figs))):
        Z [i//ncols][i % ncols] = trim_image(all_figs[i])
    num_pages = total_rows // nrows if total_rows % nrows == 0 else 1 + total_rows // nrows
    pagemapping = defaultdict(list)
    for i in range(len(Z)):
        if all_legends is None:
            concated_image = concat_horizontal(Z[i])
        else:
            img_byte_arr = io.BytesIO()
            all_legends[i].save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            concated_image = concat_vertical([concat_horizontal(Z[i]), trim_image(img_byte_arr, whitespace=20)])
        # concated_image = concat_horizontal(Z[i])
        img_byte_arr = io.BytesIO()
        concated_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        pagemapping[i%num_pages].append(trim_image(img_byte_arr, text=row_titles[i], make_box=True, scale=scale ))
    pagelist = [concat_vertical(value) for key, value in dict(pagemapping).items()]
    pagelist[0].save(f"{fname}", save_all = True, quality=100, append_images = pagelist[1:])
    return pagelist


def save_as_pdf_allbox(all_figs, all_titles, fname, all_legends=None, ncols=3, scale=1):
    nrows = 4
    total_rows = len(all_figs) // ncols if len(all_figs) % ncols == 0 else 1 + len(all_figs) // ncols
    Z = [[0 for i in range(ncols)] for j in range(total_rows)] # [[0]*ncols] * total_rows
    for i in tqdm(range(0, len(all_figs))):
        if all_legends is None:
            Z [i//ncols][i % ncols] = trim_image(all_figs[i], text=all_titles[i], make_box=True, scale=scale)
        else:
            img_byte_arr = io.BytesIO()
            all_legends[i].save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            atemp = concat_vertical([ trim_image(all_figs[i], whitespace=1), trim_image(img_byte_arr, whitespace=20)])
            img_byte_arr = io.BytesIO()
            atemp.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            Z [i//ncols][i % ncols] = trim_image(img_byte_arr, text=all_titles[i], make_box=True, scale=scale, whitespace=60)
    num_pages = total_rows // nrows if total_rows % nrows == 0 else 1 + total_rows // nrows
    pagemapping = defaultdict(list)
    for i in range(len(Z)):
        concated_image = concat_horizontal(Z[i])
        pagemapping[i%num_pages].append(concated_image)
    pagelist = [concat_vertical(value) for key, value in dict(pagemapping).items()]
    pagelist[0].save(f"{fname}", save_all = True, quality=100, append_images = pagelist[1:])
    return pagelist


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
def get_legend_box(nocolor_maps, selected_cats,  scale=1, fontsize=8, mapping_legend={}, ncols=4, type='line'):
    legend_elements = []
    for label, color in nocolor_maps.items():
        if label in selected_cats or len(selected_cats) == 0: # ['macrophages', 'alv_epithelium', 'T_cells']:
            # legend_elements.append(Patch(facecolor=color, label=label))
            # legend_elements.append(Rectangle(xy=(0,0), height=0.2, width=0.2, facecolor=color, label=label))
            if type == 'line':
                legend_name = Line2D([0], [0], color=color, linewidth=8, linestyle='-', label=mapping_legend.get(label, label), pickradius=2, antialiased=True)
            elif type == 'circle':
                # legend_name = Line2D([0], [0], color=color, linewidth=4, linestyle='-', label=mapping_legend.get(label, label), pickradius=2, antialiased=True)
                legend_name = plt.plot([], [], marker="o", ms=15, ls="", mec=None, color=color, label=mapping_legend.get(label, label))[0]
            legend_elements.append(legend_name)
            # legend_elements.append(Circle(xy=(0,0), radius=0.2, facecolor=color, label=label))

    fignew, ax = plt.subplots(figsize=(30, 4)) # figsize=(60, 4)
    plt.legend(handles=legend_elements, loc='center', ncol=ncols, fontsize=fontsize, frameon=True)
    plt.axis('off')
    b = io.BytesIO()
    plt.savefig(b, format='png', dpi=100*scale, bbox_inches='tight', pad_inches=0)
    plt.close()
    return b


def generate_dataset(results_data, input_dataset_name, parameter_fix, anglelist, num_subjects, selected_cats, colorname, scale=1, alpha=1, transparency_effect=False, time_mapping=['time_axis', {}]):
    data = results_data['complete_dataframe'].copy()
    data = data[data['input_parameter_id'] == parameter_fix]
    all_subjects = list(data['subject_id'].unique())
    random.seed(42)
    random.shuffle(all_subjects)
    # print('All subject', len(all_subjects), 'Selected_subjects', num_subjects)
    if num_subjects < 1:
        X = data.drop_duplicates(subset=['subject_id', colorname])
        y = X[colorname].values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_subjects, random_state=0)
        train_index, test_index = list(sss.split(X.values, y))[0]
        select_subject = list(set(list(X.iloc[list(test_index)]['subject_id'])))
        data = data[data['subject_id'].isin(select_subject)]
        selected_data = data.copy()
    else:
        if num_subjects > 1:
            select_subject = all_subjects[:num_subjects]
        else:
            select_subject = all_subjects
        data = data[data['subject_id'].isin(select_subject)]
        selected_data = data.copy()

    if not transparency_effect:
        color_maps, nocolor_maps, _ = generate_color_map(selected_data, colorname, alpha=alpha)
    else:
        color_maps, nocolor_maps, subject_id_map = generate_color_map(selected_data, colorname, alpha=alpha, transparency_effect=True)

    if len(selected_cats) == 0:
        pass
    else:
        selected_data = selected_data[selected_data[colorname].isin(selected_cats)]

    camera_setup_ppmi = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0), # if not input_dataset_name == 'ADNI_FOR_ALIGNED_TIME_SERIES' else dict(x=0, y=1, z=0),
        eye=dict(x=anglelist[0], y=anglelist[1],
                 z=anglelist[2])
    )
    if not transparency_effect:
        fig = return_generated_figure(selected_data, colorname, list(results_data['age_list']), color_maps, '', camera_setup_ppmi=camera_setup_ppmi,
                                  time_mapping=time_mapping, mode_str='lines+markers', rever_color=False)
    else:
        # import pdb; pdb.set_trace()
        fig = return_generated_figure(selected_data, 'subject_id', list(results_data['age_list']), subject_id_map, '',
                                      camera_setup_ppmi=camera_setup_ppmi,
                                      time_mapping=time_mapping, mode_str='lines+markers', rever_color=False)

    return plotly.io.to_image(fig, format='png', width=900, height=450, scale=scale), (color_maps, nocolor_maps)


def generate_dataset_umap(results_data, input_dataset_name, parameter_fix, anglelist, num_subjects, selected_cats, colorname, scale=1, alpha=1, transparency_effect=False, time_mapping=['time_axis', {}], mapping_legend={}):
    data = results_data['complete_dataframe'].copy()
    data = data[data['input_parameter_id'] == parameter_fix]
    all_subjects = list(data['subject_id'].unique())
    random.seed(42)
    random.shuffle(all_subjects)
    # print('All subject', len(all_subjects), 'Selected_subjects', num_subjects)
    if num_subjects < 1:
        X = data.drop_duplicates(subset=['subject_id', colorname])
        y = X[colorname].values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_subjects, random_state=0)
        train_index, test_index = list(sss.split(X.values, y))[0]
        select_subject = list(set(list(X.iloc[list(test_index)]['subject_id'])))
        data = data[data['subject_id'].isin(select_subject)]
        selected_data = data.copy()
    else:
        if num_subjects > 1:
            select_subject = all_subjects[:num_subjects]
        else:
            select_subject = all_subjects
        data = data[data['subject_id'].isin(select_subject)]
        selected_data = data.copy()

    if not transparency_effect:
        color_maps, nocolor_maps, _ = generate_color_map(selected_data, 'time_id', alpha=alpha, sequential_scale=True)
    else:
        color_maps, nocolor_maps, subject_id_map = generate_color_map(selected_data, 'time_id', alpha=alpha, transparency_effect=True, sequential_scale=True)

    if len(selected_cats) == 0:
        pass
    else:
        selected_data = selected_data[selected_data[colorname].isin(selected_cats)]

    selected_data[colorname] = selected_data[colorname].map(lambda x: mapping_legend.get(x, x))
    camera_setup_ppmi = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=anglelist[0], y=anglelist[1],
                 z=anglelist[2])
    )
    if not transparency_effect:
        fig = return_generated_figure_umap(selected_data, colorname, [], color_maps, '', camera_setup_ppmi=camera_setup_ppmi,
                                  time_mapping=time_mapping, mode_str='lines+markers', rever_color=False, )
    else:
        # import pdb; pdb.set_trace()
        fig = return_generated_figure_umap(selected_data, 'subject_id', [], subject_id_map, '',
                                      camera_setup_ppmi=camera_setup_ppmi,
                                      time_mapping=time_mapping, mode_str='lines+markers', rever_color=False)

    return plotly.io.to_image(fig, format='png', width=500, height=400, scale=scale), (color_maps, nocolor_maps)