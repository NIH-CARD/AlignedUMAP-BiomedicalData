import gdown
import os
mapping_url = {
    'NORM-COVID19Proteomics': "https://drive.google.com/drive/folders/1hvSvFaKQDX0aWECMn-GAOKxGTnQva_QY?usp=sharing",
    'ADNI_FOR_ALIGNED_TIME_SERIES': "https://drive.google.com/drive/folders/1yQOIIK_LcU5jtsGcKe0atjSugRTpV8yf?usp=sharing",
    'PPMI_FOR_ALIGNED_TIME_SERIES': "https://drive.google.com/drive/folders/18U5UbLUd8DKtAdhsJnjpXGT5oll1xrZz?usp=sharing",
    'NORM-ALVEOLAR_metacelltype': "https://drive.google.com/drive/folders/1mKBU1XFsm7Ehfnbx1ETR_5nGaOuRqnZe?usp=sharing",
    'MINMAX_MIMIC_ALLSAMPLES': "https://drive.google.com/drive/folders/1em5KTyTp9CrWToPH87WBXTRvlqgN0GWh?usp=sharing",
}
def download_data(input_dataset_name):
    input_visualization_method = "umap_aligned"
    dpath = f"results_data/{input_dataset_name}/{input_visualization_method}/generated_data/"
    os.makedirs(dpath, exist_ok=True)
    fname = f"results_data/{input_dataset_name}/{input_visualization_method}/generated_data/{input_dataset_name}_16.pickle"
    if os.path.exists(fname):
        return
    else:
        url = mapping_url[input_dataset_name]
        gdown.download_folder(url, output='results_data/', quiet=False) # for linux (streamlit deployment)
