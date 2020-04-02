from ProjPredictor import ProjPredictor
import os
import pandas as pd
from tqdm import tqdm

image_path = '/transformix_output_ilastik/result_fixed.tif'
areas = pd.read_csv('annotation_info_0118_1327.csv')
areas = areas[areas['consider'] == 1]['name'].values.tolist()

nuclei = [('DN', 'Dentate nucleus'), ('FN', 'Fastigial nucleus'), ('IN', 'Interposed nucleus')]
pp = ProjPredictor(verbose=False)
# area_filter = 'Ventral medial nucleus of the thalamus'
area_filter = 'Thalamus'
for nucleus in nuclei:
    d = f'datafornomi/{nucleus[0]}fornomi/'
    brains = os.listdir(d)
    brains = [brain for brain in brains if not brain.startswith('.')]
    for brain in tqdm(brains):
        pp.set_image_from_file(d + brain + image_path, source_area=nucleus[1], reshape=True)
        pp.threshold(0.2)
        pp.filter_by_name(area_filter)
        pp.vol_to_probs()
        pp.save_projections(f'raw_proj/{nucleus[0]}{brain[-3:]}_filter-{area_filter}_raw_proj.tiff')
        pp.save_proj_by_area(structure_name=areas,
                             normalize_source=True,
                             normalize_target=True,
                             fname=f'proj_by_area_justus/{nucleus[0]}{brain[-3:]}_filter-{area_filter}'
                                   f'_both-norm_proj_by_area.pickle')
        pp.save_proj_by_area(structure_name=areas,
                             normalize_source=False,
                             normalize_target=True,
                             fname=f'proj_by_area_justus/{nucleus[0]}{brain[-3:]}_filter-{area_filter}'
                                   f'_target-norm_proj_by_area.pickle')
        pp.save_proj_by_area(structure_name=areas,
                             normalize_source=True,
                             normalize_target=False,
                             fname=f'proj_by_area_justus/{nucleus[0]}{brain[-3:]}_filter-{area_filter}'
                                   f'_source-norm_proj_by_area.pickle')
        pp.save_proj_by_area(structure_name=areas,
                             normalize_source=False,
                             normalize_target=False,
                             fname=f'proj_by_area_justus/{nucleus[0]}{brain[-3:]}_filter-{area_filter}'
                                   f'_no-norm_proj_by_area.pickle')
