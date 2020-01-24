from ProjPredictor import ProjPredictor
import os

image_path = '/transformix_output_ilastik/result_fixed.tif'
areas = ['Somatomotor areas',
         'Somatosensory areas',
         'Gustatory areas',
         'Visceral area',
         'Auditory areas',
         'Visual areas',
         'Anterior cingulate area',
         'Orbital area']
nuclei = [('DN', 'Dentate nucleus'), ('FN', 'Fastigial nucleus'), ('IN', 'Interposed nucleus')]
pp = ProjPredictor(verbose=True)

for nucleus in nuclei:
    d = f'datafornomi/{nucleus[0]}fornomi/'
    brains = os.listdir(d)
    for brain in brains:
        pp.set_image_from_file(d + brain + image_path, source_area=nucleus[1], reshape=True)
        pp.threshold(0.2)
        pp.filter_by_name('Thalamus')
        pp.vol_to_probs()
        pp.save_projections(nucleus[0] + brain[-3:] + 'raw_proj.tiff')
        pp.save_proj_by_area(areas, nucleus[0] + brain[-3:] + 'proj_by_area.pickle')