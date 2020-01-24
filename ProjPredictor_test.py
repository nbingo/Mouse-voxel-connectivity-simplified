from ProjPredictor import ProjPredictor

pp = ProjPredictor(image_file='dentate_signal_aligned_100um_binarized.tif',
                   source_area='Dentate nucleus',
                   verbose=True)
pp.filter_by_name('Thalamus')
pp.vol_to_probs()
# pp.view_proj()
pp.save_projections('dentate_proj_signal_aligned_100um.tif')
areas = ['Somatomotor areas',
         'Somatosensory areas',
         'Gustatory areas',
         'Visceral area',
         'Auditory areas',
         'Visual areas',
         'Anterior cingulate area',
         'Orbital area']
pp.save_proj_by_area(areas, 'dentate_projs')

pp.set_image_from_file('interposed_signal_aligned_100um_binarized.tif', source_area='Interposed nucleus')
pp.filter_by_name('Thalamus')
pp.vol_to_probs()
# pp.view_proj()
pp.save_projections('interposed_proj_signal_aligned_100um.tif')
pp.save_proj_by_area(areas, 'interposed_projs')
