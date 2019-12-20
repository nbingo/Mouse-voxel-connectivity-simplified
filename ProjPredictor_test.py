from ProjPredictor import ProjPredictor

pp = ProjPredictor(image_file='dentate_signal_aligned_100um_binarized.tif', verbose=True)
pp.filter_by_name('Thalamus')
pp.vol_to_probs()
pp.view_proj()
pp.save_projections('dentate_proj_signal_aligned_100um.tif')

pp.set_image('interposed_signal_aligned_100um_binarized.tif')
pp.filter_by_name('Thalamus')
pp.vol_to_probs()
pp.view_proj()
pp.save_projections('interposed_proj_signal_aligned_100um.tif')
