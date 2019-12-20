from mcmodels.core import VoxelModelCache
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
import numpy as np
from typing import Union, List
from skimage import io
import napari


class ProjPredictor:
    """A class wrapper around the Allen Institute VoxelModelCache and
    allensdk to make extracting projection data easier.

    Attributes
    ----------
    manifest_file : str
        A string representing the manifest to read from for the voxel model cache
    ccf_version : str
        A formatted string representing the version of allensdk data to use
    image_file : str
        A filename pointing to an image to read in
    y_mirror : bool
        A boolean representing whether the image should be mirrored along the median plane
    verbose : bool
        A boolean representing whether verbose debugging messages should be printed

    Methods
    -------
    save_projections(self, filename: str) -> None
    """
    def __init__(self,
                 manifest_file: str = 'voxel_model_manifest.json',
                 ccf_version: str = MouseConnectivityApi.CCF_VERSION_DEFAULT,
                 image_file: str = None,
                 y_mirror: bool = False,
                 verbose: bool = False) -> None:
        self.y_mirror = y_mirror
        self.verbose = verbose
        if self.verbose:
            print('Loading Voxel Model Cache...')
        self.cache = VoxelModelCache(manifest_file=manifest_file, ccf_version=ccf_version)
        if self.verbose:
            print('Extracting voxel array, source mask, and target mask...')
        self.voxel_array, self.source_mask, self.target_mask = self.cache.get_voxel_connectivity_array()
        if image_file:
            if self.verbose:
                print(f'Loading image "{image_file}"...')
            self.image: np.array = io.imread(image_file)
            self.permute_pad_reflect()
        else:
            self.image: np.array = None
        self.projections: np.array = None

    def save_projections(self, filename: str) -> None:
        """Saves the projections with the given filename

        If there is no currently saved projection image, nothing happens.

        Parameters
        ----------
        filename : str
            The filename to be given to the saved projection image. It should include the file type extension.

        Returns
        -------
        Nothing
        """
        if self.projections is not None:
            io.imsave(filename, self.projections)

    def set_image(self, image_file: str, y_mirror: bool = False) -> None:
        if self.verbose:
            print(f'Loading image "{image_file}"')
        self.image = io.imread(image_file)
        self.y_mirror = y_mirror
        self.permute_pad_reflect()

    def view_source(self) -> napari.Viewer:
        with napari.gui_qt():
            return napari.view_image(self.image)

    def view_proj(self) -> napari.Viewer:
        with napari.gui_qt():
            return napari.view_image(self.projections)

    def vol_to_probs(self, save: bool = True) -> np.array:
        data_flattened = self.source_mask.mask_volume(self.image)

        row = self.voxel_array[data_flattened == 1].mean(axis=0)
        return_volume = self.target_mask.map_masked_to_annotation(row)

        if save:
            self.projections = return_volume

        return return_volume

    def permute_pad_reflect(self) -> None:
        self.image = np.transpose(self.image, (1, 0, 2))
        self.image = np.pad(self.image, ((0, 132 - 88), (80 - 65 - 10, 10), (13, 114 - 88 - 13)), 'constant')
        self.image = np.flip(self.image, axis=(0, 1))
        if self.y_mirror:
            self.image = np.fliplr(self.image)

    def filter_by_id(self, structure_id: Union[int, List[int]]) -> None:
        if not isinstance(structure_id, list):
            structure_id = [structure_id]
        mask = self.cache.get_reference_space().make_structure_mask(structure_id)
        self.image = self.image * mask

    def filter_by_name(self, structure_name: Union[str, List[str]]) -> None:
        if not isinstance(structure_name, list):
            structure_name = [structure_name]
        structures = self.cache.get_structure_tree().get_structures_by_name(structure_name)
        ids = [structure['id'] for structure in structures]
        self.filter_by_id(ids)
