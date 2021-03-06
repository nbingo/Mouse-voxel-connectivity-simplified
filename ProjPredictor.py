from mcmodels.core import VoxelModelCache
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
import numpy as np
from typing import Union, List
from skimage import io
from skimage.transform import resize
import napari
import pandas as pd
import warnings


class ProjPredictor:
    """A class wrapper around the Allen Institute VoxelModelCache and
    allensdk to make extracting projection data easier.

    Attributes
    ----------
    load_cache : bool
        Whether the cache should be loaded or not. For example, the cache isn't needed if projections have
        already been calculated.
    manifest_file : str
        A string representing the manifest to read from for the voxel model cache
    ccf_version : str
        A formatted string representing the version of allensdk data to use
    image_file : str
        A filename pointing to an image to read in
    source_area : str
        The name of the area from which the original projection data was gathered.
    y_mirror : bool
        A boolean representing whether the image should be mirrored along the median plane
    verbose : bool
        A boolean representing whether verbose debugging messages should be printed

    Methods
    -------
    save_projections(self, filename: str) -> None
    """
    def __init__(self,
                 load_cache: bool = True,
                 manifest_file: str = 'voxel_model_manifest.json',
                 ccf_version: str = MouseConnectivityApi.CCF_VERSION_DEFAULT,
                 image_file: str = None,
                 source_area: str = None,
                 filter_area: Union[str, List[str]] = None,
                 y_mirror: bool = False,
                 verbose: bool = False) -> None:
        """

        Parameters
        ----------
        load_cache : bool
            Whether the cache should be loaded or not. For example, the cache isn't needed if projections have
            already been calculated.
        manifest_file : str
            A string representing the manifest to read from for the voxel model cache
        ccf_version : str
            A formatted string representing the version of allensdk data to use
        image_file : str
            A filename pointing to an image to read in
        source_area : str
            The name of the area from which the original projection data was gathered.
        filter_area : Union[str, List[str], int, List[int]]
            Area(s) by which to filter the source voxels before getting their projections. Only give names.
        y_mirror : bool
            A boolean representing whether the image should be mirrored along the median plane
        verbose : bool
            A boolean representing whether verbose debugging messages should be printed
        """
        self.y_mirror = y_mirror
        self.verbose = verbose
        if load_cache:
            if self.verbose:
                print('Loading Voxel Model Cache...')
            self._cache = VoxelModelCache(manifest_file=manifest_file, ccf_version=ccf_version)
            if self.verbose:
                print('Extracting voxel array, source mask, and target mask...')
            self._voxel_array, self._source_mask, self._target_mask = self._cache.get_voxel_connectivity_array()
        if image_file is not None:
            if self.verbose:
                print(f'Loading image "{image_file}"...')
            self.image: np.array = io.imread(image_file)
        else:
            self._image: np.array = None
        self._projections: np.array = None
        if source_area is not None:
            self.source_area: str = source_area
        if filter_area is not None:
            self.filter_area = filter_area
        else:
            self._filter_area = None
        self.default_shape = (65, 88, 88)

    @property
    def source_area(self) -> str:
        return self._source_area

    @source_area.setter
    def source_area(self, struct_name: str) -> None:
        self.assert_valid_structure_name(struct_name)
        self._source_area = struct_name

    @property
    def filter_area(self) -> Union[str, List[str]]:
        return self._filter_area

    @filter_area.setter
    def filter_area(self, struct_name: Union[str, List[str]]) -> None:
        self.assert_valid_structure_name(struct_name)
        self._filter_area = struct_name

    def assert_valid_structure_name(self, struct_name: Union[str, List[str]]):
        if not isinstance(struct_name, list):
            struct_name = [struct_name]
        if not np.array([isinstance(name, str) for name in struct_name]).all():
            warnings.warn('Source area must be a string of the FULL name (not acronym) of the source area.',
                          UserWarning)
        names = self._cache.get_structure_tree().get_name_map().values()
        if np.array([name not in names for name in struct_name]).any():
            warnings.warn('Source area name (not acronym) cannot be found in the structure tree.', UserWarning)

    @property
    def image(self) -> np.array:
        return self._image

    @image.setter
    def image(self, image_file: Union[str, np.array]) -> None:
        if isinstance(image_file, str):
            self._image = io.imread(image_file)
        else:
            self._image = image_file
        self._permute_pad_reflect()

    @property
    def projections(self) -> np.array:
        if self._projections is None:
            self.vol_to_probs()
        return self._projections

    @projections.setter
    def projections(self, image_file: Union[str, np.array]) -> None:
        if isinstance(image_file, str):
            self._projections = io.imread(image_file)
        else:
            self._projections = image_file

    def save_projections(self, filename: str, bits: int = 32) -> None:
        """Saves the projections with the given filename

        If there is no currently saved projection image, then an attempt is made to compute the projections
        assuming that an image has been given.

        Parameters
        ----------
        filename : str
            The filename to be given to the saved projection image. It should include the file type extension.
        bits : int
            The number of bits to save the tiff file as. Choose from: 16, 32, 64

        Returns
        -------
        None
        """
        if bits == 16:
            float_type = np.float16
        elif bits == 64:
            float_type = np.float64
        else:
            float_type = np.float32
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            io.imsave(filename, self.projections.astype(float_type))

    def set_image_from_file(self, image_file: str,
                            y_mirror: bool = False,
                            source_area: str = None,
                            reshape: bool = False) -> None:
        self.y_mirror = y_mirror
        if self.verbose:
            print(f'Loading image "{image_file}"')
        im = io.imread(image_file)
        if reshape:
            im = resize(im, self.default_shape)
        self.image = im
        if source_area is not None:
            self.source_area = source_area

    def view_source(self) -> napari.Viewer:
        """Brings up a napari viewer of the source image.

        Returns
        -------
        napari viewer with the source image."""
        with napari.gui_qt():
            return napari.view_image(self.image)

    def view_proj(self) -> napari.Viewer:
        """Brings up a napari viewer of the projection image. If there is not one, then
        the projections are calculated.

        Returns
        -------
        napari viewer with the projection image."""
        with napari.gui_qt():
            return napari.view_image(self.projections)

    def threshold(self, thresh: float) -> None:
        self._image = self.image > thresh

    def vol_to_probs(self, save: bool = True) -> np.array:
        """Takes the inner source image and computes the projections from each source voxel.

        The source image must be a binary, {0,1}, image. The projections of each voxel are calculated
        and then summed at the end. If desired, this resulting projections image can be saved.

        Parameters
        ----------
        save : bool
            Whether to save the resulting projections image.

        Returns
        -------
        The resulting projection image, which has the same dimensionality as the source image.
        """
        if self.verbose:
            print('Converting source image to projection probabilities...')
        data_flattened = self._source_mask.mask_volume(self.image)

        row = self._voxel_array[data_flattened == 1].sum(axis=0)
        np.nan_to_num(row, copy=False, nan=0.0)
        return_volume = self._target_mask.map_masked_to_annotation(row)

        if save:
            self.projections = return_volume

        return return_volume

    def _permute_pad_reflect(self) -> None:
        """Permutes, pads, and reflects the stored image to match it to the 100um annotation.

        Used internally when the image is set. Currently has hard coded numbers.
        """
        if self.verbose:
            print('Permuting, padding, and reflecting source image...')
        self._image = np.transpose(self._image, (1, 0, 2))
        self._image = np.pad(self._image, ((0, 132 - 88), (80 - 65 - 10, 10), (13, 114 - 88 - 13)), 'constant')
        self._image = np.flip(self._image, axis=(0, 1))
        if self.y_mirror:
            self._image = np.fliplr(self._image)

    def _filter_by_id(self, structure_id: Union[int, List[int]]) -> None:
        """Given an id or a list of ids, only preserves voxels from the original image that are included
        in at least one of the given structures.

        Parameters
        ----------
        structure_id : Union[int, List[int]]
            A single id or a list of ids (which will be unioned together) to mask the stored image.
        """
        if self.verbose:
            print('Filtering source image by selected structures...')
        mask = self.struct_ids_to_mask(structure_id)
        self._image = self._image * mask

    def struct_ids_to_mask(self, structure_id: Union[int, List[int]]) -> np.array:
        """
        Takes in structure ids or id and creates a mask for those structures. If multiple structures,
        then the resulting mask will be a union of masks.

        Parameters
        ----------
        structure_id : Union[int, List[int]]
            A single id or a list of ids (which will be unioned together) to mask the stored image.

        Returns
        -------
        Binary array with a 1 where at least one of the given structures is present.
        """
        if not isinstance(structure_id, list):
            structure_id = [structure_id]
        mask = self._cache.get_reference_space().make_structure_mask(structure_id)
        return mask

    def filter_by_name(self, structure_name: Union[str, List[str]]) -> None:
        """Given a structure name or a list of structure names, only preserves voxels from the original image
        that are included in at least one of the given structures.

        Parameters
        ----------
        structure_name : Union[str, List[str]]
            A single structure name or list of structure names (which will be unioned together)
            to mask the stored image.
        """
        self.filter_area = structure_name
        ids = self.struct_names_to_ids(structure_name)
        self._filter_by_id(ids)

    def struct_names_to_ids(self, structure_name: Union[str, List[str]]) -> List[int]:
        """
        Takes in structure name(s) and returns their id(s).

        Parameters
        ----------
        structure_name : Union[str, List[str]]
            Single name or list of names of desired structures

        Returns
        -------
        List of ints with each id in the same order as the structures were given.
        """
        if not isinstance(structure_name, list):
            structure_name = [structure_name]
        structures = self._cache.get_structure_tree().get_structures_by_name(structure_name)
        ids = [structure['id'] for structure in structures]
        return ids

    def save_proj_by_area(self,
                          structure_name: Union[str, List[str]],
                          normalize_source: bool = False,
                          normalize_target: bool = False,
                          fname: str = 'proj_by_area') -> None:
        """
        Saves a Pandas array that contains the source area, target area, and summed projection strength from
        the source area to the target area. It will have as many rows as target areas, or one if structure_name
        is just a string and not a list.

        Parameters
        ----------
        structure_name : Union[str, List[str]]
            A string or list of strings denoting the target areas to filter and save by.
        normalize_source : bool
            Boolean indicating whether to normalize by the number of source voxels used.
        normalize_target : bool
            Boolean indicating whether to normalize by the target area (to get density of projections).
        fname : str
            The file name of the file to be saved.

        Returns
        -------
        None
        """
        if self.verbose:
            print(f'Saving projections by area to: {fname}')
        self.assert_valid_structure_name(structure_name)
        ids = self.struct_names_to_ids(structure_name)
        if not isinstance(structure_name, list):
            structure_name = [structure_name]
        if normalize_target:
            proj_strengths = [(self.struct_ids_to_mask(i) * self.projections).sum() /
                              self.struct_ids_to_mask(i).sum() for i in ids]
        else:
            proj_strengths = [(self.struct_ids_to_mask(i) * self.projections).sum() for i in ids]
        proj_strengths = np.array(proj_strengths)
        source_area_voxels = self.image.sum()
        if normalize_source:
            proj_strengths = proj_strengths / source_area_voxels
        num_target_structs = len(structure_name)
        proj_dict = {'Source area': [self.source_area] * num_target_structs,
                     'Target area': structure_name,
                     'Projection strength': proj_strengths,
                     'Normalized by source': [normalize_source] * num_target_structs,
                     'Normalized by target': [normalize_target] * num_target_structs
                     }
        if self.filter_area is not None:
            proj_dict['Filter area'] = [self.filter_area] * num_target_structs
        df = pd.DataFrame(proj_dict)
        pd.to_pickle(df, fname)
