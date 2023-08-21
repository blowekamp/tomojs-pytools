from pathlib import Path

import SimpleITK as sitk
import zarr
from typing import Optional, Iterable, Tuple, Union, AnyStr
from pytools._xml_info import OMEInfo
import logging
from pytools.zarr_extract_2d import zarr_extract_2d

logger = logging.getLogger(__name__)


class HedwigZarrImage:
    """
    Represents a OME-NGFF Zarr pyramidal image.
    """

    def __init__(self, zarr_grp: zarr.Group, _ome_info: OMEInfo, _ome_idx: int):
        self.zarr_group = zarr_grp
        self.ome_info = _ome_info
        self.ome_idx = _ome_idx

        assert "multiscales" in self.zarr_group.attrs

    @property
    def path(self) -> Path:
        """
        Returns full path to the ZARR group suitable for Neuroglancer.
        """
        return Path(self.zarr_group.store.path) / self.zarr_group.path

    @property
    def dims(self) -> str:
        """
        The used dimension of the image XY, XYC, XYZCT etc.

        Collapses ZCT dimensions if of size 1.
        """
        dims = [d for s, d in zip(self.shape, self._ome_ngff_multiscale_dims()) if d in "XY" or s > 1]
        return "".join(dims[::-1])

    @property
    def shape(self) -> Tuple[int]:
        """The size of the dimensions of the full resolution image."""
        return self._ome_ngff_multiscale_image(0).shape

    def rechunk(self, chunk_size: int) -> None:
        """
        Change the chunk size of each ZARR array in the pyramid.
        """

        return self._rechunk_group(chunk_size)

    def extract_2d(
        self,
        target_size_x: int,
        target_size_y: int,
        *,
        size_factor: float = 1.5,
        output_filename: Union[Path, str, None] = None,
    ) -> Optional[sitk.Image]:
        """Extracts a 2D image from an OME-NGFF pyramid structured with ZARR array that is 2D-like.

        The OME-NGFF pyramid structured ZARR array is assumed to have the following structure:
            - The axes spacial dimensions must be labeled as "X", "Y", and optionally "Z".
            - If a "Z" space dimension exists then it must be of size 1.
            - If a time dimension exists then it must be if of size 1.
            - If a channel dimension exists all channels are extracted.

        The extracted subvolume will be resized to the target size while maintaining the aspect ratio.

        The resized extracted subvolume with be the sample pixel type as the OME-NGFF pyramid structured ZARR array.

        If output_filename is not None then the extracted subvolume will be written to this file. The output ITK ImageIO
        used to write the file may place additional contains on the image which can be written. Such as JPEG only
        supporting uint8 pixel types and 1, 3, or 4 components.

        :param input_zarr: The path to an OME-NGFF structured ZARR array.
        :param target_size_x: The target size of the extracted subvolume in the x dimension.
        :param target_size_y: The target size of the extracted subvolume in the y dimension.
        :param size_factor: The size of the subvolume to extract will be increased by this factor so that the
            extracted subvolume can have antialiasing applied to it.
        :param output_filename: If not None then the extracted subvolume will be written to this file.
        :return: The extracted subvolume as a SimpleITK image.

        """
        return zarr_extract_2d(
            self.path, target_size_x, target_size_y, size_factor=size_factor, output_filename=output_filename
        )

    @property
    def shader_type(
        self,
    ) -> str:
        """
        Produces the shader type one of: RGB, Grayscale, or MultiChannel.
        """
        if self.ome_info.maybe_rgb(self.ome_idx):
            return "RGB"
        if len(self.ome_info.image_names()) == 1:
            return "Grayscale"
        return "MultiChannel"

    def neuroglancer_shader_parameters(self, shader_type) -> dict:
        """
        Produces the "shaderParameters" portion of the metadata for Neuroglancer
        returns JSON serializable object
        """
        if self.shader_type == "RGB":
            return {}
        if self.shader_type == "Grayscale":
            return {}
        if self.shader_type == "Multichannel":
            return {}

        raise RuntimeError(f'Unknown shader type: "{self.shader_type}"')

    def _ome_ngff_multiscales(self, idx=0):
        return self.zarr_group.attrs["multiscales"][idx]

    def _ome_ngff_multiscale_image(self, level, idx=0):
        return self.zarr_group[self._ome_ngff_multiscales(idx)["datasets"][0]["path"]]

    def _ome_ngff_multiscale_dims(self):
        dims = ""
        for ax in self._ome_ngff_multiscales()["axes"]:
            dims += ax["name"].upper()
        return dims

    @staticmethod
    def _chunk_logic_dim(drequest: int, dshape: int) -> int:
        if dshape > drequest > 0:
            return drequest
        return dshape

    def _rechunk_group(self, chunk_size: int):
        logger.info(f'Processing group: "{self.zarr_group.name}"...')
        logger.debug(self.zarr_group)

        # grok through the OME-NGFF meta-dat, for each image scale (dataset/array) with axes information
        # https://ngff.openmicroscopy.org/latest/#multiscale-md
        for image in self._ome_ngff_multiscales():
            chunk_request = tuple(chunk_size if a["type"] == "space" else -1 for a in image["axes"])

            for dataset in image["datasets"]:
                arr = self.zarr_group[dataset["path"]]
                arr_name = self.zarr_group[dataset["path"]].name
                logger.info(f'Processing array: "{arr.name}"...')
                logger.debug(arr.info)

                chunks = tuple(self._chunk_logic_dim(r, s) for r, s in zip(chunk_request, arr.shape))
                if arr.chunks == chunks:
                    logger.info("Chunks already requested size")
                    continue

                # copy array to a temp zarr array on file
                zarr.copy(
                    arr,
                    self.zarr_group,
                    name=arr_name + ".temp",
                    chunks=chunks,
                    compressor=arr.compressor,
                    dimension_separator=arr._dimension_separator,
                    filters=arr.filters,
                    overwrite=False,
                )

                logger.debug(self.zarr_group[dataset["path"] + ".temp"].info)
                logger.debug(f"replace: {self.zarr_group[dataset['path'] + '.temp'].name} -> {arr_name}")
                del self.zarr_group[dataset["path"]]
                self.zarr_group.store.rename(self.zarr_group[dataset["path"] + ".temp"].name, arr_name)


class HedwigZarrImages:
    """
    Represents the set of images in a OME-NGFF ZARR structure.
    """

    def __init__(self, zarr_path: Path):
        """
        Initialized by the path to a root of an OME zarr structure.
        """
        # check zarr is valid
        assert zarr_path.exists()
        assert zarr_path.is_dir()
        self.zarr_store = zarr.DirectoryStore(zarr_path)
        self.zarr_root = zarr.Group(store=self.zarr_store, read_only=True)

    @property
    def ome_xml_path(self) -> Optional[Path]:
        """
        Returns the path to the OME-XML file, if it exists.
        """
        if "OME" in self.zarr_root.group_keys():
            _xml_path = Path(self.zarr_store.path) / self.zarr_root["OME"].path / "METADATA.ome.xml"
            if _xml_path.exists():
                return _xml_path
            else:
                return None

    @property
    def ome_info(self) -> Optional[AnyStr]:
        """Returns OME XML as string is if exists."""

        if hasattr(self, "_ome_info"):
            return self._ome_info

        _path = self.ome_xml_path
        if not _path:
            return None
        with open(_path, "r") as fp:
            self._ome_info = OMEInfo(fp.read())
            return self._ome_info

    def get_series_keys(self) -> Iterable[str]:
        """
        Will return an iterable of strings of the names or labels of the images. Will be extracted from
        the OME-XML if available otherwise the ZARR group names.
        e.g. "label_image"
        """

        if self.ome_info:
            return self.ome_info.image_names()

        return filter(lambda x: x != "OME", self.zarr_root.group_keys())

    def __getitem__(self, item) -> HedwigZarrImage:
        for k_idx, k in enumerate(self.get_series_keys()):
            if item == k:
                ome_index_to_zarr_group = self.zarr_root["OME"].attrs["series"]
                zarr_idx = ome_index_to_zarr_group[k_idx]
                return HedwigZarrImage(self.zarr_root[zarr_idx], self.ome_info, k_idx)

        return HedwigZarrImage(self.zarr_root[item])

    def series(self) -> Iterable[Tuple[str, HedwigZarrImage]]:
        for k in self.get_series_keys():
            yield k, self[k]
