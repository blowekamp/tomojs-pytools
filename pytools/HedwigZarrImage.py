from pathlib import Path

import SimpleITK as sitk
import zarr
from typing import Optional, Tuple, Union, Dict, List
from pytools.utils.xml_info import OMEInfo
from pytools.zarr_extract_2d import zarr_extract_2d
import logging
import math
import re
import dask
from pytools.utils.histogram import DaskHistogramHelper, histogram_robust_stats, histogram_stats, weighted_quantile

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
        if self._ome_ngff_multiscale_dims()[1] == "C" and self.shape[1] == 1:
            return "Grayscale"
        return "MultiChannel"

    def neuroglancer_shader_parameters(self, mad_scale=3) -> dict:
        """
        Produces the "shaderParameters" portion of the metadata for Neuroglancer
        returns JSON serializable object
        """
        _shader_type = self.shader_type
        if _shader_type == "RGB":
            return {}
        if _shader_type == "Grayscale":
            stats = self._visual_min_max(mad_scale=5, clamp=True, channel=None)
            window = (stats["median"] - stats["mad"] * mad_scale, stats["median"] + stats["mad"] * mad_scale)
            window = (max(window[0], stats["min"]), min(window[1], stats["max"]))
            return {
                "window": [math.floor(window[0]), math.ceil(window[1])],
                "range": [math.floor(stats["min"]), math.ceil(stats["max"])],
            }

        if _shader_type == "MultiChannel":
            assert self._ome_ngff_multiscale_dims()[1] == "C"
            assert len(list(self.ome_info.channel_names(self.ome_idx))) == self.shape[1]
            color_sequence = ["red", "green", "blue", "cyan", "yellow", "magenta"]
            assert self.shape[1] <= len(color_sequence)

            json_channel_array = []

            for c, c_name in enumerate(self.ome_info.channel_names(self.ome_idx)):
                logger.debug(f"Processing channel: {c_name}")

                # replace non-alpha numeric with a underscore
                name = re.sub(r"[^a-zA-Z0-9]+", "_", c_name.lower())

                stats = self._visual_min_max(mad_scale=5, clamp=True, channel=c)
                window = (stats["median"] - stats["mad"] * mad_scale, stats["median"] + stats["mad"] * mad_scale)
                window = (max(window[0], stats["min"]), min(window[1], stats["max"]))

                json_channel_array.append(
                    {
                        "window": [math.floor(window[0]), math.ceil(window[1])],
                        "range": [math.floor(stats["min"]), math.ceil(stats["max"])],
                        "name": name,
                        "color": color_sequence[c],
                        "channel": c,
                        "clamp": False,
                        "enabled": True,
                    }
                )

            return {"brightness": 0.0, "contrast": 0.0, "channelArray": json_channel_array}

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
                arr_name = arr.name
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

    def _visual_min_max(self, mad_scale: float, clamp: bool = True, channel=None) -> Dict[str, List[int]]:
        """Processes the full resolution Zarr image. Dask is used for parallel reading and statistics computation. The
         global scheduler is used for all operations which can be changed with standard Dask configurations.

        :param mad_scale: The scale factor for the robust median absolute deviation (MAD) about the median to produce
         the "window range."

        :param clamp: If True then the minimum and maximum range will be clamped to the computed floor and limit values.


        :returns: The resulting dictionary will contain the following data elements with integer values:
        {
          "window": [ -571, 370 ],
          "range" : [ -53, 58]
        }
        Range is the minimum and maximum value in the dataset.

        """

        logger.debug(f"path: {self._ome_ngff_multiscale_image(0).path}")

        # extract channel
        assert self._ome_ngff_multiscale_dims()[1] == "C"
        if channel is not None:
            logger.info(f"Extracting channel {channel}..")
            da = dask.array.from_zarr(self._ome_ngff_multiscale_image(0))
            da = da[:, channel, ...]
        else:
            da = dask.array.from_zarr(self._ome_ngff_multiscale_image(0))

        histo = DaskHistogramHelper(da)

        logger.debug(f"dask.config.global_config: {dask.config.global_config}")

        logger.info(f'Building histogram for "{self.path}"...')
        h, bins = histo.compute_histogram(histogram_bin_edges=None, density=False)

        mids = 0.5 * (bins[1:] + bins[:-1])

        logger.info("Computing statistics...")
        stats = histogram_robust_stats(h, bins)
        stats.update(histogram_stats(h, bins))
        stats["min"], stats["max"] = weighted_quantile(mids, quantiles=[0.0, 1.0], sample_weight=h, values_sorted=True)
        logger.debug(f"stats: {stats}")

        return stats