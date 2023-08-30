from pathlib import Path

import zarr
from typing import Optional, Iterable, Tuple, AnyStr
from pytools.utils.xml_info import OMEInfo
from pytools.HedwigZarrImage import HedwigZarrImage
import logging


logger = logging.getLogger(__name__)


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
