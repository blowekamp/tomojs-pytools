#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from pytools.HedwigZarrImages import HedwigZarrImages
from pathlib import Path
import pytest

import json


zarr_path = Path("/Users/blowekamp/scratch/hedwig/TestData/3D/CryoTomo/SARsCoV2_1.zarr")


@pytest.mark.parametrize(
    "zarr_path",
    [
        Path(
            "/Users/blowekamp/scratch/hedwig/TestData/Nanostringfiles/Example IHC images from Axio/"
            "2023_03_10_P2_S8_FL.zarr"
        ),
        Path("/Users/blowekamp/scratch/hedwig/TestData/3D/CryoTomo/SARsCoV2_1.zarr"),
    ],
)
def test_HedwigZarrImage_gray(zarr_path):
    zi = HedwigZarrImages(zarr_path)
    keys = list(zi.get_series_keys())
    print(f"zarr groups: {keys}")

    print(zi.ome_xml_path)

    for k in zi.get_series_keys():
        z_img = zi[k]
        print(f'image name: "{k}"')
        print(f"\tarray shape: {z_img.shape}")
        print(f"\tzarr path: {z_img.path}")
        print(f"\tdims: {z_img.dims}")
        print(f"\tshader type: {z_img.shader_type}")
        print(f"\tNGFF dims: {z_img._ome_ngff_multiscale_dims()}")
        print(f"\tshader params: {json.dumps(z_img.neuroglancer_shader_parameters())}")
    return
