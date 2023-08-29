import click
import logging
from pathlib import Path
from pytools import __version__
from pytools.HedwigZarrImages import HedwigZarrImages
import SimpleITK as sitk


@click.command()
@click.argument("input_zarr", type=click.Path(exists=True, dir_okay=True, readable=True, path_type=Path))
@click.option(
    "--show",
    is_flag=True,
    show_default=True,
    default=False,
    help="Use SimpleITK Show function to display images at 1024x1024 resolution.",
)
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False)
)
@click.version_option(__version__)
def main(input_zarr, log_level, show):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.getLevelName(log_level))

    zi = HedwigZarrImages(input_zarr)
    keys = list(zi.get_series_keys())
    print(f"zarr groups: {keys}")

    print(zi.ome_xml_path)

    for k in zi.get_series_keys():
        z_img = zi[k]
        print(f'image name: "{k}"')
        for level in range(len(z_img._ome_ngff_multiscales(idx=0)["datasets"])):
            arr = z_img._ome_ngff_multiscale_get_array(level)
            print(f"\tarray level {level}: {arr} {arr.nchunks} chunks of {arr.chunks} ")
        print(f"\tzarr path: {z_img.path}")
        print(f"\tdims: {z_img.dims}")
        print(f"\tshader type: {z_img.shader_type}")
        print(f"\tNGFF dims: {z_img._ome_ngff_multiscale_dims()}")

        if show:
            viewer = sitk.ImageViewer()
            if z_img.shader_type == "RGB":
                img = z_img.extract_2d(1024, 1024, auto_uint8=True)
                # rotate label images counterclockwise by 90 degrees.
                if k == "label image":
                    img = sitk.Flip(sitk.PermuteAxes(img, [1, 0]), [True, False])
            elif z_img.shader_type == "MultiChannel":
                img = z_img.extract_2d(1024, 1024)
                viewer.SetFileExtension(".nii")
            else:
                img = z_img.extract_2d(1024, 1024)

            viewer.Execute(img)

        print(f"\tshader params: {z_img.neuroglancer_shader_parameters()}")


if __name__ == "__main__":
    main()
