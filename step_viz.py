import argparse
import logging


# from OCC.Display.SimpleGui import init_display
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_WHITE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepBndLib import brepbndlib_Add

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_1", type=str, required=True, help="path to the first step file"
    )
    parser.add_argument(
        "--path_2", type=str, default=None, help="path to the second step file"
    )

    args = parser.parse_args()
    return args


def translate_shape(shape, translate):
    trans = gp_Trsf()
    trans.SetTranslation(gp_Vec(translate[0], translate[1], translate[2]))
    loc = TopLoc_Location(trans)
    shape.Move(loc)
    return shape


def get_shape_width(shape, tol=1e-6, use_mesh=True):
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    if use_mesh:
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        if not mesh.IsDone():
            raise AssertionError("Mesh not done.")
    brepbndlib_Add(shape, bbox, use_mesh)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmax - xmin


def load_step(path):
    shape = read_step_file(path)
    return shape


def main():
    args = parse_args()
    display, start_display, add_menu, add_function_to_menu = init_display()
    logger.info(f"Displaying shape: {args.path_1}")
    shape1 = load_step(args.path_1)

    # For loop for the pairs of paths
    display.DisplayShape(shape1, update=True)
    if args.path_2:
        logger.info(f"Displaying shape: {args.path_2}")
        shape2 = load_step(args.path_2)
        x_shift = get_shape_width(shape1)
        shape2 = translate_shape(shape2, [2 * x_shift, 0, 0])
        display.DisplayShape(shape2, update=True)

    display.FitAll()

    display.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True,
    )
    start_display()
    ####


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
