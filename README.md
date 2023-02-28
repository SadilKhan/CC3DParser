# CC3DParser

The repository contains code for constructing vector from Json files in CC3D Parser. These vecotrs are in the format of DeepCAD input.

## Environment

```bash
$ conda env create --file=cc3dparser.yml
$ conda activate cc3dparser
```

## Data

Dataset is saved in `Sapphire` server in the following directory.

- `Json Files`: `/data/3d_cluster/CC3D-OP-SEQ/cc3d_v3.0_recs_extrude`.
- `BRep`: `/data/3d_cluster/CC3D-OP-SEQ/cc3d_v3.0_pid_steps_extrude`.

## Create Vectors from Json (Json --> H5)
Check if the `JSON_PATH` and `OUTPUT_PATH` are in correct format.

```bash
$ ./cc3dparser.sh
```

## Reconstruct Brep

- From `Json` files (Default)

```bash
./modelVisualizer.sh
```


- From `H5` Files

First create the vectors and store it in `OUTPUT_PATH` folder.
```bash
$ ./cc3dparser.sh
```
After that, change the `INPUT_DIR` in `modelVisualizer.sh` to the `OUTPUT_PATH` value and change the `FORM` to `json`.

```bash
./modelVisualizer.sh
```

## About Json

```mermaid
graph TD
    E1[ExtrudeFeature] -->|key: references| S1[Sketch];
    S1 --> |key: profiles| L1{Loop: List};
    L1 --> C1[Line];
    L1 --> C2[Circle];
    L1 --> C3[Arc];
```
### The keys in the json are described as below.

- `timeline`: The construction history of the cad model. It contains the `index` and `uuid`(The Id of the operation).

- `entity`: The information about the entity(Sketch,Extrusion).

- `extrude`: Gives the references of the Sketch Entity.

- `refAxis`: The extrusion axis. It contains the `start` coordinate and the `end` coordinate and the `direction` of the extrusion.

- `profiles`: The list of loops which constructs a sketch. Each loop contains multiple curves(Line/Arc/Circle).

- `curves`: Contains the information of a curve present in the profiles of the respective sketch.For a line, it contains `start` and `end` coordinates. For a circle, it contains `start`, `end` coordinates, `radius` and `normal` of the plane the curve is drawn at.

- `refPlane`: The plane of the sketch.