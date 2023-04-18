# Generation of synthetic dataset

We use [Blender](https://www.blender.org/) and its python API ([bpy](https://docs.blender.org/api/current/index.html)) to generate a set of Bridge images with arbitrary environmental conditions and angles of view.

This dataset will be used to train an AI model.

## Prerequisites

Python version [3.10.2](https://www.python.org/downloads/release/python-3102/) with Blender built as [python module](https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule).

`requirements.txt` contains the necessary packages.

**If a virtual environment is used in MacOS, it should be installed with `--enable-framework` option**

## Usage

The code is divided in a number of modules.
1. ``generate_3d_semantic_models.py`` generates blender geometric models starting from a set of ``.json`` files and a label dictionary. The resulting blender file has one object per building block, and collections to group the components belonging to the same semantic class.
2. ``fill_terrain.py`` takes as input a set of blender files ([wildcards](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm) allowed if on unix) of-for the moment-PIPO bridges, and fills the road and terrain around them. The wildcard can be of the form `/path/to/blend_files/bridge*.blend` for all bridges, or `/path/to/blend_files/bridge{10..20}.blend` for a specific range.
3. ``generate_synthetic_data.py`` takes as input a number of blender files (as in ``fill_terrain.py``), adds textures and environmental conditions, and produces a number of images along with their corresponding ground truth. <span style="color:red">*For the moment this script can crush and stop because of a rendering issue in Blender that I cannnot yet control*.</span>
4. ``refine_dataset.py`` takes as input a dataset folder and a ratio denoting the acceptable amount of positive (not background) class that should be in each image for the image to be considered useful, and cleans up the dataset.
5. ``dataset_overview_and_stats.py`` takes as input a dataset folder and produces a number of figures showing the image side by side with its corresponding groundtruth, and also produces some more demos and dataset statistics.

>**_NOTE:_** The functionalities of points 4. and 5. above are integrated in the script of point 3. However they are also provided seperately, in order to allow for post processing, or for processing in case the script ``generate_synthetic_data.py`` was run only partially.

<!-- ```sh
python generate_synthetic_dataset.py -input /path/to/base/xxx.json -params /path/to/parameter/set/zzz.json -textures /path/to/textures/directory -cl /path/to/semantic/classes/file -frames 2 -bridges 2
```
### Python script arguments

|Argument|Argument type| Interpretation|Default value|Required|
|--------|-------------|---------------|-------------|--------|
|`-input`|`string` value|input file path [acceptable formats: `.json`]|-|&#10003;|
|`-cl, -classes`|`string` value|path to .txt file containing the class descriptions|-|&#10008;|
|`-savefolder`|`string` value|folder to save generated dataset|same as input file directory|&#10008;|
|`-resx`|`int` value|X resolution (width) of output images |640|&#10008;|
|`-resy`|`int` value|Y resolution (height) of output images |480|&#10008;|
|`--struct-cov`|`float` value [0,1]|minimum coverage, percentage of structure (bridge) that should be in the frame for the image to be kept in the dataset|0|&#10008;|
|`--img-cov`|`float` value [0,1]|minimum coverage, percentage of image that contains the structure, for the image to be kept in the dataset|0|&#10008;|
|`-frames`|`int` value|number of camera angle views per bridge setup|1|&#10008;|
|`-bridges`|`int` value|number of different bridges (in terms of geometry) to be generated|1|&#10008;|
|`-params`|`string` value|path to parameter set file, allowing to generate different bridge geometries[acceptable formats: .json]|-|&#10008;|
|`-textures`|`string` value|path to blender file containing textures  [acceptable formats: `.blend`]|-|&#10008;|
|`--timetest`|`boolean` value|if set, run several tests with increasing number of mesh density and plot processing time|`False`|&#10008;| -->

## Input base file

A [`.json`](https://www.json.org/json-en.html) file is given, that contains all the geometrical parameters in order to generate an [`.obj`](https://en.wikipedia.org/wiki/Wavefront_.obj_file) file with its labels.

The `.obj` file defines a mesh representing a bridge instance, where the texture assigned to each face serves as a label for the face semantic category.

An abstract overview of the base `.json`:

<img src="./README_figures/base_json_element.svg" width="1000" height="500" />

The `.json` file contains a list of elements, each element representing a 3d block, in the form of an hexahedron. An example entry looks as follows:
```json
{
  "Label":"traverse",
	"Nom":"traverse-S",
	"Dimensions":
	{
		"Longueur":
		{
			"D1":6,
			"D2":6,
			"Dec":-0.05,
			"Contrainte":-1,
			"ContrainteDec":-1
		},
		"Largeur":
		{
			"D1":10,
			"D2":10,
			"Dec":0,
			"Contrainte":-1,
			"ContrainteDec":-1
		},
		"Hauteur":
		{
			"D1":0.5,
			"D2":0.5,
			"Dec":0,
			"Contrainte":-1,
			"ContrainteDec":-1
		}
	},
	"Angle":
	{
		"Heading":
		{
			"V":0,
			"Contrainte":-1
		},
		"Tilt":
		{
			"V":5,
			"Contrainte":-1
		},
		"Roll":
		{
			"V":2.5,
			"Contrainte":-1
		}
	},
	"Contrainte":
	{
		"Element":-1,
		"N1":-1,
		"N2":-1
	}
}
```

### Dimensions
  The conventions are as follows:
  - Largeur (width): along x-axis
  - Longueur (length): along y-axis
  - Hauteur (height): along z-axis

  The `D1`, `D2` and `Dec` (offset) parameters are described in the pictures below for each one of the dimensions.
<table style="margin-left:auto;margin-right:auto;">
  <tr>
    <td>
      <img src="./README_figures/largeur.svg" width="200" height="200" />
    </td>
    <td>
      <img src="./README_figures/longeur.svg" width="200" height="200" />
    </td>
    <td>
      <img src="./README_figures/hauteur.svg" width="200" height="200" />
    </td>
   </tr>
   <tr>
      <td>Largeur</td>
      <td>Longeur</td>
      <td>Hauteur</td>
  </tr>
</table>

An overview of the dimensions:
<center><img src="./README_figures/dimension_overview.svg" width="500" height="500" /></center>

When $`D1 \neq D2`$ the face becomes a *trapezoid*.

The **offset parameter** (`Dec`) allows for a face to be tilted, as is shown here
<img src="./README_figures/decalage.svg" width="200" height="200" />

The **dimension constraint parameters**(`Contrainte` and `ContrainteDec` inside the dimension object) tell us whether the corresponding dimension depends on another (parent) building block. The value is a pointer to the parent building block of the `.json` list. The convention is as follows (zero-based):
- Value = -1 : no constraints
- Value = 0 : dimension is equal to the corresponding dimension on the 1st list element (the current value is overwritten)
- Value = 1 : dimension is equal to the corresponding dimension on the 2nd list element (the current value is overwritten)
- etc.

### Angles
The **rotation** angle conventions are as follows:
  - Roll : ccw around x-axis, in degrees
  - Tilt: ccw around y-axis, in degrees
  - Heading: ccw around z-axis, in degrees

**Angle constraints** follow the same logic as the dimension constraints.

### Block constraints
The **block constraint parameters**(`Contrainte` inside the building block object) tells us whether this block should be positioned attached to another (parent) block. In that case, a specific node of this building block should be forced to have the same coordinates as a node in the parent block.
- `Element` : Parent list element, index conventions same as with dimension constraint parameters.
- `N1` : parent node
- `N2` : child node

The node numbering is shown here <img src="./README_figures/node_numbering_json.svg" width="200" height="200" />

`N1` and `N2` values of -1 denote no constraints. A value of 0 points to node 0, a value of 1 points to node 1 etc (zero-based).

## Parameter tuning
To each base file corresponds another `.json` file containing the rules for generating a random bridge while respecting some constraints.

The parameter file contains a list of elements, each element corresponding either to an entire building block category, or to a specific block.

An abstract overview of the base `.json`:
<img src="./README_figures/param_json_element.svg" width="1000" height="500" />

The entry describes which of the properties of the block are to be set, and by what rules. We can have:
- Property values in relation to other property values. In that case, a first degree equation of an arbitrary number of properties can be calculated.
- Min and Max value bounds
- relative Min and Max values, depending on the value of some other property.
- Max probability (`MaxProbability`), denoting the probability that an element attains its maximum value.

*In the case that both absolute and relative bounds are given, the more conservative option is retained.*

An example entry is:

```json
{
"Label": "mur",
"Nom": "mur-SW",
"Necessaire": false,
"Dimensions": {
    "Hauteur": [{
        "Affectation": ["Dec"],
        "Coefficient": null,
        "rMin": {
            "Dependance": [{
                "Label": "mur",
                "Nom": "mur-SW",
                "Contrainte": ["Dimensions", "Hauteur", "D1"]
            },
            {
                "Label": "mur",
                "Nom": "mur-SW",
                "Contrainte": ["Dimensions", "Hauteur", "D2"]
            }],
            "Coefficient": [{
                "Valeur": 0.5,
                "Fix": true
                },
                {
                    "Valeur": -0.5,
                    "Fix": true
                }],
            "Constante": 1.5
        }       
    }]
}
}
```

For that case, the `Dec` of the `Hauteur` for the element `mur-SW` is calculated as follows:

$`0.5\times D1_{hauteur, mur-SW} - 0.5\times D2_{hauteur, mur-SW} + 1.5`$

If the `Necessaire` is set to `false`, this block might not be generated in the 3D model.

>**_NOTE:_**  If more than 1 json entry levels affect a building block, the last one is retained for the dimensions and angles. For the visibility, a former entry may still have an effect.

## Labels

The semantic category of each mesh face is given in the form of textures.

A texture is assigned to each face, indicating that face's category.

A dictionary in the form of a simple `.txt` file lists all possible semantic categories, based on [IQOA](http://piles.cerema.fr/iqoa-ponts-r450.html).

If not given, the semantic categories are inferred from the `Label` entry of the `.json` file, and subsequently the material assigned to each face in the `.obj` file.

An example of a class dictionary:
```txt
piedroit 1
traverse 2
mur 3
gousset 4
corniche 5
```
>**_NOTE:_**  Label 0 should be reserved for the background class, otherwise the resulting masks cannot not be trusted.


## Angles of view

Initially the camera is positioned 20 meters in front of the bridge. rotated so that it has a **clear** and **centered** view of the bridge. After that, random locations and rotations are sampled, from a predefined range.

###  Visibility check

*Before rendering* a check is made to verify if a substantial part of the bridge is in the camera view. The desired percentage of bridge visibility is given as an argument to the python script (`--struct-cov`).

### Usability of rendered image

*After rendering* the image is checked, in order to see if a substantial part of the image contains a bridge (since the random camera sampling may result in images with little or no bridge visible). The images that contain an insufficient percentage of bridge are discarded. The percentage is given in the form of an argument in the python script (`--img-cov`).

## Environment

### Sky

[Dynamic Sky addon](https://docs.blender.org/manual/en/latest/addons/lighting/dynamic_sky.html) is used to generate the sky.

The parameters are randomly sampled from a range of realistic values.

### Mist

Mist is added to the scene, with a random selection of range and opacity. The [mist pass node](https://docs.blender.org/manual/en/latest/render/cycles/world_settings.html) of Blender is used in the rendering of the scene.

### Ground

A terrain is constructed around the bridge as a rough set of connecting triangles (`Terrain` class).

After this first initialisation, the terrain mesh is subdivided (for more detail), and each vertex is randomly slightly displaced (the vertices that are connected to the bridge or the ground are excluded from this displacement, so that no evident holes appear).

### Textures

Textures from the [Poly Haven](https://polyhaven.com/textures) database are downloaded.
A blender file is created that packs all the textures as assets. Asset library catalogs are also created with the groups of elements that need different textures, e.g. ground, road, walls.

The path to a blender file containing the textures is given as argument to the main script.

The textures are then read and added (linked) as materials to the working file.

>**_NOTE:_**  Even though the catalog library is not preserved when the textures file is linked (since the `.txt` file associated with it is not referenced), the
asset data maintain the catalog name information. The category can therefore be read in python, and used to apply appropriate textures to appropriate parts of the
3D model.


## Output

The folder where the dataset should be saved is given as an argument to the python script. Inside it, a "wrapper" folder is created.

Into this folder, a folder `images` and a folder `masks` is created. Inside the `masks` folder, there is a folder for each semantic category. The labels are saved in the form of binary masks.
