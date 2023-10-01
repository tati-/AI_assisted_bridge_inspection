import os
import sys
import pdb
import bpy
import numpy as np
import addon_utils
from .decorators import forall

# correspondance of bridge elements to ifc classes
ifc_classes = { 'traverse': 'IfcRoof',
                'piedroit': 'IfcColumn',
                'mur':'IfcWall',
                'gousset': 'IfcSlab',
                'corniche': 'IfcPlate',
                'ground': 'IfcFooting'
                }


if sys.platform == 'darwin':
    blenderbim_path = 'addons/blenderbim-230506-py310-macosm1.zip'
elif sys.platform == 'linux':
    blenderbim_path = 'addons/blenderbim-230504-py310-linux.zip' #'addons/blenderbim-230304-py310-linux.zip'


def enable_blenderBIM():
    """
    enable blenderBIM if it is not enabled, install it and then enable it
    if it is not installed
    """
    # if addon is enabled, do nothing
    if 'blenderbim' in [ad.module for ad in bpy.context.preferences.addons]:
        return
    blenderbim = [mod for mod in addon_utils.modules() if mod.bl_info['name'].lower()=='blenderbim']
    if blenderbim:
        # if addon is installed but non enabled, enable it
        bpy.ops.preferences.addon_enable(module='blenderbim')
    else:
        bpy.ops.preferences.addon_install(filepath=os.path.abspath(blenderbim_path))
        bpy.ops.preferences.addon_enable(module='blenderbim')
    # save user preferences
    bpy.ops.wm.save_userpref()


@forall
def assign_ifc_classes(object):
    """
    assign an ifc class to an object, based on the collection it belongs to
    and the ifc_classes dictionary defined above
    """
    if object.users_collection[0].name not in ifc_classes.keys():
        return
    # deselect possibly selected objects
    [obj.select_set(False) for obj in bpy.data.objects]
    # select object
    object.select_set(True)
    # assign ifc class
    bpy.ops.bim.assign_class(ifc_class=ifc_classes[object.users_collection[0].name])
    # move object to IFC storey
    bpy.data.collections['IfcBuildingStorey/My Storey'].objects.link(object)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ----------------------------DEBUGGING-----------------------------
# for attr in dir(variable):
#     print('{} : {}'.format(attr, getattr(variable, attr)))
# ------------------------------------------------------------------
