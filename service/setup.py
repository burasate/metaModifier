"""
KF OVERLAP SHELF INSTALLER
"""
from maya import cmds
from maya import mel
import os, json, sys, getpass

"""====================="""
# Init
"""====================="""
try:
    tool_dir = os.path.dirname(os.path.abspath(__file__))
except:
    maya_app_dir = mel.eval('getenv MAYA_APP_DIR')
    scripts_dir = os.path.abspath(maya_app_dir + os.sep + 'scripts')
    tool_dir = os.path.abspath(scripts_dir + os.sep + 'KFMataHModifier')
install_path = os.path.abspath(tool_dir + os.sep + 'Install.py')
image_path = os.path.abspath(tool_dir + os.sep + 'KFMataHModifier.png')
print([tool_dir, os.path.exists(tool_dir)])
print([install_path, os.path.exists(install_path)])
print(image_path)

if not os.path.exists(tool_dir) or not os.path.exists(install_path):
    error_msg = '''
    -------------------------------------------------------------
    Something went wrong about the installation.
    -------------------------------------------------------------

    please ensure the directory is placed correctly.
    e.g. {0}

    '''.format(tool_dir).strip()
    cmds.confirmDialog(title='', message=error_msg, button=['OK'], icn='critical', ma='center')
    raise Warning('WARNING!!\ndo not found \"Install file\" in {}'.format(tool_dir))