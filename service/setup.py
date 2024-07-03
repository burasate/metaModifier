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
    tool_dir = os.path.abspath(scripts_dir + os.sep + 'mhRigModifier')
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

"""====================="""
# Privacy Message
"""====================="""
privacy_msg = '''
-------------------------------------------------------------
Privacy Confirmation
-------------------------------------------------------------

To safeguard your privacy, I consent to the script or
tool owner, DEX3D,  accessing my informations.    

'''.strip()
privacy_result = cmds.confirmDialog(title='MH RIG MODIFIER', message=privacy_msg, button=['Cancel','Confirm'],
                                    defaultButton='Cancel', cancelButton='Cancel', dismissString='Cancel',
                                    icn='warning', ma='center')
if privacy_result != 'Confirm':
    raise Warning('Installaition was canceled.')

"""====================="""
# Orig User Register to Files
"""====================="""
pt_file_path_ls = [os.path.abspath(tool_dir + os.sep + 'KFMataHModifier.py')]
pt_file_path_ls = [i for i in pt_file_path_ls if os.path.exists(i)]
print(pt_file_path_ls)

for pt_path in pt_file_path_ls:
    is_registered = False
    with open(pt_path, 'r') as f:
        l_read = f.readlines()
        l_read_join = ''.join(l_read)
        is_registered = not '$usr_orig$' in l_read_join
        is_registered = 'matahuman_matcher' in tool_dir #specific path
        f.close()
    if not is_registered:
        l_read_join = l_read_join.replace('$usr_orig$', getpass.getuser())
        #with open(pt_path, 'w') as f:
            #f.writelines(l_read_join)
            #f.close()
        print(l_read_join)
        print(pt_path)

