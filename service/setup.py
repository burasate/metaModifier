"""
MH RIG MODIFIER INSTALLER
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
install_path = os.path.abspath(tool_dir + os.sep + 'drag_n_drop_installer.py')
image_path = os.path.abspath(tool_dir + os.sep + 'KFMataHModifier.png')
cfg_path = os.path.dirname(install_path) + os.sep + 'config.json'
print([tool_dir, os.path.exists(tool_dir)])
print([install_path, os.path.exists(install_path)])
print([image_path, os.path.exists(image_path)])
print([cfg_path, os.path.exists(cfg_path)])

for fp in [install_path, image_path, cfg_path]:
    if not os.path.exists(tool_dir) or not os.path.exists(fp):
        error_msg = '''
        -------------------------------------------------------------
        Something went wrong about the installation.
        -------------------------------------------------------------
        > {1}
    
        please ensure the directory is placed correctly.
        e.g. {0}
    
        '''.format(tool_dir, fp).strip()
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
for pt_path in pt_file_path_ls:
    if 'matahuman_matcher' in tool_dir: # specific work path
        cmds.warning(tool_dir)
        break
    is_registered = False
    with open(pt_path, 'r') as f:
        l_read = f.readlines()
        l_read_join = ''.join(l_read)
        is_registered = not '$usr_orig$' in l_read_join
        f.close()
    if not is_registered:
        l_read_join = l_read_join.replace('$usr_orig$', getpass.getuser())
        with open(pt_path, 'w') as f:
            f.writelines(l_read_join)
            f.close()
        print(pt_path)

"""====================="""
# License Key
"""====================="""
def is_valid_format(s):
    if len(s) != 35:
        return False
    for i in [8, 17, 26]:
        if s[i] != '-':
            return False
    hex_digits = '0123456789ABCDEF'
    for i in range(35):
        if i in [8, 17, 26]:
            continue
        if s[i] not in hex_digits:
            return False
    return True

while True:
    lk_result = cmds.promptDialog(message='MH Rig Modifier\nLicense Key : ',button=['OK'])
    licene_key = cmds.promptDialog(q=1, text=1)
    if is_valid_format(licene_key):
        cfg = json.load(open(cfg_path))
        cfg['license_key'] = licene_key
        json.dump(cfg, open(cfg_path, 'w'), indent=4)
        print([licene_key])
        break

"""====================="""
# Shelf
"""====================="""
# Create Shelf
top_shelf = mel.eval('$nul = $gShelfTopLevel')
cur_shelf = cmds.tabLayout(top_shelf, q=1, st=1)
is_py3 = sys.version[0] == '3'

command_py3 = '''
# -----------------------------------
# MH RIG MODIFIER
# dex3d.gumroad.com
# -----------------------------------
import imp, os, sys
# -----------------------------------
if not r'{0}' in sys.path:
    sys.path.insert(0, r'{0}')
# -----------------------------------
import KFMataHModifier
imp.reload(KFMataHModifier)
mhm = KFMataHModifier.KFMetahumanModifier()
mhm.show_ui()
# -----------------------------------
'''.format(tool_dir).strip()

if is_py3:
    cmds.shelfButton(stp='python', iol='MataH', parent=cur_shelf, ann='KF MetaH Rig Modifier', i=image_path, c=command_py3)
    cmds.confirmDialog(title='MH RIG MODIFIER', message='Create shelf button successful', button=['OK'])
