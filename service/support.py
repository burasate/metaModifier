print('Attempting to load add-on plugins...')
import os, json, sys, getpass

"""====================="""
# Update version of mhRigModifier
"""====================="""
def update_version(*_):
    import os, base64, getpass
    is_py3 = sys.version[0] == '3'
    if is_py3:
        import urllib.request as uLib
    else:
        import urllib as uLib

    try:
        tool_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        maya_app_dir = mel.eval('getenv MAYA_APP_DIR')
        scripts_dir = os.path.abspath(maya_app_dir + os.sep + 'scripts')
        tool_dir = os.path.abspath(scripts_dir + os.sep + 'mhRigModifier')

    install_path = os.path.abspath(tool_dir + os.sep + 'drag_n_drop_installer.py')
    image_path = os.path.abspath(tool_dir + os.sep + 'KFMataHModifier.png')
    #print(tool_dir, os.path.exists(tool_dir))
    #print(install_path, os.path.exists(install_path))
    #print(image_path)

    """====================="""
    # Orig User Register to Files
    """====================="""
    py_ls = [
        'KFMataHModifier.py',
        'src/blendshape_combiner.py',
        'src/dna_manager.py',
        'src/pose_wrangler.py'
    ]
    py_file_path_ls = [tool_dir + os.sep + i for i in py_ls]
    src_url = 'https://raw.githubusercontent.com/burasate/metaModifier/main/service/update'
    for py_path, src_py in zip(py_file_path_ls, py_ls):
        if not os.path.exists(py_path): continue
        print(py_path)
        #print(py_path[len(tool_dir):])
        #print(src_url + py_path[len(tool_dir):].replace('\\','/'))
        url = src_url + '/' + src_py
        if 'matahuman_matcher' in tool_dir:  # specific work path
            cmds.warning(tool_dir)
            break

        response = uLib.urlopen(url)
        read = response.read()
        read = read.decode('utf-8') if type(read) == type(b'') else read
        username = getpass.getuser()
        u_read = read.replace('$usr_orig$', username)
        print('------------')
        #print(u_read)
        print('url', url, py_path)
        #with open(py_path, 'w') as f:
            #f.writelines(u_read)
            #f.close()
            #print('{}  is  updated..'.format(py_path))
#---------------
update_version()
#---------------

'''========================================='''
# Queue Task Func
'''========================================='''
def add_queue_task(task_name, data_dict):
    global sys,json
    is_py3 = sys.version[0] == '3'
    if is_py3:
        import urllib.request as uLib
    else:
        import urllib as uLib

    if type(data_dict) != type(dict()):
        return None

    data = {
        'name': task_name,
        'data': data_dict
    }
    data['data'] = json.dumps(data['data'], indent=4, sort_keys=True, ensure_ascii=False)
    url = 'https://script.google.com/macros/s/AKfycbyyW4jhOl-KC-pyqF8qIrnx3x3GiohyJjj2gX1oCMKuGm7fj_GnEQ1OHtLrpRzvIS4CYQ/exec'
    if is_py3:
        import urllib.parse
        params = urllib.parse.urlencode(data)
    else:
        params = uLib.urlencode(data)
    params = params.encode('ascii')
    conn = uLib.urlopen(url, params)

'''========================================='''
# Check in
'''========================================='''
if sys.version_info.major >= 3:
    import urllib.request as uLib
else:
    import urllib as uLib
import datetime, getpass
from time import gmtime, strftime
try:
    user_data = {
        'script_name': 'MH Rig Modifier',
        'date_time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #'used': self.usr_data['used'],
        #'days': self.usr_data['days'],
        #'license_email': self.usr_data['license_email'],
        'ip':str(uLib.urlopen('http://v4.ident.me').read().decode('utf8')),
        'os' : str(cmds.about(operatingSystem=1)),
        'license_key' : self.func.config['license_key'] if 'license_key' in self.func.config else '',
        'script_path' : '' if __name__ == '__main__' else os.path.abspath(__file__).replace('pyc', 'py'),
        #'namespac_ls' : ','.join(cmds.namespaceInfo(lon=1)[:10]),
        'maya' : str(cmds.about(version=1)),
        'script_version' : str(self.version),
        'timezone' : str( strftime('%z', gmtime()) ),
        'scene_path' : cmds.file(q=1, sn=1),
        'time_unit' : cmds.currentUnit(q=1, t=1),
        'user_last' : getpass.getuser(),
        'user_orig' : self.user_original,
        #'fps' : scene.get_fps(),
    }
    #user_data['email'] = user_data['license_email'] if '@' in user_data['license_email'] else '{}@trial.com'.format(
        #user_data['user_last'].lower())
    add_queue_task('script_tool_check_in', user_data)
    del user_data
except:
    import traceback
    add_queue_task('checkin_error', {
        'error': str(traceback.format_exc()),
        'user_orig': getpass.getuser()
    })