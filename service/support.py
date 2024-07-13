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
    print(tool_dir, os.path.exists(tool_dir))
    print(install_path, os.path.exists(install_path))
    print(image_path)

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
    #pt_file_path_ls = [i for i in pt_file_path_ls if os.path.exists(i)]
    src_url = 'https://raw.githubusercontent.com/burasate/metaModifier/main/service/update'
    for py_path, src_py in zip(py_file_path_ls, py_ls):
        if not os.path.exists(py_path): continue
        print(py_path)
        print(py_path[len(tool_dir):])
        print(src_url + py_path[len(tool_dir):].replace('\\','/'))
        url = src_url + '/' + src_py
        print(url)
        if 'matahuman_matcher' in tool_dir:  # specific work path
            cmds.warning(tool_dir)
            break

        continue
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

update_version()