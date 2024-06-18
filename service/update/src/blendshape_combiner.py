from maya import mel
import maya.cmds as cmds
import imp, os, sys
import math
# -----------------------------------
if not r'D:\GDrive\Documents\2024\matahuman_matcher\libs' in sys.path:
    sys.path.insert(0, r'D:\GDrive\Documents\2024\matahuman_matcher\libs')
# -----------------------------------
import numpy as np

class util:
	@staticmethod
	def poly_select_traverse(mode=int):
		'''
		mode 1 extend; 2 inside; 3 border;
		'''
		traverse_cmd = 'select `ls -sl`; PolySelectTraverse {}; select `ls -sl`;'.format(mode)
		mel.eval(traverse_cmd)
		cmds.refresh(f=1)

	@staticmethod
	def ease_in(t):
		return t * t * t

	@staticmethod
	def ease_out(t):
		return (t - 1) ** 3 + 1

	@staticmethod
	def ease_in_out(x):
		denominator = 1 + 10 ** (5 * 0.5)
		simplified_denominator = 1 - 2 / denominator
		numerator = 1 / (1 + 10 ** (5 * (0.5 - x))) - 1 / denominator
		result = numerator / simplified_denominator
		return result

	@staticmethod
	def get_vertices_positions(mesh):
		cmds.refresh() #Force refresh before get data
		vertices = cmds.ls(mesh + ".vtx[*]", flatten=1)
		vertices_ls = [cmds.pointPosition(i, w=1) for i in vertices]
		return vertices_ls

	@staticmethod
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

	@staticmethod
	def assign_vertex_color(obj_ls, color):
		[cmds.setAttr(i + '.displayColors', 1) for i in cmds.ls(type='mesh')
		 if cmds.objExists(i + '.displayColors') and cmds.getAttr(i + '.displayColors') != 1]
		color_float = [float(i) for i in color]
		cmds.select(obj_ls, replace=1)
		cmds.polyColorPerVertex(rgb=color_float)
		cmds.select(cl=1)

	@staticmethod
	def delete_vertex_color(obj_ls):
		cmds.polyColorSet(obj_ls, delete=1)
		[cmds.setAttr(i + '.displayColors', 0) for i in cmds.ls(type='mesh') if cmds.objExists(i + '.displayColors')]

class skin_cluster:
	@staticmethod
	def smooth_skin_weight_selected():
		sel_vtx = cmds.ls('*.vtx[*]', flatten=1, sl=1)
		if sel_vtx == []:
			return
		mesh = sel_vtx[0].split('.')[0]
		hist = cmds.listHistory(mesh)
		skn = cmds.ls(hist, type='skinCluster')[0]
		cmds.skinCluster(skn, edit=1, smoothWeights=0, smoothWeightsMaxIterations=15, obeyMaxInfluences=0)
		expand_ls = []
		for i in range(5):
			util.poly_select_traverse(mode=1)
			cur_select = cmds.ls('*.vtx[*]', flatten=1, sl=1)
			expand_ls += [i for i in cur_select if not i in expand_ls]
			cmds.select(expand_ls)
			util.poly_select_traverse(mode=3)
			cmds.skinCluster(skn, edit=1, smoothWeights=0, smoothWeightsMaxIterations=15, obeyMaxInfluences=0)
			cmds.refresh(f=1)
		cmds.select(sel_vtx)

class blenshape_transfer():
	def __init__(self, mesh, bs):
		self.mesh, self.bs, self.exp = [mesh, bs, 'CTRL_expressions']
		self.orig_skn_verticles, self.orig_delta, self.orig_bs_verticles = None, None, None
		self.w_name_ls = cmds.listAttr('{}.w'.format(self.bs), m=1)
		if cmds.objExists(self.exp):
			self.exp_attr_ls = cmds.listAttr(self.exp, k=1)
		ma_name = os.path.basename(cmds.file(q=1, sn=1)).split('.')[0]
		ma_dir = os.path.dirname(cmds.file(q=1, sn=1))
		self.orig_skn_vtx_fn, self.orig_delta_fn, self.orig_bs_vtx_fn = [
			'{}_{}_skn_vtx'.format(mesh, bs),
			'{}_{}_skn_delta'.format(mesh, bs),
			'{}_{}_bs_vtx'.format(mesh, bs)
		]
		self.orig_skn_vtx_fp, self.orig_delta_fp, self.orig_bs_vtx_fp = [
			'{}/{}_{}.npy'.format(ma_dir, ma_name, self.orig_skn_vtx_fn),
			'{}/{}_{}.npy'.format(ma_dir, ma_name, self.orig_delta_fn),
			'{}/{}_{}.npy'.format(ma_dir, ma_name, self.orig_bs_vtx_fn)
		]
		self.exp_con_ls = None
		self.reset_all_variables()
		self.get_blendshape_verticles_cache()

	def reset_all_variables(self):
		self.new_verticles = None
		self.target_vtx_pos_dict = {}
		self.delta_vtx_pos_dict = {}
		self.mse_dict = {}
		self.bs_weight_dict = {}
		self.bs_area_w_dict = {}
		self.sum_area_w_dict = {}

	def init_area_weight(self):
		# init areas weight (orig_delta)
		self.area_vtx_w = np.abs(np.copy(self.orig_delta))
		for idx in range(self.area_vtx_w.shape[0]):  # conver 3 vector delta to [distance] * 3
			dst_np = np.linalg.norm(self.area_vtx_w[idx], axis=1)  # merge to distance
			self.area_vtx_w[idx] = np.tile(dst_np[:, np.newaxis], 3)  # assing distance to [distance] * 3
		self.area_vtx_w = self.get_normalized(self.area_vtx_w)  # normalize

	def toggle_expression(self, connect=True):
		if self.exp_con_ls is None:
			self.exp_con_ls = [cmds.listConnections('{}.{}'.format(self.exp, n), s=1, plugs=1, d=0) for n in
							   self.exp_attr_ls]
			for i in range(len((self.exp_con_ls))):
				self.exp_con_ls[i] = self.exp_con_ls[i][0] if self.exp_con_ls[i] else self.exp_con_ls[i]
			if self.exp_con_ls == []:
				cmds.select(self.exp)
				raise Warning('No any connections from Expression - Please undo')

		if not connect:
			[cmds.disconnectAttr(self.exp_con_ls[i], self.exp + '.' + self.exp_attr_ls[i])
			 for i in range(len(self.exp_attr_ls))  if self.exp_con_ls[i]]
			[cmds.setAttr(self.exp + '.' + self.exp_attr_ls[i], 0.0) for i in range(len(self.exp_attr_ls))]
		else:
			[cmds.connectAttr(self.exp_con_ls[i], self.exp + '.' + self.exp_attr_ls[i]) for i in
			 range(len(self.exp_attr_ls)) if self.exp_con_ls[i]]


	def get_blendshape_verticles_cache(self):
		bs_con_ls = [cmds.listConnections('{}.{}'.format(self.bs, n), s=1, plugs=1, d=0) for n in self.w_name_ls]
		bs_con_ls = [i[0] for i in bs_con_ls if i]
		#self.exp_con_ls = [cmds.listConnections('{}.{}'.format(self.exp, n), s=1, plugs=1, d=0) for n in self.exp_attr_ls]
		#self.exp_con_ls = [i[0] for i in self.exp_con_ls if i]
		#print(bs_con_ls)
		#print(self.exp_attr_ls)
		#print(exp_con_ls)
		if bs_con_ls == []:
			raise Warning('No any connections from Blendshape - Please undo')
		# -----------------------------------

		is_loaded = False
		if os.path.exists(self.orig_skn_vtx_fp) and os.path.exists(self.orig_delta_fp) and os.path.exists(self.orig_bs_vtx_fp):
			print('found data...')
			self.orig_skn_verticles = np.load(self.orig_skn_vtx_fp).astype(float)
			self.orig_bs_verticles = np.load(self.orig_bs_vtx_fp).astype(float)
			self.orig_delta = np.load(self.orig_delta_fp).astype(float)
			#print(self.orig_skn_verticles)
			#print('orig_verticles.shape', self.orig_skn_verticles.shape)
			self.init_area_weight()
			if self.orig_skn_verticles.shape[0] == len(self.w_name_ls):
				print('bleanshape count is matched from data...')
				is_loaded = True
		# -----------------------------------

		if not is_loaded:
			util.toggle_envelope(enable=0, skinCluster=0, blendShape=1)
			util.toggle_envelope(enable=1, skinCluster=1, blendShape=0)
			self.toggle_expression(connect=0)
			#[cmds.disconnectAttr(self.exp_con_ls[i], self.exp + '.' + self.exp_attr_ls[i]) for i in range(len(self.exp_attr_ls))]
			#[cmds.setAttr(self.exp + '.' + self.exp_attr_ls[i], 0.0) for i in range(len(self.exp_attr_ls))]
			# -----------------------------------

			# base shape data
			base_vtx_pos = util.get_vertices_positions(self.mesh)
			base_vtx_pos_np = np.array(base_vtx_pos).reshape(len(base_vtx_pos), 3)
			#print(base_vtx_pos_np)
			#print(base_vtx_pos_np.shape)
			# -----------------------------------

			self.orig_skn_verticles = np.array([base_vtx_pos_np] * len(self.w_name_ls))
			self.orig_bs_verticles = np.copy(self.orig_skn_verticles)
			self.orig_delta = np.zeros(shape=[len(self.w_name_ls)] + list(base_vtx_pos_np.shape))
			#print('orig_verticles', self.orig_skn_verticles.shape)
			#print('orig_delta', self.orig_delta.shape)
			# -----------------------------------

			bs_idx_zip = [(i, '{}.{}'.format(self.bs, n)) for i, n in list(zip(range(len(self.w_name_ls)), self.w_name_ls))]
			#print(bs_idx_zip)
			# -----------------------------------

			for idx in range(len(self.exp_attr_ls)):
				cmds.setAttr('{}.{}'.format(self.exp, self.exp_attr_ls[idx]), 1.0)
				cmds.refresh()
				print('\n{:03d} - {}.{}'.format(idx, self.exp, self.exp_attr_ls[idx]))
				print('----------------------------')

				bs_ls = [i for i in bs_idx_zip if cmds.getAttr(i[1]) > .98] # == 1.0
				if bs_ls == []:
					cmds.setAttr('{}.{}'.format(self.exp, self.exp_attr_ls[idx]), .0)
					continue
				bs_idx, bs_name = bs_ls[0]
				print(bs_ls[0])

				# enable only skinCluster
				util.toggle_envelope(enable=0, skinCluster=0, blendShape=1)
				util.toggle_envelope(enable=1, skinCluster=1, blendShape=0)
				# record skinCluster
				skn_vtx_pos = util.get_vertices_positions(self.mesh)
				skn_vtx_pos_np = np.array(skn_vtx_pos).reshape(len(skn_vtx_pos), 3)
				self.orig_skn_verticles[bs_idx] = skn_vtx_pos_np
				#print(np.round(self.orig_skn_verticles[bs_idx], 2))
				#print(np.all(self.orig_skn_verticles[bs_idx] == skn_vtx_pos_np))

				# enable only blendShape
				util.toggle_envelope(enable=0, skinCluster=1, blendShape=0)
				util.toggle_envelope(enable=1, skinCluster=0, blendShape=1)
				# record blendShape
				bs_vtx_pos = util.get_vertices_positions(self.mesh)
				bs_vtx_pos_np = np.array(bs_vtx_pos).reshape(len(bs_vtx_pos), 3)
				self.orig_bs_verticles[bs_idx] = bs_vtx_pos_np
				
				# skinCluster delta
				self.orig_delta[bs_idx] = skn_vtx_pos_np - base_vtx_pos_np
				#print(self.orig_delta[bs_idx])
				#print(np.all(self.orig_delta[bs_idx] == vtx_pos_np - base_vtx_pos_np))

				cmds.setAttr('{}.{}'.format(self.exp, self.exp_attr_ls[idx]), .0)

				#if idx == 10:
					#break
			# -----------------------------------

			util.toggle_envelope(enable=1, skinCluster=1, blendShape=1)
			self.toggle_expression(connect=1)
			#[cmds.connectAttr(self.exp_con_ls[i], self.exp + '.' + self.exp_attr_ls[i]) for i in range(len(self.exp_attr_ls))]
			# -----------------------------------

			print('[bs w count, vtx count, pos xyz]', self.orig_skn_verticles.shape)
			np.save(self.orig_skn_vtx_fp, self.orig_skn_verticles)
			np.save(self.orig_bs_vtx_fp, self.orig_bs_verticles)
			np.save(self.orig_delta_fp, self.orig_delta)

		if not is_loaded:
			self.init_area_weight()
			self.check_blendshape_verticles_cache()

	def delete_blendshape_verticles_cache(self):
		cd_result = cmds.confirmDialog(title='Delete Caches', message='Are you sure to delete all caches',
									   button=['Proceed', 'Cancel'], icn='warning')
		if cd_result == 'Cancel':
			return None
		for i in [self.orig_skn_vtx_fp, self.orig_delta_fp, self.orig_bs_vtx_fp]:
			if os.path.exists(i):
				os.remove(i)
				print('\'{}\' was deleted'.format(os.path.basename(i)))
		self.orig_skn_verticles, self.orig_delta, self.orig_bs_verticles = None, None, None

	def get_normalized(self, x):  # each channels normalized
		x_norm = x
		for idx in range(x_norm.shape[0]):
			if not np.min(x[idx]) == np.max(x[idx]):
				x_norm[idx] = (x[idx] - np.min(x[idx])) / (np.max(x[idx]) - np.min(x[idx]))
			#print([idx, [np.min(x[idx]), np.max(x[idx])], [np.min(x_norm[idx]), np.max(x_norm[idx])]])
		return x_norm

	def check_blendshape_verticles_cache(self, threshold_percentile=0.9875):
		import random
		x = np.abs(self.orig_delta)
		x_norm = self.get_normalized(x)
		x_norm_ls = sorted(x_norm.flatten().round(5).tolist())
		threshold = x_norm_ls[int(round(len(x_norm_ls) * threshold_percentile))]
		for idx in range(x_norm.shape[0]):
			t = (x_norm[idx] >= threshold).tolist()
			selected_vtx = ['{}.vtx[{}]'.format(self.mesh, i) for i in range(len(t)) if True in t[i]]
			selected_vtx = random.sample(selected_vtx, int(float(len(selected_vtx)) // 6))
			if len(selected_vtx) == 0:
				continue
			else:
				cmds.select(selected_vtx)
				cmds.refresh()
				cmds.inViewMessage(amg='Select <hl>{}</hl>.'.format(self.w_name_ls[idx]), pos='topCenter', fade=1,
								   fit=0, fot=0, fst=50)
		cmds.select(cl=1)

	def rebuild_blendshape(self, weight_name):
		if not weight_name in self.w_name_ls:
			raise Warning('no {} in weight name list'.format(weight_name))
		grp_name = 'rebuild_mesh_grp'
		if not cmds.objExists(grp_name):
			cmds.group(n=grp_name, em=1)
			cmds.hide(grp_name)
		idx = self.w_name_ls.index(weight_name)
		mesh = cmds.sculptTarget(self.bs, e=1, target=idx, regenerate=1)
		if mesh:
			cmds.parent(mesh, grp_name)
			[cmds.setAttr('{}.{}'.format(mesh[0], i), lock=1) for i in ['t', 'r']]
			return mesh[0]

	def propagate_target_mesh(self, target_mesh='', blend_weight=1.0):
		'''
		- convert target to subracted as subt mesh
		- reset transform for subt mesh
		- get vtx subt mesh position
		- calculate delta vtx from current vtx pose
		- get normalized vtx data as delta weight
		- apply (delta vtx * delta weight) * blend weight to blendshape vtx data
		'''
		#print(list(self.target_vtx_pos_dict))
		#print(list(self.delta_vtx_pos_dict))

		def paint_weight_vtx_area_each_shape(area_vtx_w, mesh='', shade_num=10 ): # Test -------------------------
			if mesh != '':
				vtx_ls = cmds.ls(mesh + ".vtx[*]", flatten=1)
			else:
				vtx_ls = cmds.ls(self.mesh + ".vtx[*]", flatten=1)
			util.assign_vertex_color(vtx_ls, [.0] * 3); cmds.refresh()
			area_vtx_w_mean = np.mean(area_vtx_w, axis=2)
			vtx_grp_range = np.linspace(0.0, 1.0, shade_num) # grayscale layer (shade number)
			for idx in range(area_vtx_w_mean.shape[0]):
				vtx_v = area_vtx_w_mean[idx]
				zip_n_v = list(zip(vtx_ls, vtx_v.tolist()))
				if min(vtx_v) == max(vtx_v):
					cmds.inViewMessage(clear=1)
					continue
				else:
					cmds.inViewMessage(amg='<hl>{}</hl>.'.format(self.w_name_ls[idx]), pos='topCenter', fade=1,
									   fit=0, fot=0, fst=50, a=.0)
				for r in range(len(vtx_grp_range)):
					if vtx_grp_range[r] == max(vtx_grp_range):
						# print(vtx_grp_range[r])
						sel_vtx = [i for i in zip_n_v if i[1] == vtx_grp_range[r]]
					else:
						# print(vtx_grp_range[r], vtx_grp_range[r+1])
						sel_vtx = [i for i in zip_n_v if i[1] >= vtx_grp_range[r] and i[1] < vtx_grp_range[r + 1]]
					if not sel_vtx == []:
						util.assign_vertex_color([i[0] for i in sel_vtx], [vtx_grp_range[r]] * 3)
				cmds.refresh()
			util.delete_vertex_color(self.mesh)

		def paint_all_weight_vtx_area(sum_area_w, mesh=''): # Test2 -------------------------
			sum_area_w_mean = np.mean(sum_area_w, axis=1)
			if mesh != '':
				vtx_ls = cmds.ls(mesh + ".vtx[*]", flatten=1)
			else:
				vtx_ls = cmds.ls(self.mesh + ".vtx[*]", flatten=1)
			#util.assign_vertex_color(vtx_ls, [.0] * 3); cmds.refresh()
			vtx_grp_range = np.linspace(0.0, 1.0, 100)  # grayscale layer (shade number)
			#---
			zip_n_v = list(zip(vtx_ls, sum_area_w_mean.tolist()))
			for r in range(len(vtx_grp_range)):
				if vtx_grp_range[r] == max(vtx_grp_range):
					sel_vtx = [i for i in zip_n_v if i[1] == vtx_grp_range[r]]
				else:
					sel_vtx = [i for i in zip_n_v if i[1] >= vtx_grp_range[r] and i[1] < vtx_grp_range[r + 1]]
				if not sel_vtx == []:
					util.assign_vertex_color([i[0] for i in sel_vtx], [vtx_grp_range[r]] * 3)
			cmds.refresh()

		# cache
		if target_mesh in list(self.target_vtx_pos_dict):
			target_vtx_pos = np.copy( self.target_vtx_pos_dict[target_mesh] )
			base_vtx_pos = np.copy( self.target_vtx_pos_dict[self.mesh] )
			bs_weight_ls = self.bs_weight_dict[target_mesh]
			bs_area_w = np.copy(self.bs_area_w_dict[target_mesh])
			sum_area_w = np.copy(self.sum_area_w_dict[target_mesh])
			#print('{} already has a cache'.format(target_mesh))
		else:
			#print('record - {}'.format(target_mesh))
			# prepare subtrack skincluster
			orig_skn = [i for i in cmds.listHistory(self.mesh) if cmds.objectType(i) == 'skinCluster'][0]
			influence_ls = cmds.skinCluster(orig_skn, q=1, influence=1)
			tmp_mesh = cmds.duplicate(target_mesh, n='tmp_mesh')[0]
			for a in cmds.listAttr(tmp_mesh, k=1):
				orig_attr = '{}.{}'.format(self.mesh, a)
				tmp_attr = '{}.{}'.format(tmp_mesh, a)
				orig_value = cmds.getAttr(orig_attr)
				cmds.setAttr(tmp_attr, lock=0)
				cmds.setAttr(tmp_attr, orig_value)
			tmp_skn = cmds.skinCluster([tmp_mesh] + influence_ls)[0]
			cmds.copySkinWeights(ss=orig_skn, ds=tmp_skn, nm=1, sa='closestPoint', uv=['uv'] * 2,
								 ia='oneToOne', spa=0, sm=1, nr=1)
			cmds.setAttr(tmp_mesh + '.visibility', 0)

			# extract base mesh
			util.toggle_envelope(enable=0, skinCluster=1, blendShape=1)
			tmp_base_mesh = cmds.duplicate(self.mesh, n='tmp_base_mesh')[0]
			cmds.setAttr(tmp_base_mesh + '.visibility', 0)
			base_vtx_pos = np.array(util.get_vertices_positions(tmp_base_mesh))
			util.toggle_envelope(enable=1, skinCluster=1, blendShape=1)

			# extract subtract mesh
			self.toggle_expression(connect=0)
			subt_mesh = cmds.duplicate(tmp_mesh, n='subt_mesh')[0]
			cmds.setAttr(subt_mesh + '.visibility', 0)
			self.toggle_expression(connect=1)

			# base mesh to deltamush
			dtmsh_mesh = cmds.duplicate(tmp_base_mesh, n='dtmsh_mesh')[0]
			cmds.setAttr(dtmsh_mesh + '.visibility', 0)
			dtmsh_bs = cmds.blendShape(subt_mesh, dtmsh_mesh, foc=0, n='dtmsh_bs')[0]
			cmds.setAttr('{}.{}'.format(dtmsh_bs, subt_mesh), 1.0)
			target_vtx_pos = np.array(util.get_vertices_positions(dtmsh_mesh))

			#smooth tmp mesh
			smooth_tmp_mesh = cmds.duplicate(self.mesh, n='tmp_sm_mesh')[0]
			cmds.setAttr(smooth_tmp_mesh + '.visibility', 0)
			sm_bs = cmds.blendShape(target_mesh, smooth_tmp_mesh, foc=0, n='sm_bs')[0]
			cmds.setAttr('{}.{}'.format(sm_bs, target_mesh), 1.0)
			cmds.deltaMush(smooth_tmp_mesh, smoothingIterations=10, smoothingStep=1.0, iwc=1.0, owc=1.0)
			target_smooth_vtx_pos = np.array(util.get_vertices_positions(smooth_tmp_mesh))

			#clear temporary meshes
			cmds.delete([tmp_mesh, tmp_base_mesh, subt_mesh, dtmsh_mesh, smooth_tmp_mesh])

			# extract relate blendshape
			bs_weight_ls = [cmds.getAttr('{}.{}'.format(self.bs, i)) for i in self.w_name_ls]
			print('rebuilding target blendshape..')
			for bs_n, bs_w, idx in zip(self.w_name_ls, bs_weight_ls, range(len(bs_weight_ls))):
				if round(util.ease_in_out(bs_w), 3) == .0: continue
				if np.sum(np.abs(self.orig_delta[idx])) == .0: continue
				if cmds.objExists(bs_n): continue
				self.rebuild_blendshape(bs_n)
			cmds.refresh()

			# change area
			change_area_w = np.abs(np.copy(target_smooth_vtx_pos) - np.array(util.get_vertices_positions(self.mesh))) + 1.0
			change_area_w = (change_area_w - np.min(change_area_w)) / (np.max(change_area_w) - np.min(change_area_w))
			change_area_w = np.clip(change_area_w * 1.75, .0, 1.0)  # multiply clip
			#paint_all_weight_vtx_area(change_area_w, mesh=target_mesh)

			# all area * bs weight 100%
			bs_area_w = np.copy(self.area_vtx_w)
			for idx in range(self.orig_skn_verticles.shape[0]):
				if not cmds.objExists(self.w_name_ls[idx]):
					bs_area_w[idx] = bs_area_w[idx] * .0
				ease_in_weight = (util.ease_in(bs_weight_ls[idx] * .8) + (bs_weight_ls[idx] * .2))
				bs_area_w[idx] = bs_area_w[idx] * ease_in_weight * change_area_w

			# sum of all blendshape weight area
			sum_area_w = np.sum(bs_area_w, axis=0)
			sum_area_w = np.clip(sum_area_w, .0, 1.0)  # clamp 1.0
			sum_area_w = (util.ease_in(sum_area_w * .5) + (sum_area_w * .5))
			# paint_all_weight_vtx_area(sum_area_w)

			# save caches
			self.target_vtx_pos_dict[target_mesh] = np.copy(target_vtx_pos)
			self.target_vtx_pos_dict[self.mesh] = np.copy(base_vtx_pos)
			self.mse_dict[target_mesh] = []
			self.bs_weight_dict[target_mesh] = bs_weight_ls
			self.bs_area_w_dict[target_mesh] = np.copy(bs_area_w)
			self.sum_area_w_dict[target_mesh] = np.copy(sum_area_w)

		if self.new_verticles is None:
			delta_vtx_pos = target_vtx_pos - np.copy(self.orig_skn_verticles)
		else:
			delta_vtx_pos = target_vtx_pos - np.copy(self.new_verticles)
		#print('delta pos\n', delta_vtx_pos)
		#print(delta_vtx_pos.shape)
		if np.all(delta_vtx_pos == .0):
			cmds.warning('has no changed on {}'.format(target_mesh))
			return

		#current verticles
		if self.new_verticles is None:
			self.new_verticles = np.copy(self.orig_bs_verticles)
		for idx in range(self.new_verticles.shape[0]):
			#error
			if idx % (len(self.w_name_ls) // 1) == 0 or idx + 1 == self.new_verticles.shape[0]:
				er = np.mean(((self.new_verticles * sum_area_w) - (target_vtx_pos * sum_area_w)) ** 2)
				print('{} - error : {}'.format(target_mesh, round(er, 10)))
				self.mse_dict[target_mesh].append(er)
			#delta
			if not cmds.objExists(self.w_name_ls[idx]): continue
			if np.sum(bs_area_w[idx]) == .0: continue
			delta_multiply = delta_vtx_pos[idx] * bs_area_w[idx]
			self.new_verticles[idx] = self.new_verticles[idx] + ( delta_multiply * blend_weight )

		#print(self.new_verticles)
		#print(self.new_verticles.shape)

		if len(self.mse_dict[target_mesh]) > 2 and (self.mse_dict[target_mesh][-1] > self.mse_dict[target_mesh][-2]):
			raise Warning('an overfitting was occured on {}'.format(target_mesh))

		mse_ls = []
		for i in list(self.mse_dict):
			mse_ls += [ self.mse_dict[i][-1] ]
		print('mse : {}'.format(round(sum(mse_ls)/len(mse_ls), 16)))

		#cmds.pause(sec=5)
		#util.delete_vertex_color(self.mesh)
		#1/0

	def batch_transfer(self, target_data, epoch=1000):
		'''
		- target mesh list
		- anim pose frame list
		- iteration count
		- iteration weight = (targ et mesh count / count)
		target_data = [[10.0, 'head_lod0_mesh_bs_f10'], ...., [20.0, 'head_lod0_mesh_bs_f20']]
		'''
		def print_result():
			for obj in list(self.mse_dict):
				overfit_str = 'overfit!' if self.mse_dict[obj][-1] > self.mse_dict[obj][0] else ''
				print('object: {} - errors: [ first: {:05f}, last: {:05f} ]  {}'.format(
					obj, self.mse_dict[obj][0], self.mse_dict[obj][-1], overfit_str
				))

		self.reset_all_variables()
		import random, time
		orig_target_data = target_data
		random.shuffle(orig_target_data)
		multiply_rate = ((2.0/9.0) * float(len(target_data))) + (7.0/9.0)  # 3.0 at 10 targets
		multiply_rate = round(multiply_rate, 2)
		limit_rate = (float(len(target_data)) / float(epoch)) * multiply_rate
		limit_rate = round(limit_rate, 5)
		target_data = target_data * (epoch // len(target_data))
		random.shuffle(target_data)
		target_data = orig_target_data + target_data
		cmds.progressWindow(t='Preparing..', pr=0, st='', ii=0, min=0, max=len(target_data))
		rate = 1 / float(len(orig_target_data))
		epoch_idx = 1
		for frame, target_mesh in target_data:
			cmds.progressWindow(edit=1, s=1, status=('[ {} / {} ] - {}'.format(epoch_idx, len(target_data), target_mesh)))
			if not cmds.objExists(target_mesh) :continue
			print('\n_______ [ {} / {} ] _______ r : {:02f}  ( {:02f}x )'.format(epoch_idx, len(target_data), rate, multiply_rate))
			cmds.currentTime(frame) #force
			self.propagate_target_mesh(target_mesh=target_mesh, blend_weight=rate)
			epoch_idx += 1
			if epoch_idx + 1 <= len(orig_target_data):
				rate = 1 / float(len(orig_target_data))
			else:
				rate = limit_rate
		cmds.progressWindow(endProgress=1)
		print_result()

		# apply blendshape
		zip_bs_ls = sorted( [(np.sum(np.copy(self.area_vtx_w[i])), i, self.w_name_ls[i])
							 for i in range(len(self.w_name_ls))
							 if cmds.objExists(self.w_name_ls[i])], reverse=True )

		cmds.progressWindow(t='Sculpting..', pr=0, st='', ii=0, min=0, max=len(zip_bs_ls))
		progress_idx = 1.0
		t_st = time.time()
		vtx_shape = self.orig_bs_verticles.shape[1]
		vtx_idx_ls = [int(i) for i in range(vtx_shape)]
		diff = np.abs(self.new_verticles.copy() - self.orig_bs_verticles.copy())
		for sum_v, bs_idx, bs_n in zip_bs_ls:
			cmds.progressWindow(edit=1, s=1, status=('[ {:03d} ] - {}'.format(bs_idx, bs_n)))
			if not cmds.objExists(bs_n):
				cmds.warning('>> can\'t found {}. then skip <<'.format(bs_n))
				continue
			if np.all(diff[bs_idx] <= 0.001): continue
			progress_percent = round((progress_idx / float(len(zip_bs_ls))) * 100, 2)
			cmds.inViewMessage(amg='Sculpting blendshape : <hl>{} % - {}</hl>.'.format(progress_percent, bs_n),
							   pos='topCenter', fade=1, fit=0, fot=0, fst=50, a=.0, dk=0)
			vtx_sel_idx_ls = [i for i in vtx_idx_ls if np.all(self.area_vtx_w[bs_idx][i] != 0)]
			[cmds.move(self.new_verticles[bs_idx][i][0], self.new_verticles[bs_idx][i][1], self.new_verticles[bs_idx][i][2],
					   bs_n + '.vtx[{}]'.format(i), a=1, r=0, ls=1) for i in vtx_sel_idx_ls]
			print('{} % sculpting [ {:03d} ] - {}'.format(progress_percent, bs_idx, bs_n, len(vtx_idx_ls)))
			cmds.refresh()
			# ++++++++++++++
			progress_idx += 1.0
			if time.time() - t_st > 20.0:
				cmds.currentTime( target_data[ random.randint(0, len(target_data)-1) ][0] )
				t_st = time.time()
			#if bs_idx >= 10:
				#break
		cmds.progressWindow(endProgress=1)

		print_result()
		print('Finish')

	def reset_selected_vertices(self):
		sel_vtx = cmds.ls(self.mesh + '.vtx[*]', flatten=1, sl=1)
		expand_count = 15
		expand_weight = round((1.0/float(expand_count)), 3)
		orig_vtx_pos = np.copy(self.orig_bs_verticles)

		weight_ls = np.linspace(.0, 1, expand_count)
		weight_ls = (.0 + (1 - .0) * util.ease_in(weight_ls))
		weight_ls = ((weight_ls * .9) + (np.linspace(.0, 1, expand_count) * .1)).tolist()
		weight_ls = sorted(weight_ls, reverse=True) # 1 - .0 / in - out

		vtx_grp_ls, all_idx_ls = [], []
		cmds.select(sel_vtx)
		for i in range(expand_count):
			util.poly_select_traverse(mode=1)
			vtx_grp = [int(i.split('[')[-1].split(']')[0])
					   for i in cmds.ls(self.mesh + '.vtx[*]', flatten=1, sl=1)]
			vtx_grp = [i for i in vtx_grp if not i in all_idx_ls]
			all_idx_ls += vtx_grp
			vtx_grp_ls.append( vtx_grp )
			cmds.refresh(f=1)

		#fix area
		fix_area_w = np.zeros(self.orig_bs_verticles.shape[1:])
		for weight, idx_ls in sorted(zip(weight_ls, vtx_grp_ls), reverse=True):
			for i in idx_ls:
				fix_area_w[i] = np.array([weight] * 3)
		#print(fix_area_w)
		#print(fix_area_w.shape)

		#affected area
		all_area_w =  np.copy(self.area_vtx_w) * fix_area_w
		print(np.min(all_area_w), np.max(all_area_w))

		def paint_all_weight_vtx_area(sum_area_w): # Test2 -------------------------
			sum_area_w_mean = np.mean(sum_area_w, axis=1)
			vtx_ls = cmds.ls(self.mesh + ".vtx[*]", flatten=1)
			#util.assign_vertex_color(vtx_ls, [.0] * 3); cmds.refresh()
			vtx_grp_range = np.linspace(0.0, 1.0, 50)  # grayscale layer (shade number)
			#---
			zip_n_v = list(zip(vtx_ls, sum_area_w_mean.tolist()))
			for r in range(len(vtx_grp_range)):
				if vtx_grp_range[r] == max(vtx_grp_range):
					sel_vtx = [i for i in zip_n_v if i[1] == vtx_grp_range[r]]
				else:
					sel_vtx = [i for i in zip_n_v if i[1] >= vtx_grp_range[r] and i[1] < vtx_grp_range[r + 1]]
				if not sel_vtx == []:
					util.assign_vertex_color([i[0] for i in sel_vtx], [vtx_grp_range[r]] * 3)
			cmds.refresh()

		#print(np.sum(all_area_w, axis=0))
		#print(np.sum(all_area_w, axis=0).shape)
		#paint_all_weight_vtx_area(np.mean(self.fix_area_w, axis=0))
		#paint_all_weight_vtx_area(fix_area_w)
		#1/0

		sum_bs_w_ls = np.sum( np.sum(all_area_w, axis=1), axis=1 ).tolist()
		sum_bs_w_ls = [round(util.ease_in(i/max(sum_bs_w_ls)), 2) for i in sum_bs_w_ls]
		#print(min(sum_bs_w_ls), max(sum_bs_w_ls))

		bs_weight_ls = [cmds.getAttr('{}.{}'.format(self.bs, i)) for i in self.w_name_ls]
		bs_w_norm_ls = [util.ease_out(i/max(sum_bs_w_ls)) for i in bs_weight_ls]
		cmds.progressWindow(t='Sculpting..', pr=0, st='', ii=0, min=0, max=len(sum_bs_w_ls))
		for bs_w, idx, bs_n in sorted(zip(sum_bs_w_ls, range(orig_vtx_pos.shape[0]), self.w_name_ls), reverse=True):
			cmds.progressWindow(edit=1, s=1, status=('[ {:03d} ] - {}'.format(idx, bs_n)))
			if not cmds.objExists(self.w_name_ls[idx]): continue
			if sum_bs_w_ls[idx] == .0: continue
			if np.sum(all_area_w[idx]) == .0: continue #only record area
			vtx_ls = [self.w_name_ls[idx] + '.vtx[{}]'.format(i) for i in all_idx_ls]
			cur_vtx_pos = np.array( [cmds.pointPosition(vtx_ls[i], w=1) for i in range(len(all_idx_ls))] )
			weight_vtx = np.array([all_area_w[idx][i] for i in all_idx_ls])
			tg_vtx_pos = np.array([orig_vtx_pos[idx][i].tolist() for i in all_idx_ls])
			delta_vtx = (tg_vtx_pos - cur_vtx_pos) * weight_vtx * bs_w_norm_ls[idx]
			#print(np.max(delta_vtx) - np.min(delta_vtx))
			#print(np.max(delta_vtx) - np.min(delta_vtx))
			if np.max(delta_vtx) - np.min(delta_vtx) <= 0.005: continue
			#[cmds.xform(vtx_ls[i], r=1, os=1, t=delta_vtx[i]) for i in range(len(all_idx_ls))]
			[cmds.move(delta_vtx[i][0], delta_vtx[i][1], delta_vtx[i][2], vtx_ls[i], a=0, r=1, ls=1) for i in range(len(all_idx_ls))]
			print([idx, self.w_name_ls[idx], [np.min(delta_vtx), np.max(delta_vtx)], bs_w, bs_w_norm_ls[idx]])
			cmds.refresh()
			#if idx >= 150: break

		cmds.progressWindow(endProgress=1)
		cmds.refresh()
		cmds.pause(sec=1)
		cmds.select(sel_vtx)

if __name__ == '__main__':
	mesh = 'head_lod0_mesh'
	bs = 'head_lod0_mesh_blendShapes'
	bstf = blenshape_transfer(mesh, bs)
	target_data = [
		[1178.0, 'head_lod0_mesh_F1178'],
		[0159.0, 'head_lod0_mesh_F0159'],
		[0994.0, 'head_lod0_mesh_F0994'],
		[0540.0, 'head_lod0_mesh_F0540'],
	]
	#bstf.batch_transfer(target_data, epoch=500, rate=0.0075)

	#bstf.reset_selected_vertices()
