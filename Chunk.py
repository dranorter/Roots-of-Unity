from godot import exposed, export
from godot.bindings import _File as File
from godot import *
import random as r
import numpy as np
import numbers

COLOR = ResourceLoader.load("res://new_spatialmaterial.tres")

@exposed(tool=True)
class Chunk(MeshInstance):
	phi = 1.61803398874989484820458683
	worldplane = np.array([[phi,0,1,phi,0,-1],[1,phi,0,-1,phi,0],[0,1,phi,0,-1,phi]])
	normalworld = worldplane / np.linalg.norm(worldplane[0])
	squareworld = normalworld.transpose().dot(normalworld)
	parallelspace = np.array([[-1/phi,0,1,-1/phi,0,-1],
								  [1,-1/phi,0,-1,-1/phi,0],
								  [0,1,-1/phi,0,-1,-1/phi]])
	normallel = parallelspace / np.linalg.norm(parallelspace[0])
	squarallel = normallel.T.dot(normallel)
	deflation_face_axes = [ [ 2, 1, 1, 1, 1,-1],
						[ 1, 2, 1,-1, 1, 1],
						[ 1, 1, 2, 1,-1, 1],
						[ 1,-1, 1, 2,-1,-1],
						[ 1, 1,-1,-1, 2, -1],
						[-1, 1, 1,-1,-1, 2]]
	ch3 = [[1,1,1,0,0,0],[1,1,0,1,0,0],[1,1,0,0,1,0],[1,1,0,0,0,1],[1,0,1,1,0,0],[1,0,1,0,1,0],
						[1,0,1,0,0,1],[1,0,0,1,1,0],[1,0,0,1,0,1],[1,0,0,0,1,1],[0,1,1,1,0,0],[0,1,1,0,1,0],
						[0,1,1,0,0,1],[0,1,0,1,1,0],[0,1,0,1,0,1],[0,1,0,0,1,1],[0,0,1,1,1,0],[0,0,1,1,0,1],
						[0,0,1,0,1,1],[0,0,0,1,1,1]]
	twoface_axes = np.array([[1,1,0,0,0,0],[1,0,1,0,0,0],[1,0,0,1,0,0],[1,0,0,0,1,0],[1,0,0,0,0,1],
			[0,1,1,0,0,0],[0,1,0,1,0,0],[0,1,0,0,1,0],[0,1,0,0,0,1],[0,0,1,1,0,0],
			[0,0,1,0,1,0],[0,0,1,0,0,1],[0,0,0,1,1,0],[0,0,0,1,0,1],[0,0,0,0,1,1]])
	twoface_projected = np.array([
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 0]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 1]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 2]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 3]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 4]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 5]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 6]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 7]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 8]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 9]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 10]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 11]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 12]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 13]],
				normallel.T[np.nonzero(twoface_axes)[1][np.nonzero(twoface_axes)[0] == 14]]
			])
	twoface_normals = np.cross(twoface_projected[:,0],twoface_projected[:,1])
	twoface_normals = twoface_normals/np.linalg.norm(twoface_normals,axis=1)[0]
	possible_centers_live = np.array([[0.5, 0.5, 0.5,0.,0.,0.],[0.5,0.5,2., 1.,-1.5, 1.], [ 0.5,1.,1.5, 0., -0.5, 1.],
			[ 0.5,  1.5,  1.  ,-0.5,  0. ,  1. ], [ 0.5,  2. ,  0.5, -1.5,  1. ,  1. ], [ 0.5 , 2. ,  2. , -0.5, -0.5,  2. ], 
			[ 1. ,  0.5,  1.5 , 1. , -0.5,  0. ], [ 1. ,  1.5,  2. ,  0.5, -0.5,  1. ], [ 1.  , 1.5,  0.5, -0.5,  1.,   0. ], 
			[ 1. ,  2. ,  1.5 ,-0.5,  0.5,  1. ], [ 1.5,  0.5,  1. ,  1. ,  0. , -0.5], [ 1.5 , 1. ,  0.5,  0. ,  1.,  -0.5], 
			[ 1.5,  1. ,  2.  , 1. , -0.5,  0.5], [ 1.5,  2. ,  1. , -0.5,  1. ,  0.5], [ 2.  , 0.5,  0.5,  1. ,  1.,  -1.5], 
			[ 2. ,  0.5,  2.  , 2. , -0.5, -0.5], [ 2. ,  1. ,  1.5,  1. ,  0.5, -0.5], [ 2.  , 1.5,  1. ,  0.5,  1.,  -0.5], 
			[ 2. ,  2. ,  0.5 ,-0.5,  2. , -0.5], [2.,  2.,  2.,  0.5, 0.5, 0.5]])
	possible_centers = ['[0.5 0.5 0.5 0.  0.  0. ]', '[ 0.5  0.5  2.   1.  -1.5  1. ]', '[ 0.5  1.   1.5  0.  -0.5  1. ]', 
			'[ 0.5  1.5  1.  -0.5  0.   1. ]', '[ 0.5  2.   0.5 -1.5  1.   1. ]', '[ 0.5  2.   2.  -0.5 -0.5  2. ]', 
			'[ 1.   0.5  1.5  1.  -0.5  0. ]', '[ 1.   1.5  2.   0.5 -0.5  1. ]', '[ 1.   1.5  0.5 -0.5  1.   0. ]', 
			'[ 1.   2.   1.5 -0.5  0.5  1. ]', '[ 1.5  0.5  1.   1.   0.  -0.5]', '[ 1.5  1.   0.5  0.   1.  -0.5]', 
			'[ 1.5  1.   2.   1.  -0.5  0.5]', '[ 1.5  2.   1.  -0.5  1.   0.5]', '[ 2.   0.5  0.5  1.   1.  -1.5]', 
			'[ 2.   0.5  2.   2.  -0.5 -0.5]', '[ 2.   1.   1.5  1.   0.5 -0.5]', '[ 2.   1.5  1.   0.5  1.  -0.5]', 
			'[ 2.   2.   0.5 -0.5  2.  -0.5]', '[2.  2.  2.  0.5 0.5 0.5]']
	chunk_rot1 = [
		[0, 1, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 1],
		[0, 0, 0, 1, 0, 0]
	]
	chunk_rot2 = [
		[0, 0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 1],
		[0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 1, 0]
	]
	chunk_rot3 = [
		[1, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0],
		[0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 1]
	]
	chunk_rot3 = [
		[0, 1, 0, 0, 0, 0],
		[1, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 1, 0, 0]
	]
	chunk_rot3 = [
		[0, 0, 1, 0, 0, 0],
		[0, 1, 0, 0, 0, 0],
		[1, 0, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 1, 0]
	]
	
	def _ready(self):
		print("Loading from existing file...")
		fs = File()
		
		constraints_sorted = dict()
		for center in self.possible_centers:
			constraints_sorted[center] = []
		
		all_constraints = []
		all_blocks = []
		all_chunks = []
		all_counters = []
		all_chosen_centers = []
		all_block_axes = []
		all_sorted_constraints = []
		for layout_file in ["res://chunklayouts_noa","res://chunklayouts_perf_relevance"]:
			try:
				fs.open(layout_file,fs.READ)
				num_to_load = 100
				while not fs.eof_reached():#len(all_chunks) < num_to_load:#not fs.eof_reached():
					# relevant chunk as chosen_center string
					ch_c = str(fs.get_line())
					# Constraint is 30 floats
					cstts = np.zeros((30))
					for i in range(30):
						cstts[i] = fs.get_real()
					cstts = cstts.reshape((15,2))
					# Numbers of inside blocks and outside blocks
					inside_ct = int(str(fs.get_line()))
					outside_ct = int(str(fs.get_line()))
					# Then retrieve the strings representing the blocks
					is_blocks = []
					os_blocks = []
					for i in range(inside_ct):
						is_blocks.append(eval(str(fs.get_line())))
					for i in range(outside_ct):
						os_blocks.append(eval(str(fs.get_line())))
					
					ch_c_live = self.possible_centers_live[self.possible_centers.index(ch_c)]
					constraint_string = str((ch_c_live,cstts))
					if constraint_string not in all_sorted_constraints:
						all_constraints.append(cstts)
						constraints_sorted[ch_c].append(cstts)
						all_sorted_constraints.append(str((ch_c_live,cstts)))
						all_chunks.append(str((ch_c_live, is_blocks, os_blocks)))
						all_blocks.append((is_blocks,os_blocks))
						all_chosen_centers.append(ch_c)
						for block in is_blocks:
							all_block_axes.append(str(block - np.floor(block)))
					print(str([len(x) for x in constraints_sorted.values()]))
			except Exception as e:
				print("Encountered some sort of problem loading.")
				print(e)
			fs.close()
		print("Constraint counts for each possible chunk: "+str([len(x) for x in constraints_sorted.values()]))
		
		# Choose one chunk to display
		chunk_num = r.choice(range(len(all_chunks)))
		chosen_center = all_chosen_centers[chunk_num]
		inside_blocks = all_blocks[chunk_num][0]
		outside_blocks = all_blocks[chunk_num][1]
		
		multiplier = 4
		array_mesh = ArrayMesh()
		self.mesh = array_mesh
		
		st = SurfaceTool()
		print("Drawing neighbor blocks next.")
		st.begin(Mesh.PRIMITIVE_LINES)
		st.add_color(Color(0,.5,0))
		for block in outside_blocks:
			face_origin = np.floor(block).dot(self.worldplane.T)*multiplier
			face_tip = np.ceil(block).dot(self.worldplane.T)*multiplier
			dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(self.worldplane.T)*multiplier
			corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = (
				face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
				face_tip - dir1, face_tip - dir2, face_tip - dir3
			)
			dir1 = Vector3(dir1[0],dir1[1],dir1[2])
			dir2 = Vector3(dir2[0],dir2[1],dir2[2])
			dir3 = Vector3(dir3[0],dir3[1],dir3[2])
			# Draw by recombining
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
		st.commit(self.mesh)

		self.mesh.surface_set_material(0,COLOR)
		print("Drawing inner blocks. ")
		st.begin(Mesh.PRIMITIVE_LINES)
		st.add_color(Color(0,1,1))
		for block in inside_blocks:
			face_origin = np.floor(block).dot(self.worldplane.T)*multiplier
			face_tip = np.ceil(block).dot(self.worldplane.T)*multiplier
			dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(self.worldplane.T)*multiplier
			dir1 = Vector3(dir1[0],dir1[1],dir1[2])
			dir2 = Vector3(dir2[0],dir2[1],dir2[2])
			dir3 = Vector3(dir3[0],dir3[1],dir3[2])
			# Draw by recombining
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
			st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
			st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
		st.commit(self.mesh)
		self.mesh.surface_set_material(1,COLOR)
		
		# Now try to find a super-chunk for this chunk
		
		# First we have to find an offset value within the chunk constraints.
		# I'll call this the "seed" since it will determine the entire lattice.
		
		center_guarantee = dict()
		for center in self.possible_centers_live:
			center_axes = 1-np.array(center - np.floor(center))*2
			center_origin = center - np.array(self.deflation_face_axes).T.dot(center_axes)/2
			print("Origin (should be zeros): "+str(center_origin))
			center_axis1 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][0]])
			center_axis2 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][1]])
			center_axis3 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][2]])
			chunk_corners = np.array([center_origin,
				center_origin+center_axis1,center_origin+center_axis2,center_origin+center_axis3,
				center_origin+center_axis1+center_axis2,center_origin+center_axis1+center_axis3,center_origin+center_axis2+center_axis3,
				center_origin+center_axis1+center_axis2+center_axis3])
			a = np.sum(chunk_corners,axis=0)/8
			center_constraints = np.sum(np.stack(np.repeat([chunk_corners - a],30,axis=0),axis=1).dot(self.normallel.T)
					* np.concatenate(np.array([self.twoface_normals,-self.twoface_normals]).reshape((1,30,3))),axis=2)
			overall_center_constraints = 0.9732489894677302/(self.phi*self.phi*self.phi) - np.max(center_constraints,axis=0)
			translated_constraints = (overall_center_constraints*np.concatenate([-np.ones(15),np.ones(15)]) 
					+ np.concatenate([self.twoface_normals,self.twoface_normals]).dot(a.dot(self.normallel.T)))
			translated_constraints = (translated_constraints).reshape((2,15)).T
			center_guarantee[str(center)] = translated_constraints
		
		# For center [0.5, 0.5, 0.5, 0, 0, 0] it appears the relevant constraints are on 0, 1 and 5
		# which are [1,1,0,0,0,0], [1,0,1,0,0,0] and [0,1,1,0,0,0].
		# Conclusion: whichever dimensions are half-integer in the center, those are the three directions
		# which should be used as basis vectors to randomly generate a seed inside the constraints for the
		# chunk.
		
#		print(chosen_center)
#		chosen_seed = [0,0,0,0,0,0]
#		for axis1 in range(13):
#			for axis2 in range(axis1+1,14):
#				for axis3 in range(axis2+1,15):
#					print("Trying axes "+str(axis1)+", "+str(axis2)+", "+str(axis3)+".")
#					attempts = []
#					for i in range(20):
#						seed =  np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
#						seed = seed*2 - 1
#						seed = seed.dot(self.squarallel)
#						generates_correct_chunk = (np.all(self.twoface_normals.dot(seed.dot(self.normallel.T)) 
#									> center_guarantee[str(chosen_center)][:,0] )
#								and np.all(self.twoface_normals.dot(seed.dot(self.normallel.T)) 
#									< center_guarantee[str(chosen_center)][:,1]))
#						counter = 0
#						while not generates_correct_chunk and counter < 10000:
#							for axis in [axis1, axis2, axis3]:
#								counter += 1
#								divergence = center_guarantee[str(chosen_center)][axis] - self.twoface_normals[axis].dot(seed.dot(self.normallel.T))
#								# Is it outside the constraints in this direction?
#								if divergence[0]*divergence[1] >= 0:
#									rand_pos = r.random()
#									move = (divergence[0]*rand_pos + divergence[1]*(1-rand_pos))*self.twoface_normals[axis]
#									seed = seed + move.dot(self.normallel)
#
#									generates_correct_chunk = (np.all(self.twoface_normals.dot(seed.dot(self.normallel.T)) 
#												> center_guarantee[str(chosen_center)][:,0] )
#											and np.all(self.twoface_normals.dot(seed.dot(self.normallel.T)) 
#												< center_guarantee[str(chosen_center)][:,1]))
#									if generates_correct_chunk:
#										# Break early before we mess it up
#										break
#						attempts.append(counter)
#						chosen_seed = seed
#					print(attempts)

class GoldenField:
	phi = 1.61803398874989484820458683
	def __init__(self, values):
		self.ndarray = np.array(values, dtype=np.int16)
		if self.ndarray.shape[-1] != 2:
			raise Exception("Not a valid golden field array; last axis must be of size 2.")
	
	def __repr__(self):
		return f"{self.__class__.__name__}({list(self.ndarray)})"
	
	def __array__(self, dtype=None):
		return self.ndarray[...,0]+self.phi*self.ndarray[...,1]
	
	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		if method == '__call__':
			# Check if all integer
			all_integer = True
			for input in inputs:
				if not isinstance(input,Integral):
					if isinstance(input,np.ndarray):
						if not (input.dtype.kind in ['u','i']):
							all_integer = False
					elif isinstance(input, self.__class__):
						pass
					else:
						all_integer = False
			if not all_integer:
				# If we're not dealing with integers, there's no point in 
				# staying a GoldenField.
				return ufunc(np.array(self), *inputs, **kwargs)
			
			if ufunc == np.add:
				returnval = np.zeros(self.ndarray.shape)
				returnval = returnval + self.ndarray
				for input in inputs:
					if isinstance(input, self.__class__):
						returnval = returnval + input.ndarray
					else:
						# Just add to the integer part
						returnval[...,0] = returnval[...,0] + input
				return self.__class__(returnval)
			elif ufunc == np.multiply:
				returnval = self.ndarray.copy()
				for input in inputs:
					intpart = np.zeros(self.ndarray[...,0].shape)
					phipart = np.zeros(self.ndarray[...,0].shape)
					if isinstance(input, self.__class__):
						intpart = returnval[...,0] * input.ndarray[...,0]
						phipart = returnval[...,0] * input.ndarray[...,1] + returnval[...,1] * input.ndarray[...,0]
						intpart = intpart + returnval[...,1] * input.ndarray[...,1]
						phipart = phipart + returnval[...,1] * input.ndarray[...,1]
					elif isinstance(input, np.ndarray):
						# Multiply both parts by the array
						intpart = returnval[...,0] * input
						phipart = returnval[...,1] * input
					elif isinstance(input, numbers.Integral):
						intpart = returnval[...,0] * input
						phipart = returnval[...,1] * input
					else:
						return NotImplemented
					returnval[...,0] = intpart
					returnval[...,1] = phipart
				return self.__class__(returnval)
			else:
				return NotImplemented
		else:
			return NotImplemented

