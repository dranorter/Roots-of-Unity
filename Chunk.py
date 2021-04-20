from godot import exposed, export
from godot.bindings import _File as File
from godot import *
import random as r
import numpy as np
import numbers, time, math
from debugging import debugging
import traceback

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
	# (1 6 2) (3 8 12) (4 9 10) (5 7 11) (13 14 15) = all ch3[0], diff. splits btw ch3[0] and ch3[14], all ch3[14].
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
	const_rot1 = [
		[0, 0, 0, 0, 0, 1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
		[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		
		[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		
		[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0],
		[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
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
		
	]
	
	def convert_chunklayouts(self,filename="res://chunklayouts_perf_14"):
		
		# TODO Identify all the blocks which are always present,
		# then exclude those when saving, so they don't have to be loaded.
		# Save them in a separate (and of course human-readable) file, or maybe
		# as a first element in the layouts file.
		# TODO Figure out all the rotation matrices mapping between chunk
		# orientations, so that I can save 1/10th as many of these templates.
		# Also, check if other symmetries apply - individual chunks are
		# rotationally symmetrical too.
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
		dupecounter = 0
		transformations = [(np.eye(6),np.eye(15)),
			(np.array(self.chunk_rot1),np.array(self.const_rot1)),
			(np.array(self.chunk_rot2),np.array(self.const_rot1).T)]
		
		for layout_file in [filename]:
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
					cstts = [list(a) for a in cstts]
					# Numbers of inside blocks and outside blocks
					inside_ct = int(str(fs.get_line()))
					outside_ct = int(str(fs.get_line()))
					# Then retrieve the strings representing the blocks
					is_blocks = []
					os_blocks = []
					for i in range(inside_ct):
						is_blocks.append(eval(str(fs.get_line())))
					for i in range(outside_ct):
						# Remove redundancy: inside blocks shouldn't also be outside
						# blocks. How does this even happen?
						osb = eval(str(fs.get_line()))
#						in_isb = False
#						for j in range(inside_ct):
#							if np.all(np.array(is_blocks[j]) - np.array(osb) == 0):
#								in_isb = True
#								continue
						os_blocks.append(osb)
					for m,n in transformations:
						ch_c_live = self.possible_centers_live[self.possible_centers.index(ch_c)]
						ch_c_live = m.dot(ch_c_live)
						t_ch_c = str(ch_c_live)
						t_cstts = n.dot(cstts)
						t_is_blocks = m.dot(np.array(is_blocks).T).tolist()
						t_os_blocks = m.dot(np.array(os_blocks).T).tolist()
						
						constraint_string = str((ch_c_live,t_cstts))
						if constraint_string not in all_sorted_constraints:
							all_constraints.append(t_cstts)
							constraints_sorted[t_ch_c].append(t_cstts)
							all_sorted_constraints.append(str((ch_c_live,t_cstts)))
							all_chunks.append(str((ch_c_live, t_is_blocks, t_os_blocks)))
							all_blocks.append((t_is_blocks,t_os_blocks))
							all_chosen_centers.append(t_ch_c)
							for block in t_is_blocks:
								all_block_axes.append(str(block - np.floor(block)))
						else:
							dupecounter += 1
							print("Found duplicate under rotation "+m)
					print("Loading chunklayouts..."+str(round(100*sum([len(x) for x in constraints_sorted.values()])/5000))+"%")
			except Exception as e:
				print("Encountered some sort of problem loading.")
				traceback.print_exc()
			fs.close()
			print("Duplicates due to symmetry: "+str(dupecounter))
		#print("Constraint counts for each possible chunk: "+str([len(x) for x in constraints_sorted.values()]))
		#print(time.perf_counter()-starttime)
		
		# Save file in a faster to load format
		fs.open("res://temp_test",fs.WRITE)
		fs.store_line(repr(all_constraints).replace('\n',''))
		fs.store_line(repr(constraints_sorted).replace('\n',''))
		fs.store_line(repr(all_blocks).replace('\n',''))
		fs.store_line(repr(all_chosen_centers).replace('\n',''))
		fs.close()
	
	def _ready(self):
		starttime = time.perf_counter()
		
		#self.convert_chunklayouts()
		
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
		
		print(time.perf_counter()-starttime)
		print("Loading...")
		print(time.perf_counter()-starttime)
		fs.open("res://chunk_layouts.repr",fs.READ)
		all_constraints = eval(str(fs.get_line()))
		print("Loaded part 1")
		print(time.perf_counter()-starttime)
		constraints_sorted = eval(str(fs.get_line()))
		print("Loaded part 2")
		print(time.perf_counter()-starttime)
		all_blocks = eval(str(fs.get_line()))
		print("Loaded part 3")
		print(time.perf_counter()-starttime)
		all_chosen_centers = eval(str(fs.get_line()))
		print("Loaded part 4")
		print(time.perf_counter()-starttime)
		fs.close()
		
		print("Done loading "+str(len(all_constraints))+" templates")
		print(time.perf_counter()-starttime)
		
		# Interesting note: Though all constraints are unique, they consist
		# of a very limited set, of just 11 numbers (6 after absolute value).
		# Well, with no rounding, it's 633 numbers...
		# Despite this, all but 100 of the 4980 constraints can be distinguished 
		# by which numbers are present, together with sign. (All can be 
		# distinguished if we leave out the rounding.)
		constraint_numsets = []
		numset_counts = []
		numset_members = []
		numset_ids = []
		numset_offsets = []
		constraint_numbers = set()
		for i in range(len(all_constraints)):
			match = False
			center = self.possible_centers_live[self.possible_centers.index(all_chosen_centers[i])]
			center_axes = 1-np.array(center - np.floor(center))*2
			center_origin = center - np.array(self.deflation_face_axes).T.dot(center_axes)/2
			center_axis1 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][0]])
			center_axis2 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][1]])
			center_axis3 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][2]])
			chunk_corners = np.array([center_origin,
				center_origin+center_axis1,center_origin+center_axis2,center_origin+center_axis3,
				center_origin+center_axis1+center_axis2,center_origin+center_axis1+center_axis3,center_origin+center_axis2+center_axis3,
				center_origin+center_axis1+center_axis2+center_axis3])
			# We consider all translations, since different chunks have the origin
			# at different corners.
			for corner in chunk_corners:
				shift = (-corner).dot(self.normallel.T)
				shifted_constraints = all_constraints[i] + np.repeat(shift.dot(self.twoface_normals.T).reshape(15,1),2,axis=1)
				#str_c = [str(pair) for pair in np.array(all_constraints[i])]
				str_c = np.abs(np.round(shifted_constraints,13)).flatten().tolist()
				for num in str_c: constraint_numbers.add(num)
				str_c.sort()
				str_c = str(str_c)
				if str_c not in set(constraint_numsets):
					constraint_numsets.append(str_c)
					numset_counts.append(1)
					numset_members.append([i])
					numset_offsets.append([corner])
					match = True
				else:
					numset_counts[constraint_numsets.index(str_c)] += 1
					numset_members[constraint_numsets.index(str_c)].append(i)
					numset_offsets[constraint_numsets.index(str_c)].append(corner)
					numset_ids.append(i)
					#print("Match with "+str(all_constraints[i]))
		print(str(len(constraint_numbers))+" constraint numbers.")
		print(str(len(constraint_numsets))+" constraint numsets.")
		print(str(len([x for x in numset_counts if x == 1]))+" lonely chunks.")
		
		first_sixer = numset_counts.index(18)
		
		for i in numset_members[first_sixer]:
			translation = numset_offsets[first_sixer][numset_members[first_sixer].index(i)]
			shift = (-translation).dot(self.normallel.T)
			shifted_constraints = all_constraints[i] + np.repeat(shift.dot(self.twoface_normals.T).reshape(15,1),2,axis=1)
			print(np.round(shifted_constraints,5))
			# Waste time so we can print
			#for j in range(5000000):
			#	_ = i + j
			#	_ = _ * _ * _ * _
			print(self.possible_centers_live[self.possible_centers.index(all_chosen_centers[i])] - translation)
			print("(originally "+all_chosen_centers[i]+")")
			print(translation)
			print(i)
		
		numset_counts.sort()
		print(numset_counts)
#[[0.1419951170682907, 0.22975291311740875], [-0.22975291311740875, -0.1419951170682907], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.08775780349969864, -2.0816681711721685e-16], [0.1419951170682907, 0.22975291311740875], [1.9081958235744878e-16, 0.08775780349969864], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.1419951170682907, -0.08775780349969864], [3.469446951953614e-16, 0.08775780349969864], [0.08775780349969864, 0.1419951170682907], [2.498001805406602e-16, 0.1419951170682907], [-0.1419951170682907, -3.400058012914542e-16], [1.8041124150158794e-16, 0.1419951170682907]]
#[[0.1419951170682907, 0.22975291311740875], [-0.22975291311740875, -0.1419951170682907], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [3.7470027081099033e-16, 0.08775780349969864], [0.1419951170682907, 0.22975291311740875], [-0.08775780349969864, -3.677613769070831e-16], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.1419951170682907, -0.08775780349969864], [-0.08775780349969864, -3.3306690738754696e-16], [0.08775780349969864, 0.1419951170682907], [3.608224830031759e-16, 0.1419951170682907], [-0.1419951170682907, -3.3306690738754696e-16], [1.3877787807814457e-16, 0.1419951170682907]]
#[[0.1419951170682907, 0.22975291311740875], [-0.22975291311740875, -0.1419951170682907], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [2.7755575615628914e-16, 0.08775780349969864], [0.1419951170682907, 0.22975291311740875], [1.6653345369377348e-16, 0.08775780349969864], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.1419951170682907, -0.08775780349969864], [-0.08775780349969864, -1.249000902703301e-16], [0.08775780349969864, 0.1419951170682907], [2.0816681711721685e-16, 0.1419951170682907], [-0.1419951170682907, -4.2327252813834093e-16], [4.0245584642661925e-16, 0.1419951170682907]]
#[[0.1419951170682907, 0.22975291311740875], [-0.22975291311740875, -0.1419951170682907], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.08775780349969864, -3.469446951953614e-16], [0.1419951170682907, 0.22975291311740875], [-0.08775780349969864, -1.942890293094024e-16], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.1419951170682907, -0.08775780349969864], [2.3592239273284576e-16, 0.08775780349969864], [0.08775780349969864, 0.1419951170682907], [3.0531133177191805e-16, 0.1419951170682907], [-0.1419951170682907, -3.885780586188048e-16], [2.498001805406602e-16, 0.1419951170682907]]
#[[0.1419951170682907, 0.22975291311740875], [-0.22975291311740875, -0.1419951170682907], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [2.498001805406602e-16, 0.08775780349969864], [0.1419951170682907, 0.22975291311740875], [-0.08775780349969864, -2.983724378680108e-16], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.1419951170682907, -0.08775780349969864], [4.996003610813204e-16, 0.08775780349969864], [0.08775780349969864, 0.1419951170682907], [3.95516952522712e-16, 0.1419951170682907], [-0.1419951170682907, -2.7755575615628914e-16], [3.469446951953614e-16, 0.1419951170682907]]
#[[0.1419951170682907, 0.22975291311740875], [-0.22975291311740875, -0.1419951170682907], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.08775780349969864, -1.8041124150158794e-16], [0.1419951170682907, 0.22975291311740875], [2.220446049250313e-16, 0.08775780349969864], [0.08775780349969864, 0.1419951170682907], [-0.1419951170682907, -0.08775780349969864], [-0.1419951170682907, -0.08775780349969864], [-0.08775780349969864, -3.608224830031759e-16], [0.08775780349969864, 0.1419951170682907], [3.191891195797325e-16, 0.1419951170682907], [-0.1419951170682907, -4.0245584642661925e-16], [3.400058012914542e-16, 0.1419951170682907]]
		
#		ordered_str = [[str(pair) for pair in np.round(np.array(all_constraints[u]))] for u in numset_ids]
#		for i in range(len(all_constraints)):
#
#			permutation = [ordered_str.index(x) for x in [str(pair) for pair in np.array(all_constraints[i])]]
#			print(permutation)
		
		# TODO Need to experiment more with faster loading methods for these.
		
#		possible_blocks = set()
#		for blocklayout in all_blocks:
#			combined = np.concatenate([blocklayout[0],blocklayout[1]])
#			combined = combined * 2
#			combined = np.array(np.round(combined),dtype=np.int64)
#			combined = [repr(list(x)) for x in combined]
#			for block in combined:
#				possible_blocks.add(block)
#		print("Set up possible blocks list. "+str(len(possible_blocks))+" occur.")#4042
#		print(time.perf_counter()-starttime)
#
#		possible_layouts = []
#		blocklist = [eval(x) for x in possible_blocks]
#		for blocklayout in all_blocks:
#			combined = np.concatenate([blocklayout[0],blocklayout[1]])
#			combined = np.round(combined * 2)
#			layout = np.any(np.all(np.repeat(blocklist,len(combined),axis=0).reshape(-1,len(combined),6) - combined == 0,axis=2),axis=1)
#			novel = True
#			for poss in possible_layouts:
#				if np.all(layout == poss):
#					novel = False
#					debugging.breakpoint()
#			if novel:
#				possible_layouts.append(layout)
#		print("Number of unique layouts according to more careful calculation:")
#		print(len(possible_layouts))
#		print(time.perf_counter()-starttime)
		
		# Choose one chunk to display
		chunk_num = r.choice(range(len(all_blocks)))
		#chunk_num = first_sixer
		chosen_center = self.possible_centers_live[self.possible_centers.index(all_chosen_centers[chunk_num])]
		inside_blocks = all_blocks[chunk_num][0]
		outside_blocks = all_blocks[chunk_num][1]
		
		print("Chosen center: "+str(chosen_center))
		
		multiplier = 1
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
		legit_superchunk = True
		while legit_superchunk:
			#seed = np.zeros(6)# Why was this line ever here?
			seed = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
			seed = seed.dot(self.squarallel.T)# Oddly, this line doesn't seem necessary.
			
			center_guarantee = dict()
			for center in self.possible_centers_live:
				center_axes = 1-np.array(center - np.floor(center))*2
				center_origin = center - np.array(self.deflation_face_axes).T.dot(center_axes)/2
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
			
			ch3_member = np.ceil(chosen_center)-np.floor(chosen_center)
			three_axes = np.nonzero(ch3_member)[0]
			constraint_dims = np.nonzero(1-np.any(self.twoface_axes - ch3_member > 0,axis=1))[0]
			# constraint_dims gives us indices into center_guarantee as well as twoface_axes,
			# twoface_normals and twoface_projected.
			for i in constraint_dims:
				third_axis = np.nonzero(ch3_member - self.twoface_axes[i])[0][0]
				axis_scale = np.eye(6)[third_axis].dot(self.normallel.T).dot(self.twoface_normals[i])
				divergence = center_guarantee[str(chosen_center)][i] - self.twoface_normals[i].dot(seed.dot(self.normallel.T))
				# Is point outside the constraints in this direction?
				if divergence[0]*divergence[1] >= 0:
					rand_pos = r.random()
					move = (divergence[0]*rand_pos + divergence[1]*(1-rand_pos))*np.eye(6)[third_axis].dot(self.normallel.T)/axis_scale
					seed = seed + move.dot(self.normallel)
					
					generates_correct_chunk = (np.all(self.twoface_normals.dot(seed.dot(self.normallel.T)) 
								> center_guarantee[str(chosen_center)][:,0] )
							and np.all(self.twoface_normals.dot(seed.dot(self.normallel.T)) < center_guarantee[str(chosen_center)][:,1]))
					if generates_correct_chunk:
						# Break early before we mess it up
						break
			
			# Now that "seed" is a valid offset for our chosen chunk, we need to 
			# determine which superchunk it can fit in.
			# There should logically be just one option, since the seed uniquely
			# determines the whole grid. We want to choose the chunk whose voxels
			# could all have fit into the original grid *as chunks* - meaning we
			# want to choose the chunk whose points are all within the tighter 
			# chunk-level constraints, IE, constraints nearer to the world-plane
			# by phi^3.
			# Does that mean we could just divide the constraints by phi^3? Well,
			# what the constraints accompanying a chunk mean is that if the seed
			# falls between those constraint values, then all the points listed will
			# be present within a grid with that seed. What it would mean to be present
			# "as chunks" depends on a chosen transformation between the block scale 
			# and the chunk scale -- the literal blocks aren't present "as chunks" since
			# they're at the wrong scale. We just mean that they would correspond 
			# to chunks within the (tighter) chunk constraint distance, under an
			# appropriate transformation.
			# Under such a transformation, then, each point would first of all be
			# brought closer to the worldplane by a ratio of phi^3, since, that's
			# the defining feature of chunks. It seems to me the constraints *do* 
			# simply shrink by phi^3 as well. I guess there's some question as to
			# whether perhaps a rotation is forced.
			# Two things worth noting: we have to move the offset (and indeed, the
			# entire worldplane) by the same transformation, and generally we need
			# to consider translations; because although I suppose there must be 
			# choices of offset which place the origin on the appropriate corner of
			# the new superchunk, that won't generally mean that it's also on the 
			# corner of the old chunk - meaning that a translated version of the old
			# seed is what needs to be tested against the new constraints, or vice
			# versa.
			
			chunk_axes = np.array(self.deflation_face_axes)[np.array(np.nonzero(1-ch3_member)[0])]
			#chunk_corners = 
			reproportioned_center = chosen_center.dot(self.squareworld)+math.pow(self.phi,6)*chosen_center.dot(self.squarallel)
			reproportioned_seed = seed*math.pow(self.phi,6)
			print("Reproportioned center: "+str(reproportioned_center))
			rescaled_center = reproportioned_center/(self.phi*self.phi*self.phi)#(2*reproportioned_center.dot(chunk_axes[0]))
			rescaled_seed = reproportioned_seed/(self.phi*self.phi*self.phi)#(2*reproportioned_center.dot(chunk_axes[0]))
			print("Rescaled center: "+str(rescaled_center))
			
			chunk_axes_inv = np.linalg.inv(np.array(self.deflation_face_axes).T)
			chunk_axes_inv_seed = chunk_axes_inv.dot(seed)
			print("Using chunk inverse: "+str(chunk_axes_inv.dot(chosen_center)))
			print("Rescaled seed: "+str(chunk_axes_inv_seed))
			print("Rescaled seed, made perpendicular: "+str(chunk_axes_inv_seed.dot(self.squarallel)))
			
			# Does this seed fall within our constraint region?
			print(str(np.any([np.all(self.twoface_normals.dot(chunk_axes_inv_seed.dot(self.normallel.T)) >= center_guarantee[chunk][:,0] )
						and np.all(self.twoface_normals.dot(chunk_axes_inv_seed.dot(self.normallel.T)) <= center_guarantee[chunk][:,1])
						for chunk in self.possible_centers])))
			positive_seed = chunk_axes_inv_seed + (chunk_axes_inv_seed < 0)
			print(str(np.any([np.all(self.twoface_normals.dot(positive_seed.dot(self.normallel.T)) >= center_guarantee[chunk][:,0] )
						and np.all(self.twoface_normals.dot(positive_seed.dot(self.normallel.T)) <= center_guarantee[chunk][:,1])
						for chunk in self.possible_centers])))
			
			# With the chunk and seed mapped down to block size, we can picture the old blocks present
			# as sub-blocks -- points which could be included in the grid if we widened the neighbor
			# requirements by a factor of phi cubed.
			
			# More importantly, there's a long-winded approach we can use to check a given chunk
			# for admissibility as a superchunk. We first choose a translation, which maps our
			# block (mapped down from a chunk) into an appropriately shaped block within the
			# chunk. Then we can reverse the scaling-down  map (so, use just the plain
			# deflation_face_axes matrix) on each point in the stored template, to get its
			# position as a potential chunk in the original grid. Then we test that point
			# under the more exacting constraints for chunk-hood. If all the points of a 
			# proposed chunk + translation pass this test with our chosen offset, we've
			# found the correct superchunk. (I suppose the corners of the chunk also have
			# to qualify as superchunks, falling within even more restricted ranges.)
			
			# So, if we were to take those restricted chunk (and superchunk) constraintns 
			# and map them backwards -- applying chunk_axes_inv -- then take their intersection,
			# would we get the saved chunk template constraints? TODO write an actual test.
			# But for now I think that's the case.
			
			# The upshot is, we take "chunk_axes_inv_seed" and consider all its possible
			# translations within a given chunk (which amount to new seed values). Then
			# we ask whether any of those seed values fall within the constraints for
			# the chunk. If so, this is our superchunk, and the corresponding translation
			# is our position in the superchunk. And we expect this proceduce to yield
			# a unique choice of superchunk and translation, except for the possibility
			# that we are a boundary chunk shared by several superchunks. (We also 
			# suspect there must be a faster procedure than checking all the translations.)
			
			# I realize all of that was basically repeated from the comment block 
			# before, but it's very late and writing it out helped.
			
			# OK, the procedure isn't identifying a single unique superchunk, but
			# I think the reason is that some blocks have their centers exactly on
			# the edges of chunks. I want to try and tweak my chunk template files
			# to fix this -- how do I do that? Systematically searching for examples
			# would consist of generating translated versions of the template 
			# constraints, corresponding to each block in the chunk; then checking for
			# overlap amongst all chunks, between constraints coming from blocks
			# of the same orientation and shape. Realistically I should store all
			# the translated constraints anyway, sorted by block shape, and have some
			# fast lookup method for identifying which (template, translation) pair
			# a given offset falls within. 
			# I should be able to introduce a rule for which chunk "owns" a block
			# within such a lookup method.
			
			chunk_as_block_center = np.round(chunk_axes_inv.dot(chosen_center)*2)/2.0
			
			closest_non_hit = None
			inside_hits = []
			outside_hits = []
			for i in range(len(all_blocks)):
				#TODO: Load all parent chunks?
				inside_blocks = np.array(all_blocks[i][0])
				constraint = np.array(all_constraints[i])
				# Blocks with orientation matching our chosen chunk
				aligned_blocks = inside_blocks[np.nonzero(np.all(inside_blocks - chunk_as_block_center
					- np.round(inside_blocks - chunk_as_block_center) == 0,axis=1))[0]]
				for block in (aligned_blocks - chunk_as_block_center):
					# Calculate translated seed
					translated_seed = (chunk_axes_inv_seed + block).dot(self.squarallel)
					# Does it fall in the constraints?
					positive_means_yes = min(np.min(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) - constraint[:,0]),
											np.min(-self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) + constraint[:,1]))
					# (initialize closest_non_hit)
					if closest_non_hit is None:
						closest_non_hit = (i, block, positive_means_yes)
					if (np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) >= constraint[:,0] )
									and np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) <= constraint[:,1])):
					#if positive_means_yes >= 0:
						print("Actually inside this: template#"+str(i)+", offset "+str(block))
						inside_hits.append((i, block))
					else:
						# Is it the closest non-hit so far?
						if positive_means_yes > closest_non_hit[2]:
							closest_non_hit = (i, block, positive_means_yes)
			# Checking outside blocks takes a lot longer, we only want to  do it if we need to
			if len(inside_hits) == 0:
				for i in range(len(all_blocks)):
					outside_blocks = np.array(all_blocks[i][1])
					constraint = np.array(all_constraints[i])
					aligned_blocks = outside_blocks[np.nonzero(np.all(outside_blocks - chunk_as_block_center
						- np.round(outside_blocks - chunk_as_block_center) == 0,axis=1))[0]]
					if len(list(aligned_blocks)) == 0:
						#print("Somehow no aligned blocks on template#"+str(i))
						continue
					block_offsets = aligned_blocks - chunk_as_block_center
					translated_seeds = (chunk_axes_inv_seed + block_offsets).dot(self.normallel.T)
					a_hit = np.all([np.all(self.twoface_normals.dot( translated_seeds.T ).T >= constraint[:,0],axis=1),
							np.all(self.twoface_normals.dot( translated_seeds.T ).T <= constraint[:,1],axis=1)],axis=0)
					if np.any(a_hit):
						print("Some sort of neighbor block hit! ")
						outside_hits = [(i,block_offsets[i]) for i in np.nonzero(a_hit)[0]]
						break
			if len(inside_hits) == 0 and len(outside_hits) == 0:
				# The slow way
				print("Trying exhaustive search")
				for i in range(len(all_blocks)):
					outside_blocks = np.array(all_blocks[i][1])
					constraint = np.array(all_constraints[i])
					aligned_blocks = outside_blocks[np.nonzero(np.all(outside_blocks - chunk_as_block_center
						- np.round(outside_blocks - chunk_as_block_center) == 0,axis=1))[0]]
					for block in (aligned_blocks - chunk_as_block_center):
						translated_seed = (chunk_axes_inv_seed + block).dot(self.squarallel)
						positive_means_yes = min(np.min(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) - constraint[:,0]),
												np.min(-self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) + constraint[:,1]))
						if (np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) >= constraint[:,0] )
										and np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) <= constraint[:,1])):
						#if positive_means_yes >= 0:
							print("Some corner inside this: template#"+str(i)+", offset "+str(block))
							outside_hits.append((i, block))
							block_offsets = aligned_blocks - chunk_as_block_center
							translated_seeds = (chunk_axes_inv_seed + block_offsets).dot(self.normallel.T)
							a_hit = np.all([np.all(self.twoface_normals.dot( translated_seeds.T ).T >= constraint[:,0],axis=1),
									np.all(self.twoface_normals.dot( translated_seeds.T ).T <= constraint[:,1],axis=1)],axis=0)
							print(a_hit)
							print(translated_seed.dot(self.normallel.T) - translated_seeds)
							print(block - block_offsets)
							print(np.all(self.twoface_normals.dot( translated_seeds.T ).T >= constraint[:,0],axis=0))
							print(self.twoface_normals.dot( translated_seeds.T ).T - constraint[:,0])
							print(np.all(self.twoface_normals.dot( translated_seeds.T ).T <= constraint[:,1],axis=0))
							print(self.twoface_normals.dot( translated_seeds.T ).T - constraint[:,1])
						else:
							# Is it the closest non-hit so far?
							if positive_means_yes > closest_non_hit[2]:
								closest_non_hit = (i, block, positive_means_yes)
			hits = inside_hits + outside_hits
			print("Found "+str(len(hits))+" possible superchunks.")
			print(time.perf_counter()-starttime)
			if len(hits) == 0:
				print("Using closest non-hit, which misses the constraint by:")
				print(closest_non_hit[2])
				hits = [(closest_non_hit[0],closest_non_hit[1])]
				legit_superchunk = False
		
		# Draw the valid chunk(s)
		
		for superchunk in hits:
			i, offset = superchunk
			multiplier = math.pow(self.phi,3)
			st = SurfaceTool()
			
			st.begin(Mesh.PRIMITIVE_LINES)
			st.add_color(Color(1,.2,1))
			for block in (np.array(all_blocks[i][1]) - offset):
				face_origin = np.floor(block).dot(self.worldplane.T)*multiplier
				face_tip = np.ceil(block).dot(self.worldplane.T)*multiplier
				dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)
								-np.floor(block))[0]].dot(self.worldplane.T)*multiplier
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
			self.mesh.surface_set_material(2,COLOR)

			self.mesh.surface_set_material(0,COLOR)
			st.begin(Mesh.PRIMITIVE_LINES)
			st.add_color(Color(0,0,0))
			for block in (np.array(all_blocks[i][0]) - offset):
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
			self.mesh.surface_set_material(3,COLOR)
		
		
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

