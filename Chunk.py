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
	
	twoface_projected_w = np.array([
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
	twoface_normals_w = np.cross(twoface_projected_w[:,0],twoface_projected_w[:,1])
	twoface_normals_w = twoface_normals_w/np.linalg.norm(twoface_normals_w,axis=1)[0]
	
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
		transformations = [(np.eye(6),np.eye(15))]#,
		#	(np.array(self.chunk_rot1),np.array(self.const_rot1)),
		#	(np.array(self.chunk_rot2),np.array(self.const_rot1).T)]
		
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
						t_is_blocks = m.dot(np.array(is_blocks).T).T.tolist()
						t_os_blocks = m.dot(np.array(os_blocks).T).T.tolist()

						constraint_string = str((ch_c_live,t_cstts))
						if constraint_string not in all_sorted_constraints:
							all_constraints.append(t_cstts.tolist())
							constraints_sorted[t_ch_c].append(t_cstts.tolist())
							all_sorted_constraints.append(str((ch_c_live,t_cstts.tolist())))
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
			#print("Duplicates due to symmetry: "+str(dupecounter))
		print("Constraint counts for each possible chunk: "+str([len(x) for x in constraints_sorted.values()]))
		
		# Save file in a faster to load format
#		fs.open("res://temp_test",fs.WRITE)
#		fs.store_line(repr(all_constraints).replace('\n',''))
#		fs.store_line(repr(constraints_sorted).replace('\n',''))
#		fs.store_line(repr(all_blocks).replace('\n',''))
#		fs.store_line(repr(all_chosen_centers).replace('\n',''))
#		fs.close()
		
		possible_blocks = set()
		for blocklayout in all_blocks:
			combined = np.concatenate([blocklayout[0],blocklayout[1]])
			combined = combined * 2
			combined = np.array(np.round(combined),dtype=np.int64)/2
			combined = [repr(list(x)) for x in combined]
			for block in combined:
				possible_blocks.add(block)
		blocklist = [eval(x) for x in possible_blocks]
		
		inside_bool = np.zeros((len(all_blocks),len(blocklist)),dtype=np.bool)
		outside_bool = np.zeros((len(all_blocks),len(blocklist)),dtype=np.bool)
		for i in range(len(all_blocks)):
			inside_bool[i] = np.any(np.all(np.repeat(all_blocks[i][0],len(blocklist),axis=0).reshape(-1,len(blocklist),6)
					-np.array(blocklist)==0,axis=2).T,axis=1)
			outside_bool[i] = np.any(np.all(np.repeat(all_blocks[i][1],len(blocklist),axis=0).reshape(-1,len(blocklist),6)
					-np.array(blocklist)==0,axis=2).T,axis=1)
#			for j in range(len(blocklist)):
#				inside_bool[i,j] = np.any(np.all(np.array(all_blocks[i][0])-np.array(blocklist[j])==0,axis=1))
#				outside_bool[i,j] = np.any(np.all(np.array(all_blocks[i][1])-np.array(blocklist[j])==0,axis=1))
			print("Computing booleans..."+str(round(100*i/len(all_blocks)))+"%")
		fs.open("res://temp_test",fs.WRITE)
		fs.store_line(repr(list(all_constraints)).replace('\n',''))
		fs.store_line(repr(list(all_chosen_centers)).replace('\n',''))
		fs.store_line(repr(blocklist).replace('\n',''))
		fs.close()
		np.save("temp_test_is",inside_bool,allow_pickle=False)
		np.save("temp_test_os",outside_bool,allow_pickle=False)
	
	def draw_block_wireframe(self,block,st,multiplier):
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
	
	def draw_block(self,block,st,multiplier):
		face_origin = np.floor(block).dot(self.worldplane.T)*multiplier
		face_tip = np.ceil(block).dot(self.worldplane.T)*multiplier
		dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(self.worldplane.T)*multiplier
		# Make "right hand rule" apply
		if np.cross(dir1,dir2).dot(dir3) < 0:
			_ = dir1
			dir1 = dir2
			dir2 = _
		corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = (
			face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
			face_tip - dir1, face_tip - dir2, face_tip - dir3
		)
		dir1 = Vector3(dir1[0],dir1[1],dir1[2])
		dir2 = Vector3(dir2[0],dir2[1],dir2[2])
		dir3 = Vector3(dir3[0],dir3[1],dir3[2])
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1+dir2)
		
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1+dir2)
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
		
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2)
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2+dir3)
		
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir2+dir3)
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
		
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3)
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3+dir1)
		
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2]))
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir3+dir1)
		st.add_vertex(Vector3(face_origin[0],face_origin[1],face_origin[2])+dir1)
		
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1-dir2)
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
		
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1-dir2)
		
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2-dir3)
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2)
		
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir2-dir3)
		
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3-dir1)
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3)
		
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2]))
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir1)
		st.add_vertex(Vector3(face_tip[0],face_tip[1],face_tip[2])-dir3-dir1)
	
	def load_templates_npy(self):
		fs = File()
		fs.open("res://temp_test",fs.READ)
		self.all_constraints = eval(str(fs.get_line()))
		self.all_chosen_centers = eval(str(fs.get_line()))
		self.blocklist = np.array(eval(str(fs.get_line())))
		fs.close()
		inside_blocks_bools = np.load("temp_test_is.npy")
		outside_blocks_bools = np.load("temp_test_os.npy")
		for i in range(inside_blocks_bools.shape[0]):
			self.all_blocks.append((self.blocklist[inside_blocks_bools[i]],self.blocklist[outside_blocks_bools[i]]))
	
	def load_templates_repr(self):
		fs = File()
		fs.open("res://chunk_layouts.repr",fs.READ)
		self.all_constraints = eval(str(fs.get_line()))
		print("Loaded part 1")
		print(time.perf_counter()-starttime)
		self.constraints_sorted = eval(str(fs.get_line()))
		print("Loaded part 2")
		print(time.perf_counter()-starttime)
		self.all_blocks = eval(str(fs.get_line()))
		print("Loaded part 3")
		print(time.perf_counter()-starttime)
		self.all_chosen_centers = eval(str(fs.get_line()))
		print("Loaded part 4")
		print(time.perf_counter()-starttime)
		fs.close()
	
	def make_seed_within_constraints(self,constraints):
		upper_limit = 10000
		counter = 0
		while counter < upper_limit and not self.satisfies(self.seed,constraints,strict = True):
			counter += 1
			axes = list(range(15))
			r.shuffle(axes)
			for axis in axes:
				proj_seed = self.seed.dot(self.normallel.T).dot(self.twoface_normals[axis])
				if proj_seed <= np.array(constraints)[axis,0] or proj_seed >= np.array(constraints)[axis,1]:
					self.seed -= proj_seed*self.twoface_normals[axis].dot(self.normallel)
					new_dist = r.random()
					self.seed += ((1-new_dist)*np.array(constraints)[axis,0]*self.twoface_normals[axis].dot(self.normallel)
								+ new_dist*np.array(constraints)[axis,1]*self.twoface_normals[axis].dot(self.normallel))
				if self.satisfies(self.seed,constraints,strict=True):
					# Stop before we ruin it
					break
		if counter >= upper_limit:
			raise Exception("Exceeded allowed tries while trying to satisfy constraints "+str(constraints))
		
	def satisfies(self,vector,constraints,strict=False):
		"""
		Returns True if the vector (array or ndarray, 3D or 6D)	falls inside the 
		constraints. A 6D vector is first projected into the 3D 'parallel space'.
		Constraints should be of shape (15,2), representing first a lower and then
		an upper bound in the 15 directions perpendicular to the faces of a 
		triacontahedron, in the ordering represented by self.twoface_normals.
		If strict is set False, the vector may lie on the boundary; otherwise,
		vectors on the boundary result in a return value of False.
		"""
		threevector = np.zeros(3)
		if vector.shape[-1] == 6:
			threevector = vector.dot(self.normallel.T)
		if vector.shape[-1] == 3:
			threevector = vector
		if strict:
			return (np.all(self.twoface_normals.dot(threevector) 
						> np.array(constraints)[:,0] )
				and np.all(self.twoface_normals.dot(threevector)
						< np.array(constraints)[:,1]))
		return (np.all(self.twoface_normals.dot(threevector)
						>= np.array(constraints)[:,0] )
				and np.all(self.twoface_normals.dot(threevector)
						<= np.array(constraints)[:,1]))
	
	def satisfies_by(self,vector,constraints):
		"""
		Returns the margin by which the vector (array or ndarray, 3D or 6D)
		falls inside the constraints. A 6D vector is first projected into the 3D 
		'parallel space'. If vector is within the constraints, return value is 
		positive, and represents the distance the vector would have to be moved 
		to exit the constaints. If vector is outside the constraints, the 
		return value is negative, and represents the distance the vector would
		have to be moved to satisfy whicever constraint it is furthest from
		satisfying. This can be somewhat smaller than the distance it would
		need to move to actually satisfy the constraint.
		Constraints should be of shape (15,2), representing first a lower and then
		an upper bound in the 15 directions perpendicular to the faces of a 
		triacontahedron, in the ordering represented by self.twoface_normals.
		"""
		threevector = np.zeros(3)
		if vector.shape[-1] == 6:
			threevector = vector.dot(self.normallel.T)
		if vector.shape[-1] == 3:
			threevector = vector
		return min(np.min(self.twoface_normals.dot(threevector) - np.array(constraints)[:,0]),
					np.min(-self.twoface_normals.dot(threevector) + np.array(constraints)[:,1]))
	
	def generate_parents(self,template_index,offset):
		"""
		Takes a chunk template (as an index into self.all_blocks etc) and the
		offset at which the chunk's relative origin sits, and returns at least
		one valid superchunk for that chunk given the current seed. Return
		value is a list of tuples (index, offset) indicating each superchunk's
		template along with where its own origin will belong. Note, the returned
		offsets are scaled as if the child chunk were a single block, and 
		should be scaled up in order to be drawn etc.
		TODO: This should take a 'level' argument or require an appropriately 
		scaled 'seed' so that it can be used to generate other levels of the
		hierarchy.
		"""
		if np.linalg.norm(offset) > 0:
			raise NotImplementedError("Nonzero offset not implemented yet.")
		
		chosen_center = self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[template_index])]
		ch3_member = np.ceil(chosen_center)-np.floor(chosen_center)
		
		# (Keeping all the rambly figuring-things-out comments for now)
		
		# We want to choose the chunk whose voxels could all have fit into the
		# original grid *as chunks* - meaning we want to choose the chunk whose
		# points are all within the tighter chunk-level constraints, IE,
		# constraints nearer to the world-plane by phi^3.
		#
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
		reproportioned_center = chosen_center.dot(self.squareworld)+math.pow(self.phi,6)*chosen_center.dot(self.squarallel)
		reproportioned_seed = self.seed*math.pow(self.phi,6)
		print("Reproportioned center: "+str(reproportioned_center))
		rescaled_center = reproportioned_center/(self.phi*self.phi*self.phi)#(2*reproportioned_center.dot(chunk_axes[0]))
		rescaled_seed = reproportioned_seed/(self.phi*self.phi*self.phi)#(2*reproportioned_center.dot(chunk_axes[0]))
		print("Rescaled center: "+str(rescaled_center))
		
		chunk_axes_inv = np.linalg.inv(np.array(self.deflation_face_axes).T)
		#chunk_axes_inv = np.linalg.inv(np.array(self.deflation_face_axes))
		chunk_axes_inv_seed = chunk_axes_inv.dot(self.seed)
		print("Using chunk inverse: "+str(chunk_axes_inv.dot(chosen_center)))
		print("Rescaled seed: "+str(chunk_axes_inv_seed))
		
		# Does this seed fall within our constraint region?
		# Seeds are equivalent under integer addition and subtration, but
		# I'm not sure what shape our constraint space takes, particularly once
		# we include all the translations - IE, for each chunk layout, we slide
		# the constraints over in accordance with each valid block, when checking
		# whether the layout is a valid superchunk. It does seem that the translations
		# don't overlap with one another, and I expect that they share edges and fill 
		# some space, but I don't know what shape it turns out to be for sure.
		# If everything is working fine, it has volume 1 and would capture any
		# possible seed if it were translated by all the 6D basis vectors; but 
		# I need to restrict the seeds to fall inside it, meaning I need to 
		# preemptively perform the translation. Right now, seeds are initialized
		# to a random position inside the chosen chunk orientation's constraints, 
		# so "chunk_axes_inv_seed" should fall within the constraints for a
		# correspondingly-shaped block existing at the 3D origin. I guess the 
		# question is, would all template constraints compatible with that block's 
		# presence overlap it (falling within it exactly for seed values where the 
		# block is actually present)? When the block is literally present in them,
		# the answer is yes since those constraints are being calculated the same
		# way.
		# I guess I'm thinking of this a bit wrong - adding, say, (1,1,1,0,0,0) to
		# the seed doesn't affect the appearance of the grid, but it affects the
		# 6D coordinates of every block present. Constraints for a specific block's 
		# presence or absence are absolute, rather than determined up to translation.
		# The "rescaling" transformation above, combined with all the translations
		# of the seed occurring in the search below, check the seed absolutely
		# for presence of the specific blocks in the templates. 
		print(str(np.any([np.all(self.twoface_normals.dot(chunk_axes_inv_seed.dot(self.normallel.T)) 
								>= self.center_guarantee[chunk][:,0] )
					and np.all(self.twoface_normals.dot(chunk_axes_inv_seed.dot(self.normallel.T)) 
								<= self.center_guarantee[chunk][:,1])
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
		for i in range(len(self.all_blocks)):
			inside_blocks = np.array(self.all_blocks[i][0])
			constraint = np.array(self.all_constraints[i])
			aligned_blocks = inside_blocks[np.nonzero(np.all(inside_blocks - chunk_as_block_center
				- np.round(inside_blocks - chunk_as_block_center) == 0,axis=1))[0]]
			if len(list(aligned_blocks)) == 0:
				continue
			block_offsets = aligned_blocks - chunk_as_block_center
			translated_seeds = (chunk_axes_inv_seed + block_offsets).dot(self.normallel.T)
			#translated_seeds = (chunk_axes_inv_seed - block_offsets).dot(self.normallel.T)
			a_hit = np.all([np.all(self.twoface_normals.dot( translated_seeds.T ).T >= constraint[:,0],axis=1),
					np.all(self.twoface_normals.dot( translated_seeds.T ).T <= constraint[:,1],axis=1)],axis=0)
			if np.any(a_hit):
				print("We're inside template #"+str(i)+", at offset"+str(block_offsets[np.array(np.nonzero(a_hit))[0,0]]))
				inside_hits = [(i,block_offsets[j]) for j in np.nonzero(a_hit)[0]]
				break
#		if len(inside_hits) == 0:
#			print("Trying for-loop search")
#			for i in range(len(self.all_blocks)):
#				#TODO: Load all parent chunks?
#				inside_blocks = np.array(self.all_blocks[i][0])
#				constraint = np.array(self.all_constraints[i])
#				# Blocks with orientation matching our chosen chunk
#				aligned_blocks = inside_blocks[np.nonzero(np.all(inside_blocks - chunk_as_block_center
#					- np.round(inside_blocks - chunk_as_block_center) == 0,axis=1))[0]]
#				for block in (aligned_blocks - chunk_as_block_center):
#					# Calculate translated seed
#					translated_seed = (chunk_axes_inv_seed + block).dot(self.squarallel)
#					#translated_seed = (chunk_axes_inv_seed - block).dot(self.squarallel)
#					# Does it fall in the constraints?
#					# TODO This doesn't seem to work (compared with the explicit <= and >=) but I don't see why.
#					positive_means_yes = min(np.min(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) - constraint[:,0]),
#											np.min(-self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) + constraint[:,1]))
#					# (initialize closest_non_hit)
#					if closest_non_hit is None:
#						closest_non_hit = (i, block, positive_means_yes)
#					if (np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) >= constraint[:,0] )
#									and np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) <= constraint[:,1])):
#					#if positive_means_yes >= 0:
#						print("Actually inside this: template#"+str(i)+", offset "+str(block))
#						inside_hits.append((i, block))
#						if positive_means_yes < 0:
#							print("positive_means_yes evaluated to "+str(positive_means_yes)+" (rather than being >= 0)")
#					else:
#						if positive_means_yes >= 0:
#							print("positive_means_yes evaluated to "+str(positive_means_yes)+" (rather than being < 0)")
#						# Is it the closest non-hit so far?
#						if positive_means_yes > closest_non_hit[2]:
#							closest_non_hit = (i, block, positive_means_yes)
		# Checking outside blocks takes a lot longer, we only want to  do it if we need to
		# TODO This faster numpy version doesn't seem to agree with the slower code - what's 
		# the difference?
		if len(inside_hits) == 0:
			for i in range(len(self.all_blocks)):
				outside_blocks = np.array(self.all_blocks[i][1])
				constraint = np.array(self.all_constraints[i])
				aligned_blocks = outside_blocks[np.nonzero(np.all(outside_blocks - chunk_as_block_center
					- np.round(outside_blocks - chunk_as_block_center) == 0,axis=1))[0]]
				if len(list(aligned_blocks)) == 0:
					#print("Somehow no aligned blocks on template#"+str(i))
					continue
				block_offsets = aligned_blocks - chunk_as_block_center
				translated_seeds = (chunk_axes_inv_seed + block_offsets).dot(self.normallel.T)
				#translated_seeds = (chunk_axes_inv_seed - block_offsets).dot(self.normallel.T)
				a_hit = np.all([np.all(self.twoface_normals.dot( translated_seeds.T ).T >= constraint[:,0],axis=1),
						np.all(self.twoface_normals.dot( translated_seeds.T ).T <= constraint[:,1],axis=1)],axis=0)
				if np.any(a_hit):
					print("Some sort of neighbor block hit! ")
					outside_hits = [(i,block_offsets[j]) for j in np.nonzero(a_hit)[0]]
					break
#		if len(inside_hits) == 0 and len(outside_hits) == 0:
#			# The slow way
#			print("Trying exhaustive search")
#			for i in range(len(self.all_blocks)):
#				outside_blocks = np.array(self.all_blocks[i][1])
#				constraint = np.array(self.all_constraints[i])
#				aligned_blocks = outside_blocks[np.nonzero(np.all(outside_blocks - chunk_as_block_center
#					- np.round(outside_blocks - chunk_as_block_center) == 0,axis=1))[0]]
#				for block in (aligned_blocks - chunk_as_block_center):
#					translated_seed = (chunk_axes_inv_seed + block).dot(self.squarallel)
#					#translated_seed = (chunk_axes_inv_seed - block).dot(self.squarallel)
#					positive_means_yes = min(np.min(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) - constraint[:,0]),
#											np.min(-self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) + constraint[:,1]))
#					if (np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) >= constraint[:,0] )
#									and np.all(self.twoface_normals.dot(translated_seed.dot(self.normallel.T)) <= constraint[:,1])):
#					#if positive_means_yes >= 0:
#						print("Some corner inside this: template#"+str(i)+", offset "+str(block))
#						outside_hits.append((i, block))
#						block_offsets = aligned_blocks - chunk_as_block_center
#						translated_seeds = (chunk_axes_inv_seed + block_offsets).dot(self.normallel.T)
#						a_hit = np.all([np.all(self.twoface_normals.dot( translated_seeds.T ).T >= constraint[:,0],axis=1),
#								np.all(self.twoface_normals.dot( translated_seeds.T ).T <= constraint[:,1],axis=1)],axis=0)
#						print(a_hit)
#						print(translated_seed.dot(self.normallel.T) - translated_seeds)
#						print(block - block_offsets)
#						print(np.all(self.twoface_normals.dot( translated_seeds.T ).T >= constraint[:,0],axis=0))
#						print(self.twoface_normals.dot( translated_seeds.T ).T - constraint[:,0])
#						print(np.all(self.twoface_normals.dot( translated_seeds.T ).T <= constraint[:,1],axis=0))
#						print(self.twoface_normals.dot( translated_seeds.T ).T - constraint[:,1])
#					else:
#						# Is it the closest non-hit so far?
#						if positive_means_yes > closest_non_hit[2]:
#							closest_non_hit = (i, block, positive_means_yes)
		hits = inside_hits + outside_hits
		print("Found "+str(len(hits))+" possible superchunks.")
		if len(hits) == 0:
			print("(WARNING) Using closest non-hit, which misses the constraint by:")
			print(closest_non_hit[2])
			hits = [(closest_non_hit[0],closest_non_hit[1])]
		
		return hits
	
	def generate_children(self,i,offset):
		"""
		
		"""
		
		chosen_center = self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[i])]
		chunk_axes_inv = np.linalg.inv(np.array(self.deflation_face_axes).T)
		chunk_as_block_center = np.round(chunk_axes_inv.dot(chosen_center)*2)/2.0
		
		children = []
		
		multiplier = 1
		for chunk in np.concatenate([np.array(self.all_blocks[i][0]) - offset, np.array(self.all_blocks[i][1]) - offset]):
			if np.all(chunk == chunk_as_block_center):
				print("Looking at the original chunk... "+str(chunk))
			st = SurfaceTool()
			st.begin(Mesh.PRIMITIVE_TRIANGLES)
			st.add_color(Color(r.random(),r.random(),r.random()))
			# We need to use "offset" plus the current chunk's coordinates to move "seed", 
			# and combine that with the orientation of the current chunk to
			# determine which block template to use. The "offset" is 
			# represented as if chunks were block-sized though, and needs to
			# be put through the transformation.
			# The order in which this is done seems pretty influential for 
			# amount of floating point error.
			chunk_center_pos = chunk - np.floor(chunk)
			chunkscaled_position = (np.array(self.deflation_face_axes).T).dot(np.floor(chunk))
			#chunk_seed = (seed + chunkscaled_position)#.dot(self.squarallel)
			chunk_seed = (self.seed - chunkscaled_position).dot(self.squarallel)
#				if np.all(chunk == chunk_as_block_center):
#					print("Scaled pos was "+str(np.round(chunkscaled_position,3)))
#					print("Seed comparison:")
#					print(self.seed)
#					print(chunk_seed)
#					print()
			matches = 0
			for template_index in range(len(self.all_blocks)):
				# Check alignment
				# TODO Checking alignment shouldn't be necessary - any 'extra'
				# chunks found would still be rooted at the correct position and
				# would still belong on the tesselation.
				template_center = self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[template_index])]
				if np.all((template_center + chunk) 
						- np.round(template_center + chunk) != 0):
					# Then check constraint
					constraint = np.array(self.all_constraints[template_index])
#					if np.all(chunk == chunk_as_block_center):
#						if chunk_num == template_index:
#							print("Found original template.")
#						if (np.all(self.twoface_normals.dot(chunk_seed.dot(self.normallel.T)) >= constraint[:,0] )
#									and np.all(self.twoface_normals.dot(chunk_seed.dot(self.normallel.T)) <= constraint[:,1])):
					if self.satisfies(chunk_seed, constraint):
						# Then draw! We'll expect a unique match but not check.
						matches = matches + 1
						for block in (np.concatenate([np.array(self.all_blocks[template_index][0]),
									np.array(self.all_blocks[template_index][1])]) + chunkscaled_position):
							children.append(block)
							if ( block[0] + block[1] + block[2] == 5):# and block[5] in list(np.arange(20)-10.5)):#block.dot(self.normalworld.T).dot(self.normalworld.T[0]) < 5
								#and block[0] == 0):
								self.draw_block(block,st,multiplier)
#						if chunk_num == template_index:
#							print("And our seed satisfies the original constraints!")
#					else:
#						if np.all(chunk==chunk_as_block_center) and chunk_num == template_index:
#							print("But seed isn't inside its constraints.")
			if matches > 0:
				st.generate_normals()
				st.commit(self.mesh)
				self.mesh.surface_set_material(self.mesh.get_surface_count()-1,COLOR)
			if matches > 1:
				print("More than one templates matched a chunk... found "+str(matches)+".")
			if matches == 0:
				print("No templates matched the chunk "+str(chunk))
#					print("Scaled pos was "+str(np.round(chunkscaled_position,3)))
		return children
	
	def _ready(self):
		starttime = time.perf_counter()
		
		#self.convert_chunklayouts()
		
		print("Loading from existing file...")
		
		self.all_constraints = []
		self.all_blocks = []
		self.blocklist = []
		
		print(time.perf_counter()-starttime)
		print("Loading...")
		
		self.load_templates_npy()
		print("Done loading "+str(len(self.all_constraints))+" templates")
		print(time.perf_counter()-starttime)
		
		
		# Choose one chunk to display
		chunk_num = r.choice(range(len(self.all_blocks)))
		chosen_center = self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[chunk_num])]
		inside_blocks = self.all_blocks[chunk_num][0]
		outside_blocks = self.all_blocks[chunk_num][1]
		
		print("Chosen center: "+str(chosen_center))
		
		array_mesh = ArrayMesh()
		self.mesh = array_mesh
		
		
		# Now try to find a super-chunk for this chunk
		
		# First we have to find an offset value within the chunk constraints.
		# I'll call this the "seed" since it will determine the entire lattice.
		
		# Starting value, will get moved to inside the chosen chunk orientation constraints.
		self.seed = np.array([-r.random()*0.1,-r.random()*0.1,-r.random()*0.1,-r.random()*0.1,-r.random()*0.1,-r.random()*0.1])
		self.seed = self.seed.dot(self.squarallel)
		
		self.center_guarantee = dict()
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
			self.center_guarantee[str(center)] = translated_constraints
		
		ch3_member = np.ceil(chosen_center)-np.floor(chosen_center)
		three_axes = np.nonzero(ch3_member)[0]
		constraint_dims = np.nonzero(1-np.any(self.twoface_axes - ch3_member > 0,axis=1))[0]
		# constraint_dims gives us indices into center_guarantee as well as twoface_axes,
		# twoface_normals and twoface_projected.
		for i in constraint_dims:
			third_axis = np.nonzero(ch3_member - self.twoface_axes[i])[0][0]
			axis_scale = np.eye(6)[third_axis].dot(self.normallel.T).dot(self.twoface_normals[i])
			divergence = self.center_guarantee[str(chosen_center)][i] - self.twoface_normals[i].dot(self.seed.dot(self.normallel.T))
			# Is point outside the constraints in this direction?
			if divergence[0]*divergence[1] >= 0:
				rand_pos = r.random()
				move = (divergence[0]*rand_pos + divergence[1]*(1-rand_pos))*np.eye(6)[third_axis].dot(self.normallel.T)/axis_scale
				self.seed = self.seed + move.dot(self.normallel)
				
				generates_correct_chunk = (np.all(self.twoface_normals.dot(self.seed.dot(self.normallel.T)) 
							> self.center_guarantee[str(chosen_center)][:,0] )
						and np.all(self.twoface_normals.dot(self.seed.dot(self.normallel.T)) 
								< self.center_guarantee[str(chosen_center)][:,1]))
				if generates_correct_chunk:
					# Break early before we mess it up
					break
		self.make_seed_within_constraints(self.all_constraints[chunk_num])
		print("Chose a seed within constraints for template #"+str(chunk_num)+":")
		print(self.seed)
		
		# Now that "seed" is a valid offset for our chosen chunk, we need to 
		# determine which superchunk it can fit in.
		# There should logically be just one option, since the seed uniquely
		# determines the whole grid.
		# TODO: (Runs so far have always found either 1 or 2 valid superchunks.
		# But, should do a more rigorous test, ideally verifying back in 
		# numpylattice.py that this is inevitable.)
		
		hits = self.generate_parents(chunk_num,[0,0,0,0,0,0])
		
		# Draw the valid chunk(s)
		for superchunk in hits:
			i, offset = superchunk
			multiplier = math.pow(self.phi,3)
			st = SurfaceTool()
			
			st.begin(Mesh.PRIMITIVE_LINES)
			st.add_color(Color(1,.2,1))
			for block in (np.array(self.all_blocks[i][1]) - offset):
				self.draw_block_wireframe(block,st,multiplier)
			st.commit(self.mesh)
			self.mesh.surface_set_material(self.mesh.get_surface_count()-1,COLOR)
			
			st.begin(Mesh.PRIMITIVE_LINES)
			st.add_color(Color(0.5,0,1))
			for block in (np.array(self.all_blocks[i][0]) - offset):
				self.draw_block_wireframe(block,st,multiplier)
			st.commit(self.mesh)
			self.mesh.surface_set_material(self.mesh.get_surface_count()-1,COLOR)
		
		# Now we draw the blocks inside those chunks.
#		sanity_test = (np.array(self.deflation_face_axes).T).dot(chunk_axes_inv_seed)
#		print("Sanity test:")
#		print(sanity_test)
#		print(self.seed)
		# For now we'll draw strictly the "interior" blocks, but
		# include the "exterior" chunks.
		children = []
		for i, offset in hits:
			children += self.generate_children(i, offset)
		
	
	
	
	
	
	def test_templates(self):
		possible_blocks = set()
		for blocklayout in self.all_blocks:
			combined = np.concatenate([blocklayout[0],blocklayout[1]])
			combined = combined * 2
			combined = np.array(np.round(combined),dtype=np.int64)
			combined = [repr(list(x)) for x in combined]
			for block in combined:
				possible_blocks.add(block)
		print("Set up possible blocks list. "+str(len(possible_blocks))+" occur.")#4042
		print(time.perf_counter()-starttime)

		possible_layouts = []
		blocklist = [eval(x) for x in possible_blocks]
		for blocklayout in self.all_blocks:
			combined = np.concatenate([blocklayout[0],blocklayout[1]])
			combined = np.round(combined * 2)
			layout = np.any(np.all(np.repeat(blocklist,len(combined),axis=0).reshape(-1,len(combined),6) - combined == 0,axis=2),axis=1)
			novel = True
			for poss in possible_layouts:
				if np.all(layout == poss):
					novel = False
					debugging.breakpoint()
			if novel:
				possible_layouts.append(layout)
		print("Number of unique layouts according to more careful calculation:")
		print(len(possible_layouts))
		print(time.perf_counter()-starttime)
	
	def symmetries_search(self):
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
		for i in range(len(self.all_constraints)):
			match = False
			center = self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[i])]
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
				shifted_constraints = self.all_constraints[i] + np.repeat(shift.dot(self.twoface_normals.T).reshape(15,1),2,axis=1)
				#str_c = [str(pair) for pair in np.array(self.all_constraints[i])]
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
					#print("Match with "+str(self.all_constraints[i]))
		print(str(len(constraint_numbers))+" constraint numbers.")
		print(str(len(constraint_numsets))+" constraint numsets.")
		print(str(len([x for x in numset_counts if x == 1]))+" lonely chunks.")

		first_sixer = numset_counts.index(18)

		for i in numset_members[first_sixer]:
			translation = numset_offsets[first_sixer][numset_members[first_sixer].index(i)]
			shift = (-translation).dot(self.normallel.T)
			shifted_constraints = self.all_constraints[i] + np.repeat(shift.dot(self.twoface_normals.T).reshape(15,1),2,axis=1)
			print(np.round(shifted_constraints,5))
			# Waste time so we can print
			#for j in range(5000000):
			#	_ = i + j
			#	_ = _ * _ * _ * _
			print(self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[i])] - translation)
			print("(originally "+self.all_chosen_centers[i]+")")
			print(translation)
			print(i)

		numset_counts.sort()
		print(numset_counts)

		ordered_str = [[str(pair) for pair in np.round(np.array(self.all_constraints[u]))] for u in numset_ids]
		for i in range(len(self.all_constraints)):

			permutation = [ordered_str.index(x) for x in [str(pair) for pair in np.array(self.all_constraints[i])]]
			print(permutation)

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

