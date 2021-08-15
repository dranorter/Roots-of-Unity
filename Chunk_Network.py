from godot import exposed, export
from godot.bindings import _File as File
from godot import *
import random as r
import numpy as np
import numbers, time, math
from itertools import chain
from debugging import debugging
import traceback

COLOR = ResourceLoader.load("res://new_spatialmaterial.tres")


@exposed(tool=False)
class Chunk_Network(MeshInstance):
	phi = 1.61803398874989484820458683
	phi_powers = np.power(np.array([phi]*50), np.arange(50))
	worldplane = np.array([[phi, 0, 1, phi, 0, -1], [1, phi, 0, -1, phi, 0], [0, 1, phi, 0, -1, phi]])
	normalworld = worldplane / np.linalg.norm(worldplane[0])
	squareworld = normalworld.transpose().dot(normalworld)
	parallelspace = np.array([[-1 / phi, 0, 1, -1 / phi, 0, -1],
							  [1, -1 / phi, 0, -1, -1 / phi, 0],
							  [0, 1, -1 / phi, 0, -1, -1 / phi]])
	normallel = parallelspace / np.linalg.norm(parallelspace[0])
	squarallel = normallel.T.dot(normallel)
	deflation_face_axes = [[2, 1, 1, 1, 1, -1],
						   [1, 2, 1, -1, 1, 1],
						   [1, 1, 2, 1, -1, 1],
						   [1, -1, 1, 2, -1, -1],
						   [1, 1, -1, -1, 2, -1],
						   [-1, 1, 1, -1, -1, 2]]
	ch3 = [[1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 0],
		   [1, 0, 1, 0, 1, 0],
		   [1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0],
		   [0, 1, 1, 0, 1, 0],
		   [0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0],
		   [0, 0, 1, 1, 0, 1],
		   [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1]]
	twoface_axes = np.array(
		[[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1],
		 [0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0],
		 [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1]])

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
	twoface_normals = np.cross(twoface_projected[:, 0], twoface_projected[:, 1])
	twoface_normals = twoface_normals / np.linalg.norm(twoface_normals, axis=1)[0]

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
	twoface_normals_w = np.cross(twoface_projected_w[:, 0], twoface_projected_w[:, 1])
	twoface_normals_w = twoface_normals_w / np.linalg.norm(twoface_normals_w, axis=1)[0]

	possible_centers_live = np.array(
		[[0.5, 0.5, 0.5, 0., 0., 0.], [0.5, 0.5, 2., 1., -1.5, 1.], [0.5, 1., 1.5, 0., -0.5, 1.],
		 [0.5, 1.5, 1., -0.5, 0., 1.], [0.5, 2., 0.5, -1.5, 1., 1.], [0.5, 2., 2., -0.5, -0.5, 2.],
		 [1., 0.5, 1.5, 1., -0.5, 0.], [1., 1.5, 2., 0.5, -0.5, 1.], [1., 1.5, 0.5, -0.5, 1., 0.],
		 [1., 2., 1.5, -0.5, 0.5, 1.], [1.5, 0.5, 1., 1., 0., -0.5], [1.5, 1., 0.5, 0., 1., -0.5],
		 [1.5, 1., 2., 1., -0.5, 0.5], [1.5, 2., 1., -0.5, 1., 0.5], [2., 0.5, 0.5, 1., 1., -1.5],
		 [2., 0.5, 2., 2., -0.5, -0.5], [2., 1., 1.5, 1., 0.5, -0.5], [2., 1.5, 1., 0.5, 1., -0.5],
		 [2., 2., 0.5, -0.5, 2., -0.5], [2., 2., 2., 0.5, 0.5, 0.5]])
	possible_centers = ['[0.5 0.5 0.5 0.  0.  0. ]', '[ 0.5  0.5  2.   1.  -1.5  1. ]',
						'[ 0.5  1.   1.5  0.  -0.5  1. ]',
						'[ 0.5  1.5  1.  -0.5  0.   1. ]', '[ 0.5  2.   0.5 -1.5  1.   1. ]',
						'[ 0.5  2.   2.  -0.5 -0.5  2. ]',
						'[ 1.   0.5  1.5  1.  -0.5  0. ]', '[ 1.   1.5  2.   0.5 -0.5  1. ]',
						'[ 1.   1.5  0.5 -0.5  1.   0. ]',
						'[ 1.   2.   1.5 -0.5  0.5  1. ]', '[ 1.5  0.5  1.   1.   0.  -0.5]',
						'[ 1.5  1.   0.5  0.   1.  -0.5]',
						'[ 1.5  1.   2.   1.  -0.5  0.5]', '[ 1.5  2.   1.  -0.5  1.   0.5]',
						'[ 2.   0.5  0.5  1.   1.  -1.5]',
						'[ 2.   0.5  2.   2.  -0.5 -0.5]', '[ 2.   1.   1.5  1.   0.5 -0.5]',
						'[ 2.   1.5  1.   0.5  1.  -0.5]',
						'[ 2.   2.   0.5 -0.5  2.  -0.5]', '[2.  2.  2.  0.5 0.5 0.5]']

	player_pos = np.zeros((3,))

	def convert_chunklayouts(self, filename="res://chunklayouts_perf_14"):
		"""
		Loads a pure-Python repr of the chunk layouts (as generated by
		numpylattice.py), and then saves that as three separate files, one
		of which is in the more compact numpy save format.
		"""
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
		transformations = [(np.eye(6), np.eye(15))]  # ,
		#	(np.array(self.chunk_rot1),np.array(self.const_rot1)),
		#	(np.array(self.chunk_rot2),np.array(self.const_rot1).T)]

		for layout_file in [filename]:
			try:
				fs.open(layout_file, fs.READ)
				num_to_load = 100
				while not fs.eof_reached():  # len(all_chunks) < num_to_load:#not fs.eof_reached():
					# relevant chunk as chosen_center string
					ch_c = str(fs.get_line())
					# Constraint is 30 floats
					cstts = np.zeros((30))
					for i in range(30):
						cstts[i] = fs.get_real()
					cstts = cstts.reshape((15, 2))
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
					for m, n in transformations:
						ch_c_live = self.possible_centers_live[self.possible_centers.index(ch_c)]
						ch_c_live = m.dot(ch_c_live)
						t_ch_c = str(ch_c_live)
						t_cstts = n.dot(cstts)
						t_is_blocks = m.dot(np.array(is_blocks).T).T.tolist()
						t_os_blocks = m.dot(np.array(os_blocks).T).T.tolist()

						constraint_string = str((ch_c_live, t_cstts))
						if constraint_string not in all_sorted_constraints:
							all_constraints.append(t_cstts.tolist())
							constraints_sorted[t_ch_c].append(t_cstts.tolist())
							all_sorted_constraints.append(str((ch_c_live, t_cstts.tolist())))
							all_chunks.append(str((ch_c_live, t_is_blocks, t_os_blocks)))
							all_blocks.append((t_is_blocks, t_os_blocks))
							all_chosen_centers.append(t_ch_c)
							for block in t_is_blocks:
								all_block_axes.append(str(block - np.floor(block)))
						else:
							dupecounter += 1
							print("Found duplicate under rotation " + m)
					print("Loading chunklayouts..." + str(
						round(100 * sum([len(x) for x in constraints_sorted.values()]) / 5000)) + "%")
			except Exception as e:
				print("Encountered some sort of problem loading.")
				traceback.print_exc()
			fs.close()
		# print("Duplicates due to symmetry: "+str(dupecounter))
		print("Constraint counts for each possible chunk: " + str([len(x) for x in constraints_sorted.values()]))

		# Save file in a faster to load format
		#		fs.open("res://temp_test",fs.WRITE)
		#		fs.store_line(repr(all_constraints).replace('\n',''))
		#		fs.store_line(repr(constraints_sorted).replace('\n',''))
		#		fs.store_line(repr(all_blocks).replace('\n',''))
		#		fs.store_line(repr(all_chosen_centers).replace('\n',''))
		#		fs.close()

		possible_blocks = set()
		for blocklayout in all_blocks:
			combined = np.concatenate([blocklayout[0], blocklayout[1]])
			combined = combined * 2
			combined = np.array(np.round(combined), dtype=np.int64) / 2
			combined = [repr(list(x)) for x in combined]
			for block in combined:
				possible_blocks.add(block)
		blocklist = [eval(x) for x in possible_blocks]

		inside_bool = np.zeros((len(all_blocks), len(blocklist)), dtype=np.bool)
		outside_bool = np.zeros((len(all_blocks), len(blocklist)), dtype=np.bool)
		for i in range(len(all_blocks)):
			inside_bool[i] = np.any(
				np.all(np.repeat(all_blocks[i][0], len(blocklist), axis=0).reshape(-1, len(blocklist), 6)
					   - np.array(blocklist) == 0, axis=2).T, axis=1)
			outside_bool[i] = np.any(
				np.all(np.repeat(all_blocks[i][1], len(blocklist), axis=0).reshape(-1, len(blocklist), 6)
					   - np.array(blocklist) == 0, axis=2).T, axis=1)
			#			for j in range(len(blocklist)):
			#				inside_bool[i,j] = np.any(np.all(np.array(all_blocks[i][0])-np.array(blocklist[j])==0,axis=1))
			#				outside_bool[i,j] = np.any(np.all(np.array(all_blocks[i][1])-np.array(blocklist[j])==0,axis=1))
			print("Computing booleans..." + str(round(100 * i / len(all_blocks))) + "%")
		fs.open("res://temp_test", fs.WRITE)
		fs.store_line(repr((all_constraints).tolist()).replace('\n', ''))
		fs.store_line(repr((all_chosen_centers).tolist()).replace('\n', ''))
		fs.store_line(repr(blocklist).replace('\n', ''))
		fs.close()
		np.save("temp_test_is", inside_bool, allow_pickle=False)
		np.save("temp_test_os", outside_bool, allow_pickle=False)

	def save_templates_npy(self, filename="templates_dump"):
		"""
		Saves the current list of constraints in the three-file npy format.
		"""
		inside_bool = np.zeros((len(self.all_blocks), len(self.blocklist)), dtype=np.bool)
		outside_bool = np.zeros((len(self.all_blocks), len(self.blocklist)), dtype=np.bool)
		for i in range(len(self.all_blocks)):
			inside_bool[i] = np.any(np.all(np.repeat(self.all_blocks[i][0],
													 len(self.blocklist), axis=0).reshape(-1, len(self.blocklist), 6)
										   - np.array(self.blocklist) == 0, axis=2).T, axis=1)
			outside_bool[i] = np.any(np.all(np.repeat(self.all_blocks[i][1],
													  len(self.blocklist), axis=0).reshape(-1, len(self.blocklist), 6)
											- np.array(self.blocklist) == 0, axis=2).T, axis=1)
			print("Computing booleans..." + str(round(100 * i / len(self.all_blocks))) + "%")
		fs = File()
		fs.open("res://" + filename, fs.WRITE)
		# The data can sometimes vary between ndarrays and lists, depending on
		# what format it was loaded from, whether it was freshly processed/compactified,
		# etc.
		if type(self.all_constraints) == type(np.arange(1)):
			fs.store_line(repr((self.all_constraints.tolist())).replace('\n', ''))
		else:
			if type(self.all_constraints[0]) == type(np.arange(1)):
				fs.store_line(repr([constrt.tolist() for constrt in self.all_constraints]).replace('\n', ''))
			else:
				fs.store_line(repr(self.all_constraints).replace('\n', ''))
		if type(self.all_chosen_centers) == type(np.arange(1)):
			fs.store_line(repr((self.all_chosen_centers).tolist()).replace('\n', ''))
		else:
			if type(self.all_chosen_centers[0]) == type(np.arange(1)):
				fs.store_line(repr([center.tolist() for center in self.all_chosen_centers].replace('\n', '')))
			else:
				fs.store_line(repr(self.all_chosen_centers).replace('\n', ''))
		if type(self.blocklist) == type(np.arange(1)):
			fs.store_line(repr(self.blocklist.tolist()))
		else:
			if type(self.blocklist[0]) == type(np.arange(1)):
				fs.store_line(repr([block.tolist() for block in self.blocklist]).replace('\n', ''))
			else:
				fs.store_line(repr(self.blocklist))
		fs.close()
		np.save(filename + "_is", inside_bool, allow_pickle=False)
		np.save(filename + "_os", outside_bool, allow_pickle=False)

	def load_templates_npy(self, filename="simplified_constraints"):  # was "temp_test" for a long time
		fs = File()
		fs.open("res://" + filename, fs.READ)
		self.all_constraints = eval(str(fs.get_line()))
		self.all_chosen_centers = eval(str(fs.get_line()))
		self.blocklist = np.array(eval(str(fs.get_line())))
		fs.close()
		inside_blocks_bools = np.load(filename + "_is.npy")
		outside_blocks_bools = np.load(filename + "_os.npy")
		for i in range(inside_blocks_bools.shape[0]):
			self.all_blocks.append((self.blocklist[inside_blocks_bools[i]], self.blocklist[outside_blocks_bools[i]]))

	def load_templates_repr(self):
		starttime = time.perf_counter()
		fs = File()
		fs.open("res://chunk_layouts.repr", fs.READ)
		self.all_constraints = eval(str(fs.get_line()))
		print("Loaded part 1")
		print(time.perf_counter() - starttime)
		self.constraints_sorted = eval(str(fs.get_line()))
		print("Loaded part 2")
		print(time.perf_counter() - starttime)
		self.all_blocks = eval(str(fs.get_line()))
		print("Loaded part 3")
		print(time.perf_counter() - starttime)
		self.all_chosen_centers = eval(str(fs.get_line()))
		print("Loaded part 4")
		print(time.perf_counter() - starttime)
		fs.close()

	def simplify_constraints(self, no_new_values=True):
		"""
		The original generated constraints in self.all_constraints (as created by
		numpylattice.py) are very broad in some directions, sometimes spanning most
		of the space despite the small size of the actual constrained region (ie,
		other dimensions are doing most of the work). This function moves constraints
		inwards, closer to the actual region, so that some other processes can be
		faster.

		The new constraints are simply placed in self.all_constraints, and
		self.constraint_nums is updated accordingly.

		If no_new_values is True, this function will guarantee that the new
		self.constraint_nums is not larger than the old one, at a cost to
		constraint snugness.
		"""
		# We want to simplify the constraints. Many span more than half of one
		# of their 15 dimensions, but those values aren't actually relevant in
		# the 3D constraint space
		# because some of the other dimensions are pinned down to a range of
		# 1 or 2 constraint_nums values. So we want to see what values are actually
		# realizable, and collapse these large ranges down.
		self.all_simplified = []
		for i in range(len(self.all_constraints)):
			# print( [self.constraint_nums[dim].index(np.array(self.all_constraints[i])[dim,1])
			#				- self.constraint_nums[dim].index(np.array(self.all_constraints[i])[dim,0]) for dim in range(15)] )
			# To figure out what's relevant, we acquire a point inside the constraints
			# and re-base them as distance from that point. (This would have been easier
			# done when they were first created, since they were originally generated
			# as relative distance like this.)
			# (This step adds about 40 seconds)
			self.seed = np.array([r.random(), r.random(), r.random(), r.random(), r.random(), r.random()])
			self.make_seed_within_constraints(self.all_constraints[i])
			translated_constraints = (np.array(self.all_constraints[i]).T - np.array(self.twoface_normals).dot(
				self.seed.dot(self.normallel.T))).T

			# Sort all 30 constraints by how distant they are from this point.
			fields = [('tc', np.float), ('index', np.int), ('dir', np.int)]
			aug_tc = ([(np.abs(translated_constraints[j][0]), j, -1) for j in range(15)]
					  + [(np.abs(translated_constraints[j][1]), j, 1) for j in range(15)])
			sorted_tc = np.sort(np.array(aug_tc, dtype=fields), order='tc')


			# We're going to need two small margins to try and avoid floating point errors;
			# the second one very slightly bigger than the first. These of course limit how
			# snug our constraints will become. Larger margins also slow down the snugging process.
			margin1 = 0.00000000000001
			margin2 = 0.000000000000011
			# TODO Generating all the corners would enable very snug
			# constraints, and speed up the constraint search by a bit.
			# (I'm uncertain now whether it's very much.)
			# One direction might be to store, alongside each corner, information about which
			# plane intersections generated it. That way, another plane which cuts that corner
			# off knows which planes to use to generate new corners, and check whether those
			# corners are really part of the shape.
			corners = []
			# First we generate all corners between constraint planes.

			# Choosing the first 6 planes guarantees some intersections.
			closest_six = set()
			for cst in sorted_tc:
				closest_six.add(cst['index'])
				if len(closest_six) >= 6:
					break
			closest_six = list(closest_six)
			for index_i in range(6):
				for index_j in range(index_i, 6):
					for index_k in range(index_j, 6):
						plane_i = closest_six[index_i]
						plane_j = closest_six[index_j]
						plane_k = closest_six[index_k]
						basis_matrix = np.array([self.twoface_normals[plane_i], self.twoface_normals[plane_j],
												 self.twoface_normals[plane_k]])
						# Not all planes work together. (Even if distinct, some never meet in a corner.)
						if np.linalg.det(basis_matrix) != 0.0:
							for dirspec in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0),
											(1, 1, 1)]:
								new_corner = np.linalg.solve(basis_matrix,
															 [self.all_constraints[i][plane_i][dirspec[0]],
															  self.all_constraints[i][plane_j][dirspec[1]],
															  self.all_constraints[i][plane_k][dirspec[2]]])
								corners.append(new_corner)

			# Then we throw out all corners which fall outside the constraint
			# region, leaving only actual corners.
			# NOTE This has been altered to only use planes in "closest_six"
			# for its checks, rather than genuinely discarding all exterior points,
			# which only works if we successfully generated them all.
			for cst in sorted_tc:
				if cst['index'] in closest_six:
					# Get all corners' magnitude along this direction.
					# print(len(corners))
					corner_dots = [corner.dot(self.twoface_normals[cst['index']]) for corner in corners]

					if cst['dir'] == 1:
						if self.all_constraints[i][cst['index']][1] < max(corner_dots):
							# Constraint plane falls inside corners; we need to
							# remove some corners.

							# We need to be careful about floating point error here. Don't delete
							# corners which are just barely outside.
							smaller_corners_list = [corners[index] for index in range(len(corners))
													if corner_dots[index] <= self.all_constraints[i][cst['index']][
														1] + margin1]
							corners = smaller_corners_list
					else:
						# So, this is the case cst['dir'] == -1
						if self.all_constraints[i][cst['index']][0] > min(corner_dots):
							# Constraint plane falls inside corners; some corners not relevant.

							smaller_corners_list = [corners[index] for index in range(len(corners))
													if corner_dots[index] >= self.all_constraints[i][cst['index']][
														0] - margin1]
							corners = smaller_corners_list

			# Finally, we snug up all constraints to lie near these corners.
			snugged = np.array(self.all_constraints[i])
			for cst in sorted_tc:
				# Check all corners' magnitude in this direction
				corner_dots = np.sort([corner.dot(self.twoface_normals[cst['index']]) for corner in corners])

				# Is the seed inside the corners?
				seed_dot = self.seed.dot(self.normallel.T).dot(self.twoface_normals[cst['index']])
				if seed_dot > corner_dots[-1] or seed_dot < corner_dots[0]:
					print("\t\t\tSEED OUTSIDE ITS BOX")
					raise Exception("Seed outside its box; " + str(seed_dot) + " not between " + str(
						corner_dots[0]) + " and " + str(corner_dots[-1]) + "."
									+ "\n Actual constraint was from " + str(self.all_constraints[i][cst['index']][0])
									+ " to " + str(self.all_constraints[i][cst['index']][1]) + ".")

				# Want to use the lowest constraint_num beyond our highest corner.
				nums = np.array(self.constraint_nums[cst['index']])

				if cst['dir'] == 1:
					if self.all_constraints[i][cst['index']][1] > max(corner_dots):
						if no_new_values:
							# This direction doesn't affect the overall constraint. Can
							# we bring it closer?
							smallest = snugged[cst['index']][1]
							for f in range(len(nums)):
								if nums[f] >= max(corner_dots):
									smallest = nums[f]
									break
							snugged[cst['index']][1] = smallest
						else:
							# We must be careful about floating point error. If the corner we're
							# snugging up to lies very slightly inside the constraint region,
							# we may add a tiny facet.
							# TODO Certainly this could be a bit cleaner. Perhaps test the corner
							# to see if we can use its literal value.
							snugged[cst['index']][1] = min(max(corner_dots) + margin1,
														   self.all_constraints[i][cst['index']][1])
					else:
						# This indicates a constraint plane falling inside the
						# corners, which can't happen if we've chosen the true
						# corners. If the amount is greater than reasonable
						# floating point error, something has gone wrong.
						if self.all_constraints[i][cst['index']][1] - max(corner_dots) > margin2:
							raise Exception("A corner has been incorrectly included in constraint simplification.")
				if cst['dir'] == -1:
					smallest = snugged[cst['index']][0]
					if self.all_constraints[i][cst['index']][0] < min(corner_dots):
						if no_new_values:
							# Search for suitable member of nums
							for f in range(len(nums) - 1, -1, -1):
								if nums[f] <= min(corner_dots):
									smallest = nums[f]
									break
							snugged[cst['index']][0] = smallest
						else:
							# Maximal snugness save cautious margin
							snugged[cst['index']][0] = max(min(corner_dots) - margin1,
														   self.all_constraints[i][cst['index']][0])
					else:
						if self.all_constraints[i][cst['index']][0] - min(corner_dots) < -margin2:
							raise Exception("A corner has been incorrectly included in constraint simplification.")
				# smallest = cst['dir']*max(cst['dir']*nums[cst['dir']*nums >= cst['dir']*corner_dots[-1*cst['dir']]])

				if not self.satisfies_by(self.seed, snugged) > 0:
					print(str(cst['dir']) + " Seed got excluded")
					raise Exception(str(cst['dir']) + " Seed got excluded")
				if (self.all_constraints[i][cst['index']][1] - self.all_constraints[i][cst['index']][0]
						< snugged[cst['index']][1] - snugged[cst['index']][0]):
					print(str(cst['dir']) + "Somehow made constraint wider")
			self.all_simplified.append(snugged)

		#			print("Simplified comparison:")
		#			print(self.satisfies_by(self.seed,self.all_constraints[i]))
		#			print(self.satisfies_by(self.seed,self.all_simplified[i]))
		print("Length of simplified constraints:")
		print(len(self.all_simplified))

		self.simplified_constraint_nums = [set(), set(), set(), set(), set(), set(), set(), set(), set(), set(),
										   set(), set(), set(), set(), set()]
		for i in range(len(self.all_simplified)):
			for j in range(15):
				self.simplified_constraint_nums[j] = self.simplified_constraint_nums[j].union(
					set(self.all_simplified[i][j]))
		for i in range(15):
			sorted = list(self.simplified_constraint_nums[i])
			sorted.sort()
			self.simplified_constraint_nums[i] = sorted
		print("Number of constraint planes before simplification:")
		print(sum([len(nums) for nums in self.constraint_nums]))
		print("Number of constraint planes after simplification:")
		print(sum([len(nums) for nums in self.simplified_constraint_nums]))
		print([len(dimnums) for dimnums in self.simplified_constraint_nums])

		self.all_constraints = self.all_simplified
		self.constraint_nums = self.simplified_constraint_nums

	def draw_block_wireframe(self, block, st, multiplier):
		face_origin = np.floor(block).dot(self.worldplane.T) * multiplier
		face_tip = np.ceil(block).dot(self.worldplane.T) * multiplier
		dir1, dir2, dir3 = np.eye(6)[np.nonzero(np.ceil(block) - np.floor(block))[0]].dot(
			self.worldplane.T) * multiplier
		corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8 = (
			face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
			face_tip - dir1, face_tip - dir2, face_tip - dir3
		)
		dir1 = Vector3(dir1[0], dir1[1], dir1[2])
		dir2 = Vector3(dir2[0], dir2[1], dir2[2])
		dir3 = Vector3(dir3[0], dir3[1], dir3[2])
		# Draw by recombining
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)

	def draw_block(self, block, st, multiplier):
		face_origin = np.floor(block).dot(self.worldplane.T) * multiplier
		face_tip = np.ceil(block).dot(self.worldplane.T) * multiplier
		dir1, dir2, dir3 = np.eye(6)[np.nonzero(np.ceil(block) - np.floor(block))[0]].dot(
			self.worldplane.T) * multiplier
		# Make "right hand rule" apply
		if np.cross(dir1, dir2).dot(dir3) < 0:
			_ = dir1
			dir1 = dir2
			dir2 = _
		corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8 = (
			face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
			face_tip - dir1, face_tip - dir2, face_tip - dir3
		)
		dir1 = Vector3(dir1[0], dir1[1], dir1[2])
		dir2 = Vector3(dir2[0], dir2[1], dir2[2])
		dir3 = Vector3(dir3[0], dir3[1], dir3[2])
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1 + dir2)

		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1 + dir2)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)

		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2 + dir3)

		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2 + dir3)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)

		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3 + dir1)

		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3 + dir1)
		st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)

		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1 - dir2)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)

		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1 - dir2)

		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2 - dir3)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)

		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2 - dir3)

		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3 - dir1)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)

		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)
		st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3 - dir1)

	def make_seed_within_constraints(self, constraints):
		upper_limit = 10000
		counter = 0
		while counter < upper_limit and not self.satisfies(self.seed, constraints, strict=True):
			counter += 1
			axes = list(range(15))
			r.shuffle(axes)
			for axis in axes:
				proj_seed = self.seed.dot(self.normallel.T).dot(self.twoface_normals[axis])
				if proj_seed <= np.array(constraints)[axis, 0] or proj_seed >= np.array(constraints)[axis, 1]:
					self.seed -= proj_seed * self.twoface_normals[axis].dot(self.normallel)
					new_dist = r.random()
					self.seed += ((1 - new_dist) * np.array(constraints)[axis, 0] * self.twoface_normals[axis].dot(
						self.normallel)
								  + new_dist * np.array(constraints)[axis, 1] * self.twoface_normals[axis].dot(
								self.normallel))
				if self.satisfies(self.seed, constraints, strict=True):
					# Stop before we ruin it
					break
		if counter >= upper_limit:
			raise Exception("Exceeded allowed tries while trying to satisfy constraints " + str(constraints))

	def satisfies(self, vector, constraints, strict=False):
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
						   > np.array(constraints)[:, 0])
					and np.all(self.twoface_normals.dot(threevector)
							   < np.array(constraints)[:, 1]))
		return (np.all(self.twoface_normals.dot(threevector)
					   >= np.array(constraints)[:, 0])
				and np.all(self.twoface_normals.dot(threevector)
						   <= np.array(constraints)[:, 1]))

	def satisfies_by(self, vector, constraints):
		"""
		Returns the margin by which the vector (array or ndarray, 3D or 6D)
		falls inside the constraints. A 6D vector is first projected into the 3D
		'parallel space'. If vector is within the constraints, return value is
		positive, and represents the distance the vector would have to be moved
		to exit the constraints. If vector is outside the constraints, the
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
		return min(np.min(self.twoface_normals.dot(threevector) - np.array(constraints)[:, 0]),
				   np.min(-self.twoface_normals.dot(threevector) + np.array(constraints)[:, 1]))

	def generate_parents(self, template_index, offset, level=1):
		"""
		Takes a chunk template (as an index into self.all_blocks etc) and the
		offset at which the chunk's relative origin sits, and returns at least
		one valid superchunk for that chunk given the current seed. Return
		value is a list of tuples (index, offset) indicating each superchunk's
		template along with where its own origin will belong. Note, the returned
		offsets are scaled as if the child chunk were a single block, and
		should be scaled up in order to be drawn etc.
		'level' specifies the level of the chunk being handed to us; the returned
		chunk will be one level higher. Level '0' represents blocks, '1'
		represents the first chunks, etc. The effect of the level parameter is
		to map self.seed to an equivalent value for level + 1, before performing
		the search; the 'offset' is mapped up one level regardless of the given
		'level', being assumed to give coordinates of the given chunk in terms
		of blocks.
		"""

		chosen_center = self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[template_index])]
		ch3_member = np.ceil(chosen_center) - np.floor(chosen_center)

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

		# Raising the matrix to the power -level, we get the inverse of the
		# matrix raised to the power.
		chunk_axes_inv = np.linalg.matrix_power(np.array(self.deflation_face_axes).T, -level)

		chunk_axes_inv_seed = chunk_axes_inv.dot((self.seed).dot(self.squarallel))

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
								 >= self.center_guarantee[chunk][:, 0])
						  and np.all(self.twoface_normals.dot(chunk_axes_inv_seed.dot(self.normallel.T))
									 <= self.center_guarantee[chunk][:, 1])
						  for chunk in self.possible_centers])))

		# Nonetheless: successive applications of the chunk transformation make
		# our seed further and further from the origin. What's happening can be
		# pictured as: we start with an origin which is not a superchunk corner,
		# not a supersuperchunk corner, not a supersupersuperchunk corner, etc. -
		# so if we repeatedly convert the seed to more-and-more-super coordinates,
		# it eventually falls outside the constraints, representative of the fact
		# that our origin point is not in the lattice on this level. At such a
		# point, it no longer makes sense to translate the seed by the current
		# level's integer lattice, to check whether a point is sufficiently
		# near the worldplane.

		# So we need a process where we carefully step up the levels. Whenever
		# we find that a step leaves us measuring the seed from a point which
		# is no longer close enough to the worldplane be included, we need to
		# switch to a nearby point which *is* included. The way in which this
		# is navigated undoubtedly will affect how much floating point error
		# is accumulated.

		# TODO figure out what the best way of minimizing floating point error is.
		# For now I'll start with a point we know is on the lattice at the right
		# level - namely the "offset" we're handed - and transform the seed in
		# one step.

		# "offset" starts out at the level "level", and the seed starts out at
		# level 1, so we want to apply deflation_face_axes level-1 times.
		lowered_offset = np.linalg.matrix_power(np.array(self.deflation_face_axes), (level - 1)).dot(offset)
		safer_seed = (self.seed - lowered_offset).dot(self.squarallel)
		chunk_axes_inv_seed = chunk_axes_inv.dot(safer_seed)
		print("Calculating seed from safe point " + str(lowered_offset))
		print("Translated seed (should be < " + str(round(math.pow(1 / 4.7, level), 2)) + "):" + str(safer_seed))
		print("Rescaled seed: " + str(chunk_axes_inv_seed))

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

		# We only want to apply the matrix here once; we're assuming both
		# chunk_center and offset are already in terms of the scale one level
		# down.
		chunk_as_block_center = np.round(
			(np.linalg.inv(self.deflation_face_axes)).dot(chosen_center + offset) * 2) / 2.0
		chunk_as_block_origin = np.round(np.linalg.inv(self.deflation_face_axes).dot(offset))
		print("Rescaled chunk center: " + str(chunk_as_block_center))
		print("Rescaled chunk origin: " + str(np.linalg.inv(self.deflation_face_axes).dot(offset)))

		closest_non_hit = None
		inside_hits = []
		outside_hits = []
		for i in range(len(self.all_blocks)):
			inside_blocks = np.array(self.all_blocks[i][0])
			constraint = np.array(self.all_constraints[i])
			# aligned_blocks still have coords relative to the template
			aligned_blocks = inside_blocks[np.nonzero(np.all(inside_blocks - chunk_as_block_center
															 - np.round(inside_blocks - chunk_as_block_center) == 0,
															 axis=1))[0]]
			if len(list(aligned_blocks)) == 0:
				continue
			# block_offsets measures from template origin to the canonical corner of each block
			block_offsets = aligned_blocks - chunk_as_block_center + chunk_as_block_origin
			# We find proposed placement of the template by subtracting these.
			proposed_origins = -block_offsets
			# proposed_origins are relative to where the seed's been *translated*, ie,
			# chunk_as_block_origin. We then translate the seed the extra inch...
			translated_seeds = (chunk_axes_inv_seed - proposed_origins).dot(self.normallel.T)
			a_hit = np.all([np.all(self.twoface_normals.dot(translated_seeds.T).T >= constraint[:, 0], axis=1),
							np.all(self.twoface_normals.dot(translated_seeds.T).T <= constraint[:, 1], axis=1)], axis=0)
			if np.any(a_hit):
				print("We're inside template #" + str(i) + ", at offset" + str(
					block_offsets[np.array(np.nonzero(a_hit))[0, 0]]))
				inside_hits = [(i, proposed_origins[j] + chunk_as_block_origin) for j in np.nonzero(a_hit)[0]]
				break
		# Checking outside blocks takes a lot longer, we only want to  do it if we need to
		if len(inside_hits) == 0:
			for i in range(len(self.all_blocks)):
				outside_blocks = np.array(self.all_blocks[i][1])
				constraint = np.array(self.all_constraints[i])
				aligned_blocks = outside_blocks[np.nonzero(np.all(outside_blocks - chunk_as_block_center
																  - np.round(
					outside_blocks - chunk_as_block_center) == 0, axis=1))[0]]
				if len(list(aligned_blocks)) == 0:
					# print("Somehow no aligned blocks on template#"+str(i))
					continue
				block_offsets = aligned_blocks - chunk_as_block_center + chunk_as_block_origin
				proposed_origins = -block_offsets
				translated_seeds = (chunk_axes_inv_seed - proposed_origins).dot(self.normallel.T)
				a_hit = np.all([np.all(self.twoface_normals.dot(translated_seeds.T).T >= constraint[:, 0], axis=1),
								np.all(self.twoface_normals.dot(translated_seeds.T).T <= constraint[:, 1], axis=1)],
							   axis=0)
				if np.any(a_hit):
					print("Some sort of neighbor block hit! ")
					outside_hits = [(i, proposed_origins[j] + chunk_as_block_origin) for j in np.nonzero(a_hit)[0]]
					break
		hits = inside_hits + outside_hits
		print("Found " + str(len(hits)) + " possible superchunks.")
		if len(hits) == 0:
			print("(WARNING) Using closest non-hit, which misses the constraint by:")
			print(closest_non_hit[2])
			hits = [(closest_non_hit[0], closest_non_hit[1])]

		return hits

	def generate_children(self, i, offset, level=2):
		"""
		Takes a chunk, represented by a chunk template (as an index i suitable
		for self.all_blocks etc.) together with an offset for where that chunk
		lies, and returns appropriate child chunks, based on the seed, which
		would fill out each 'block' in the given chunk.
		The 'level' of the chunk is the number of times one would have to
		generate children to get to 'block' scale; IE, a level 1 chunk contains
		blocks, a level 2 chunk contains chunks that contain blocks. When given
		a level of 1 or below, however, chunk templates are still returned,
		allowing arbitrary subdivision.
		'offset' is interpreted in the coordinates of the blocks of the chunk -
		ie, at one level below what's given in 'level'.
		Return value is a list of tuples (index,location), giving an index
		suitable for self.all_blocks etc, and a location in coordinates of the
		same level as the children.
		"""

		# Move seed to our offset, then scale it
		absolute_offset = np.linalg.matrix_power(self.deflation_face_axes, level - 1).dot(offset)
		translated_seed = (self.seed - absolute_offset).dot(self.squarallel)
		# Seed belongs two levels below the current one
		current_level_seed = np.linalg.matrix_power(self.deflation_face_axes, 2 - level).dot(translated_seed)

		children = []

		# TODO I'm making a major leap of faith here, that the "inside blocks"
		# are genuinely all we need. It does look fine, though.
		chunks = np.array(
			self.all_blocks[i][0])  # np.concatenate([np.array(self.all_blocks[i][0]), np.array(self.all_blocks[i][1])])
		# print("Trying to find "+str(len(chunks))+" children")
		# Take the floor to get the correct origin point of the chunk for lookup.
		points = np.floor(chunks)

		# Taking the floor produces duplicates - an average of 3 copies per
		# point. Removing those would be nice for performance, but removing them
		# right away is inconvenient because when we get several chunks back
		# from the self.find_satisfied call below, we wouldn't know how to try
		# to match them up with the template and discard those not in the "chunks"
		# list. So we create a lookup table so we only have to do the work once.
		unique_points = [points[0]]
		unique_lookup = np.zeros(points[:, 0].shape, dtype=int)
		for chunk_i in range(len(chunks)):
			identical_points = np.all(np.abs(points - points[chunk_i]) < 0.1, axis=1)
			if not np.any(identical_points[:chunk_i]):
				unique_lookup[identical_points] = len(unique_points)
				unique_points.append(points[chunk_i])
		#		points = np.array(unique_points)

		chunkscaled_positions = (np.array(self.deflation_face_axes)).dot(np.transpose(unique_points)).T
		chunkscaled_offset = (np.array(self.deflation_face_axes)).dot(offset)
		chunk_seeds = (current_level_seed - chunkscaled_positions).dot(self.squarallel)

		found_satisfied = [self.find_satisfied(seed) for seed in chunk_seeds]
		# TODO Nested for loops; optimize?
		# for seed_i in range(len(chunk_seeds)):
		for chunk_i in range(len(chunks)):
			# Index into the computations
			u_i = unique_lookup[chunk_i]
			# children.append((self.find_satisfied(chunk_seeds[seed_i]),chunkscaled_positions[seed_i]+chunkscaled_offset))

			location = chunkscaled_positions[u_i] + chunkscaled_offset
			for template_index in found_satisfied[u_i]:
				template_center = self.possible_centers_live[
					self.possible_centers.index(self.all_chosen_centers[template_index])]
				if np.all((template_center + chunks[chunk_i]) - np.round(template_center + chunks[chunk_i]) != 0):
					children.append((template_index, location))
		return children

	def find_satisfied(self, seedvalue):
		"""
		Takes a seedvalue and returns a list of chunk template indices which
		would satisfy that seed at the origin.
		"""
		seed_15 = self.twoface_normals.dot(seedvalue.dot(self.normallel.T))

		# TODO When I add some sort of testing system, there should be a test
		# here for whether the single-hit cases still pass the self.satisfies
		# test.
		# The tree ought to get us 1 to 3 possibilities to sort through.
		hits = self.constraint_tree.find(seed_15)
		# Hits can be spurious if the seed doesn't happen to be separated
		# from the constraint region by any of the planes which the tree
		# made use of. (If the seed is outside of a constraint, there's
		# always some plane the tree *could have* tested in order to see this.
		# But when several constraint regions are separated by no single plane,
		# the tree won't test every plane.)
		# TODO If I ever wanted to remove this call to self.satisfies, I could
		# have the leaf nodes of the tree do some final tests. In principle
		# the leaf nodes could know which planes haven't been tested yet and
		# cross their regions, and then use those to determine which
		# single template to return. Almost equivalent, but easier on the
		# lookup tree code, would be cutting up all the constraint regions
		# so that the overlapping regions are their own separate entries on
		# the constraints list. I'd have a choice of exactly how to cut
		# everything up yet leave the regions convex.
		if len(hits) == 1:
			# If there's just one, the point must be inside it for two reasons.
			# One, all the bounding constraints got checked. Two, any seed
			# must be in at least one region.
			return hits
		real_hits = []
		for hit in hits:
			if self.satisfies(seedvalue, self.all_constraints[hit]):
				real_hits.append(hit)
		if len(real_hits) > 0:
			return real_hits
		raise Exception("No valid hits out of " + str(len(hits)) + " hits.")

	def update_player_pos(self, pos, transform):
		trans_inv = transform.basis.inverse()
		rotation = np.array([[trans_inv.x.x,trans_inv.x.y,trans_inv.x.z],
							 [trans_inv.y.x,trans_inv.y.y,trans_inv.y.z],
							 [trans_inv.z.x,trans_inv.z.y,trans_inv.z.z]])
		position = np.array([pos.x,pos.y,pos.z])
		self.player_pos = position.dot(rotation.T)

	def _update(self):
		# Manage a sphere (or maybe spheroid) of loaded blocks around the player.
		# Want to load in just one or a few per update, unload distant ones as the player
		# moves, stop expanding loaded sphere before its size causes frame or memory issues,
		# and on top of all that load really distant super-super-chunks to provide low-LOD distant rendering.
		# To load chunks based on distance, each chunk in a bigger sphere (of superchunks) must already
		# know its center point. To keep the bigger sphere available, each superchunk in a yet-bigger sphere
		# (of super-super-chunks) must know its center-point.
		# Say for sake of argument we have a tiny sphere of blocks, fitting inside one chunk. The sphere of chunks
		# just needs to be big enough that each chunk inside the sphere of blocks (IE, with blocks filled in) has all
		# its neighbors exist (IE, without blocks filled in). So when the sphere of blocks falls exactly within one
		# chunk, there could be just one chunk with loaded blocks, and all its neighbors loaded without their blocks.
		# Next, to maintain this state as we move, we need any superchunk with loaded chunks to also have all
		# neighboring superchunks loaded.
		# Clearly if we proceed like this upwards, we load an infinite number of higher-level chunks. At some point
		# we're loading way more than we need. A really high-level chunk doesn't need to load any neighbors if the
		# player is deep within, thousands of blocks from any face of the chunk.
		# Actually I think it's all very simple. There is a sphere around the player, and we need to generate all
		# blocks within that sphere. This means we need to generate all chunks which might contain blocks within the
		# sphere - which we can simplify to, all chunks with any overlap with the sphere. And to ensure we do that, we
		# generate all superchunks with any overlap with the sphere. We proceed upwards like this until the sphere
		# is strictly contained in some higher-level chunk.
		# There are three problems with that which need addressed.
		# One, what if this never terminates? Couldn't we be on a chunk corner which is also a superchunk corner which
		# is also a super-super-chunk corner, and so on ad infinitum? In the cube case I avoided this issue by
		# selecting an origin for the player to start out at, where they're in the middle of a chunk which is in the
		# middle of a superchunk which is in the middle of a super-super-chunk, and so on and so on ad infinitum. This
		# way, the player is infinitely far from any points with that problem. But on the aperiodic grid, I'm actually
		# starting the player off at a random point, so how can I be sure? Well, every higher level of chunk corners
		# actually represents 6D lattice points which are an order of magnitude closer to the world-plane. So a point
		# could be extremely close and remain an intersection for quite a while, but eventually there will be some
		# level of chunk where it's excluded. This means that chunk corners are guaranteed get further and further away
		# if we go up enough levels. Since the seed is chosen to be non-singular, no 6D point lies exactly on the world.
		# Second, blocks cross chunk boundaries! So the loading sphere could be entirely contained within some
		# super-chunk, yet it might intersect a chunk which belongs to some neighbor super-chunk. This could be addressed
		# directly by using the actual ownership boundary of each chunk (but just based on direct sub-chunks) instead
		# of the rhombohedral boundary. An alternative would be to determine some distance from the boundary and
		# simply assume that points closer than that might be intersecting a neighbor. This effectively means the sphere
		# gets bigger at each stage, yet my instinct is that it would be OK. Finally, the way I've calculated chunk
		# templates, each one actually knows positions of any neighboring blocks which reach inside the template. So
		# a chunk can just check whether neighboring sub-chunks intersect the sphere, and then load in the corresponding
		# neighboring chunk.
		# Third, checking for actual overlap with a sphere may be expensive, even if we're just doing it with
		# rhombohedra rather than a sub-chunk outline. One solution might be to, again, increase the size of the sphere
		# at each stage, just enough so that we can switch to a check for intersecting corners rather than arbitrary
		# intersection. So at each level we'd start with the original sphere, then consider the worst-case placement of
		# that sphere tangent to a golden rhombus of the current chunk scale, and increase the radius just enough that
		# it would contain the rhombus' corners in such a case. Would the process ever stop making superchunks? I think
		# it would, but it would produce many more levels of superchunk than necessary. Better to just write a nice
		# sphere intersection test.
		# Finally, we shouldn't just expand such a sphere until we flirt with memory and rendering limits. The
		# dropoff in detail after the sphere is exited is too harsh. Ideally what we want is for parts of the world
		# which the player can actually see to become more detailed, without too much switching back and forth as to
		# what is and isn't loaded, and with fairly even coverage.

		# For now, we use a loading radius, which cautiously expands.


		# Is the player contained in a fully generated chunk?
		#	# If not, generating that chunk is enough for this frame; do it and return.
		#   # But first, reduce the loading_radius since it obviously failed.

		# Starting at the top-level chunk:
		# (These checks might not be expensive but maybe we could start lower sometimes)
		#	# Does this chunk fully contain the sphere? That is, does the sphere fall within 'safe' inner margins,
		#	# where neighbor chunks won't intrude?
		#	#	# If yes, do procedure 'alpha':
		#	#	#  Identify all sub-chunks which overlap the sphere.
		#	#	#  Is any of them not subdivided into sub-sub-chunks?
		#	#	#	# If yes, do procedure 'add and prune':
		#	#	#	#	# Are we at the limit of number of loaded chunks?
		#	#	#	#	#	# If yes, identify a low-priority chunk to unload.
		#	#	#	#	# Generate children for the identified chunk.
		#	#	#	# If no, repeat proceduce alpha on each sub-chunk.
		#	#	# If no: do any neighbor sub-chunks (from the outer part of our template) overlap the sphere?
		#	#	#	# If yes, add a new top-level chunk, and re-start the process there.
		#	#	#	# If no, we do fully contain it. Do procedure 'alhpa' above.
		pass

	def _ready(self):
		starttime = time.perf_counter()

		# This variable is temporary until I use actual round planets
		self.gravity_direction = Vector3(0, -1, 0)

		# self.convert_chunklayouts()

		print("Loading from existing file...")

		self.all_constraints = []
		self.all_blocks = []
		self.blocklist = []

		print(time.perf_counter() - starttime)
		print("Loading...")

		self.load_templates_npy()
		print("Done loading " + str(len(self.all_constraints)) + " templates")
		print(time.perf_counter() - starttime)

		self.constraint_nums = [set(), set(), set(), set(), set(), set(), set(), set(), set(), set(),
								set(), set(), set(), set(), set()]
		for i in range(len(self.all_constraints)):
			for j in range(15):
				self.constraint_nums[j] = self.constraint_nums[j].union(set(self.all_constraints[i][j]))
		for i in range(15):
			sorted = list(self.constraint_nums[i])
			sorted.sort()
			self.constraint_nums[i] = sorted

		# Original, pre-simplification size of constraint_nums in each dimension:
		# 400, 364, 574, 372, 395, 359, 504, 553, 382, 369, 470, 556, 404, 351, 359
		# After 20 calls of self.simplify_constraints():
		# 359, 315, 514, 336, 343, 325, 440, 509, 339, 338, 411, 500, 377, 328, 324

		# TODO Could still try optimizing generation by pre-generating higher-level chunks --
		# e.g. making templates for supersuperchunks. But at this point it feels like that
		# would be premature optimization.

		# Now with somewhat more compact constraints, we can create a decent-speed
		# lookup tree.

		# self.constraint_tree = ConstraintTree.sort(self.all_constraints, list(range(len(self.all_constraints))), self.constraint_nums)
		# self.constraint_tree.save()
		# print("Tree has been saved!")
		# print("Done constructing constraint tree")
		self.constraint_tree = ConstraintTree.load()
		print("Done loading constraint search tree.")
		print(time.perf_counter() - starttime)

		# Choose one chunk to display
		chunk_num = r.choice(range(len(self.all_blocks)))
		chosen_center = self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[chunk_num])]
		inside_blocks = self.all_blocks[chunk_num][0]
		outside_blocks = self.all_blocks[chunk_num][1]

		print("Chosen center: " + str(chosen_center))

		array_mesh = ArrayMesh()
		self.mesh = array_mesh

		# Now try to find a super-chunk for this chunk

		# First we have to find an offset value within the chunk constraints.
		# I'll call this the "seed" since it will determine the entire lattice.

		# Starting value, will get moved to inside the chosen chunk orientation constraints.
		self.seed = np.array([r.random(), r.random(), r.random(), r.random(), r.random(), r.random()])
		self.seed = self.seed.dot(self.squarallel)

		self.center_guarantee = dict()
		for center in self.possible_centers_live:
			center_axes = 1 - np.array(center - np.floor(center)) * 2
			center_origin = center - np.array(self.deflation_face_axes).T.dot(center_axes) / 2
			center_axis1 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][0]])
			center_axis2 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][1]])
			center_axis3 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][2]])
			chunk_corners = np.array([center_origin,
									  center_origin + center_axis1, center_origin + center_axis2,
									  center_origin + center_axis3,
									  center_origin + center_axis1 + center_axis2,
									  center_origin + center_axis1 + center_axis3,
									  center_origin + center_axis2 + center_axis3,
									  center_origin + center_axis1 + center_axis2 + center_axis3])
			a = np.sum(chunk_corners, axis=0) / 8
			center_constraints = np.sum(
				np.stack(np.repeat([chunk_corners - a], 30, axis=0), axis=1).dot(self.normallel.T)
				* np.concatenate(np.array([self.twoface_normals, -self.twoface_normals]).reshape((1, 30, 3))), axis=2)
			overall_center_constraints = 0.9732489894677302 / (self.phi * self.phi * self.phi) - np.max(
				center_constraints, axis=0)
			translated_constraints = (overall_center_constraints * np.concatenate([-np.ones(15), np.ones(15)])
									  + np.concatenate([self.twoface_normals, self.twoface_normals]).dot(
						a.dot(self.normallel.T)))
			translated_constraints = (translated_constraints).reshape((2, 15)).T
			self.center_guarantee[str(center)] = translated_constraints

		ch3_member = np.ceil(chosen_center) - np.floor(chosen_center)
		three_axes = np.nonzero(ch3_member)[0]
		constraint_dims = np.nonzero(1 - np.any(self.twoface_axes - ch3_member > 0, axis=1))[0]
		# constraint_dims gives us indices into center_guarantee as well as twoface_axes,
		# twoface_normals and twoface_projected.
		for i in constraint_dims:
			third_axis = np.nonzero(ch3_member - self.twoface_axes[i])[0][0]
			axis_scale = np.eye(6)[third_axis].dot(self.normallel.T).dot(self.twoface_normals[i])
			divergence = self.center_guarantee[str(chosen_center)][i] - self.twoface_normals[i].dot(
				self.seed.dot(self.normallel.T))
			# Is point outside the constraints in this direction?
			if divergence[0] * divergence[1] >= 0:
				rand_pos = r.random()
				move = (divergence[0] * rand_pos + divergence[1] * (1 - rand_pos)) * np.eye(6)[third_axis].dot(
					self.normallel.T) / axis_scale
				self.seed = self.seed + move.dot(self.normallel)

				generates_correct_chunk = (np.all(self.twoface_normals.dot(self.seed.dot(self.normallel.T))
												  > self.center_guarantee[str(chosen_center)][:, 0])
										   and np.all(self.twoface_normals.dot(self.seed.dot(self.normallel.T))
													  < self.center_guarantee[str(chosen_center)][:, 1]))
				if generates_correct_chunk:
					# Break early before we mess it up
					break
		self.make_seed_within_constraints(self.all_constraints[chunk_num])
		print("Chose a seed within constraints for template #" + str(chunk_num) + ":")
		print(self.seed)

		# Now that "seed" is a valid offset for our chosen chunk, we need to
		# determine which superchunk it can fit in.
		# There should logically be just one option, since the seed uniquely
		# determines the whole grid.
		# TODO: (Runs so far have always found either 1 or 2 valid superchunks.
		# But, should do a more rigorous test, ideally verifying back in
		# numpylattice.py that this is inevitable.)

		self.highest_chunk = Chunk(self,chunk_num,[0,0,0,0,0,0],1)
		self.highest_chunk.is_topmost = True

		#hits = self.generate_parents(chunk_num, [0, 0, 0, 0, 0, 0])
		print("getting superchunk")
		self.highest_chunk.get_parent()
		print(time.perf_counter() - starttime)

		# print("Trying to generate super-super-chunk...")
		# print(time.perf_counter() - starttime)
		# superhits = self.generate_parents(hits[0][0], hits[0][1], level=2)
		# print("Got " + str(len(superhits)) + " supersuperchunks")
		# print(time.perf_counter() - starttime)
		print("getting super-super-chunk")
		self.highest_chunk.get_parent()
		print(time.perf_counter() - starttime)

		# supersuperhits = self.generate_parents(superhits[0][0],superhits[0][1],level=3)
		# print("Got "+str(len(supersuperhits))+" supersupersuperchunks")
		# print(time.perf_counter()-starttime)

		# all_supersuperchunks = []
		# for i, offset in supersuperhits:
		#	all_supersuperchunks += self.generate_children(i, offset, level=4)
		# print("Supersuperchunks total: "+str(len(all_supersuperchunks)))
		# superhits = all_supersuperchunks

		# all_superchunks = []
		# for i, offset in superhits:
		# 	all_superchunks += self.generate_children(i, offset,
		# 											  level=3)  # [(int(l[0]),l[1:]) for l in self.generate_children(i,offset,level=3)]
		# print("Superchunks total: " + str(len(all_superchunks)))
		all_superchunks = self.highest_chunk.get_children()

		# Draw the valid chunk(s)
		# for superchunk in hits:
		# 	i, offset = superchunk
		# 	multiplier = math.pow(self.phi, 3)
		# 	st = SurfaceTool()
		#
		# 	st.begin(Mesh.PRIMITIVE_LINES)
		# 	st.add_color(Color(1, .2, 1))
		# 	for block in (np.array(self.all_blocks[i][1]) + offset):
		# 		self.draw_block_wireframe(block, st, multiplier)
		# 	st.commit(self.mesh)
		# 	self.mesh.surface_set_material(self.mesh.get_surface_count() - 1, COLOR)
		#
		# 	st.begin(Mesh.PRIMITIVE_LINES)
		# 	st.add_color(Color(0.5, 0, 1))
		# 	for block in (np.array(self.all_blocks[i][0]) + offset):
		# 		self.draw_block_wireframe(block, st, multiplier)
		# 	st.commit(self.mesh)
		# 	self.mesh.surface_set_material(self.mesh.get_surface_count() - 1, COLOR)

		# Draw the valid superchunks?
		# for (i, offset) in superhits:
		# 	multiplier = math.pow(self.phi, 6)
		# 	st = SurfaceTool()
		#
		# 	st.begin(Mesh.PRIMITIVE_LINES)
		# 	st.add_color(Color(.2, 1, 1))
		# 	for block in (np.array(self.all_blocks[i][1]) + offset):
		# 		self.draw_block_wireframe(block, st, multiplier)
		# 	st.commit(self.mesh)
		# 	self.mesh.surface_set_material(self.mesh.get_surface_count() - 1, COLOR)
		#
		# 	st.begin(Mesh.PRIMITIVE_LINES)
		# 	st.add_color(Color(0, 0.5, 1))
		# 	for block in (np.array(self.all_blocks[i][0]) + offset):
		# 		self.draw_block_wireframe(block, st, multiplier)
		# 	st.commit(self.mesh)
		# 	self.mesh.surface_set_material(self.mesh.get_surface_count() - 1, COLOR)

		# Now we draw the blocks inside those chunks.

		# children = chain(*[self.generate_children(i,offset) for i, offset in all_superchunks])
		# Cleverer line above didn't turn out to be faster.
		# children = []
		# for i, offset in all_superchunks:  # all_superchunks:#hits:
		# 	children += self.generate_children(i, offset)
		children = []
		for c in all_superchunks:
			children += c.get_children()
		# These children are the chunks already drawn above, but now
		# accompanied by an appropriate offset and template index so we can draw
		# their blocks. So, test: do the children correspond to the aldready-drawn
		# chunks?
		#		children2 = []
		#		for i, offset in hits:
		#
		print("All " + str(len(children)) + " chunks now generated. Time:")
		print(time.perf_counter() - starttime)
		# Draw these
		# multiplier = 1
		# st = SurfaceTool()
		# st.begin(Mesh.PRIMITIVE_TRIANGLES)
		# st.add_color(Color(r.random(), r.random(), r.random()))

		# List the block coordinates
		# List comp. version faster, but crashes w/ large numbers of blocks.
		# block_comprehension = list(
		# 	chain(*[[block for block in np.concatenate([self.all_blocks[i][0], self.all_blocks[i][1]]) + offset]
		# 			for i, offset in children]))
		# block_comprehension = np.zeros((0,6))
		# for i, offset in children:
		#	block_comprehension = np.concatenate([block_comprehension,self.all_blocks[i][0]])
		# print("All " + str(len(block_comprehension)) + " blocks generated. Time:")
		# print(time.perf_counter() - starttime)

		# Now pass them to the draw function
		# When no drawing occurs, takes about 21 seconds (for-loop version took 40)
		# With drawing, takes about 42 seconds
		def decide(block):
			#print("Entered drawp. "+ str(block[0] + block[1] + block[2]))
			return block[0] + block[1] + block[2] == 0.5

		#list(map(decide, block_comprehension))
		# for block in block_comprehension:
		#	if block[0] + block[1] + block[2] == 0.5:
		#		self.draw_block(block,st,multiplier)
		# for i, offset in children:
		#	for block in (np.concatenate([self.all_blocks[i][0],self.all_blocks[i][1]]) + offset):
		#		if ( block[0] + block[1] + block[3] in [0,1,2]):
		#			self.draw_block(block,st,multiplier)
		for c in children:
			c.draw_mesh()
		print("Done calling draw. Time:")
		print(time.perf_counter() - starttime)
		# if len(children) > 0:
		# 	st.generate_normals()
		# 	st.commit(self.mesh)
		# 	self.mesh.surface_set_material(self.mesh.get_surface_count() - 1, COLOR)
		# print("Mesh updated. Time:")
		# print(time.perf_counter() - starttime)

	def test_templates(self):
		starttime = time.perf_counter()
		possible_blocks = set()
		for blocklayout in self.all_blocks:
			combined = np.concatenate([blocklayout[0], blocklayout[1]])
			combined = combined * 2
			combined = np.array(np.round(combined), dtype=np.int64)
			combined = [repr(list(x)) for x in combined]
			for block in combined:
				possible_blocks.add(block)
		print("Set up possible blocks list. " + str(len(possible_blocks)) + " occur.")  # 4042
		print(time.perf_counter() - starttime)

		possible_layouts = []
		blocklist = [eval(x) for x in possible_blocks]
		for blocklayout in self.all_blocks:
			combined = np.concatenate([blocklayout[0], blocklayout[1]])
			combined = np.round(combined * 2)
			layout = np.any(
				np.all(np.repeat(blocklist, len(combined), axis=0).reshape(-1, len(combined), 6) - combined == 0,
					   axis=2), axis=1)
			novel = True
			for poss in possible_layouts:
				if np.all(layout == poss):
					novel = False
					debugging.breakpoint()
			if novel:
				possible_layouts.append(layout)
		print("Number of unique layouts according to more careful calculation:")
		print(len(possible_layouts))
		print(time.perf_counter() - starttime)

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
			center_axes = 1 - np.array(center - np.floor(center)) * 2
			center_origin = center - np.array(self.deflation_face_axes).T.dot(center_axes) / 2
			center_axis1 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][0]])
			center_axis2 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][1]])
			center_axis3 = np.array(self.deflation_face_axes[np.nonzero(center_axes)[0][2]])
			chunk_corners = np.array([center_origin,
									  center_origin + center_axis1, center_origin + center_axis2,
									  center_origin + center_axis3,
									  center_origin + center_axis1 + center_axis2,
									  center_origin + center_axis1 + center_axis3,
									  center_origin + center_axis2 + center_axis3,
									  center_origin + center_axis1 + center_axis2 + center_axis3])
			# We consider all translations, since different chunks have the origin
			# at different corners.
			for corner in chunk_corners:
				shift = (-corner).dot(self.normallel.T)
				shifted_constraints = self.all_constraints[i] + np.repeat(
					shift.dot(self.twoface_normals.T).reshape(15, 1), 2, axis=1)
				# str_c = [str(pair) for pair in np.array(self.all_constraints[i])]
				str_c = np.abs(np.round(shifted_constraints, 13)).flatten().tolist()
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
				# print("Match with "+str(self.all_constraints[i]))
		print(str(len(constraint_numbers)) + " constraint numbers.")
		print(str(len(constraint_numsets)) + " constraint numsets.")
		print(str(len([x for x in numset_counts if x == 1])) + " lonely chunks.")

		first_sixer = numset_counts.index(18)

		for i in numset_members[first_sixer]:
			translation = numset_offsets[first_sixer][numset_members[first_sixer].index(i)]
			shift = (-translation).dot(self.normallel.T)
			shifted_constraints = self.all_constraints[i] + np.repeat(shift.dot(self.twoface_normals.T).reshape(15, 1),
																	  2, axis=1)
			print(np.round(shifted_constraints, 5))
			# Waste time so we can print
			# for j in range(5000000):
			#	_ = i + j
			#	_ = _ * _ * _ * _
			print(self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[i])] - translation)
			print("(originally " + self.all_chosen_centers[i] + ")")
			print(translation)
			print(i)

		numset_counts.sort()
		print(numset_counts)

		ordered_str = [[str(pair) for pair in np.round(np.array(self.all_constraints[u]))] for u in numset_ids]
		for i in range(len(self.all_constraints)):
			permutation = [ordered_str.index(x) for x in [str(pair) for pair in np.array(self.all_constraints[i])]]
			print(permutation)


class ConstraintTree:
	"""
	Stores a hierarchy of sorted constraints, which can then be queried with a 15-dimensional point.
	The return value, however, is an index into the constraint-set, so this class has to know the
	overall set intended for indexing.
	"""
	sort_dim = 0
	sort_index = 0
	sort_threshhold = 0
	values = []
	is_leaf = False

	def __init__(self):
		sort_dim = 0
		sort_index = 0
		sort_threshhold = 0
		values = []
		is_leaf = False

	@classmethod
	def sort(cls, constraints, constraint_indices, constraint_nums):
		"""
		Takes a set of constraints, along with a set of indices which indicate which constraints will
		be found on this tree. Constructs a tree for fast lookup of the constraints. Also requires
		a list, constraint_nums, of the floats which occur in the entire set of constraints.
		"""
		self = cls()
		self.values = constraint_indices
		if len(constraint_indices) > 1:
			print("Sorting " + str(len(constraint_indices)) + " constraints.")
			# Determine the best way to split up the given constraints. For a given plane,
			# a constraint could fall 'below' it or 'above' it, or could cross through that
			# plane. Crossing through is bad news since those essentially go both below
			# and above, meaning they're not eliminated. What we need to optimize, then,
			# is the product of how many fall below with how many fall above.
			# TODO - although this should never happen, I should think about how to
			# handle values exactly on a boundary.

			below_counts = [[0 for x in dim_nums] for dim_nums in constraint_nums]
			above_counts = [[0 for x in dim_nums] for dim_nums in constraint_nums]

			# TODO The nested for loops really slow things down if the number of
			# constraint_nums goes up, which it does if I try to make constraints more
			# snug. Would be nice to speed this up so I could comfortably test
			# speedups from sharper constraints.
			for i in constraint_indices:
				for dim in range(15):
					start = list(constraint_nums[dim]).index(constraints[i][dim][0])
					end = list(constraint_nums[dim]).index(constraints[i][dim][1])
					# Choices of split up to and including the start number will
					# place this constraint in the 'above' category.
					for j in range(start + 1):
						above_counts[dim][j] += 1
					# Choices of split >= the end number will place this constraint 'below'
					for j in range(end, len(constraint_nums[dim])):
						below_counts[dim][j] += 1
			# Find maximum and store it in the class
			for dim in range(15):
				for num_index in range(len(constraint_nums[dim])):
					if (below_counts[self.sort_dim][self.sort_index] * above_counts[self.sort_dim][self.sort_index]
							< below_counts[dim][num_index] * above_counts[dim][num_index]):
						self.sort_dim = dim
						self.sort_index = num_index
			self.sort_threshhold = constraint_nums[self.sort_dim][self.sort_index]
			if below_counts[self.sort_dim][self.sort_index] * above_counts[self.sort_dim][self.sort_index] == 0:
				# The constraint planes are not differentiating between these two. So, we just leave them
				# both unsorted and declare ourselves a leaf node.
				# TODO: I could add a test here, which makes sure the constraints are nonoverlapping.
				# TODO: Lookup would be *slightly* faster if I sorted these somehow.
				self.is_leaf = True
			# print(str(below_counts[self.sort_dim][self.sort_index])+" below; "
			#	+str(above_counts[self.sort_dim][self.sort_index])+" above; "
			#	+str(len(constraint_indices) - below_counts[self.sort_dim][self.sort_index]
			#	- above_counts[self.sort_dim][self.sort_index])+" between.")
			if not self.is_leaf:
				# Now we just need to do the actual sorting.
				# For these purposes, constraints which are both below and above get
				# handed to *both* child nodes.
				below = []
				above = []
				for i in constraint_indices:
					start = constraint_nums[self.sort_dim].index(constraints[i][self.sort_dim][0])
					end = constraint_nums[self.sort_dim].index(constraints[i][self.sort_dim][1])
					if end > self.sort_index:
						above.append(i)
					if start < self.sort_index:
						below.append(i)
				self.below = ConstraintTree.sort(constraints, below, constraint_nums)
				self.above = ConstraintTree.sort(constraints, above, constraint_nums)
		else:
			# No need to sort, we're a leaf node
			self.is_leaf = True
		return self

	def find(self, point):
		"""
		Accepts a 3D parallel-space point, as represented by its 15 dot products
		with normal vectors to a rhombic triacontahedron's faces. Returns a small
		list of indices to constraints (as listed in self.all_constraints), one
		of which will contain the point.
		"""
		if self.is_leaf:
			return self.values
		if point[self.sort_dim] < self.sort_threshhold:
			return self.below.find(point)
		# Should be fine to include points equal to threshhold on either side;
		# any constraint straddling the threshhold was sent to both.
		if point[self.sort_dim] >= self.sort_threshhold:
			return self.above.find(point)

	@classmethod
	def load(cls, filename="constraint_tree"):
		fs = File()
		fs.open("res://" + filename, fs.READ)
		new_tree = eval(str(fs.get_line()))
		fs.close()
		return new_tree

	def save(self, filename="constraint_tree"):
		fs = File()
		fs.open("res://" + filename, fs.WRITE)
		fs.store_line(repr(self))
		fs.close()

	@classmethod
	def from_dict(cls, description):
		self = cls()
		self.is_leaf = description["is_leaf"]
		if self.is_leaf:
			self.values = description["values"]
		else:
			self.sort_dim = description["sort_dim"]
			self.sort_index = description["sort_index"]
			self.sort_threshhold = description["sort_threshhold"]
			self.below = description["below"]
			self.above = description["above"]
			self.values = self.below.values + self.above.values

		return self

	def __string__(self):
		return "{} with {} values".f(self.__class__.__name__, len(self.values))

	def __repr__(self):
		# Includes all information except:
		# - self.all_constraints
		# - self.constraint_nu
		if self.is_leaf:
			# For leaf nodes, we include self.values, and of course no children
			# or sort parameters
			return "ConstraintTree.from_dict(" + repr({"values": self.values, "is_leaf": self.is_leaf}) + ")"
		else:
			return "ConstraintTree.from_dict(" + repr({"sort_dim": self.sort_dim, "sort_index": self.sort_index,
													   "sort_threshhold": self.sort_threshhold, "is_leaf": self.is_leaf,
													   "below": self.below, "above": self.above}) + ")"


class Chunk:

	def __init__(self, network, template_index, offset, level=1):
		self.is_topmost = False
		self.network = network
		self.template_index = template_index
		self.blocks = network.all_blocks[template_index][0]
		self.block_values = np.zeros((len(self.blocks)), dtype=np.int)
		# Block centers are neighbors if two dim.s differ by 0.5 and no others differ.
		# TODO I've got to include neighbor information in the templates themselves.
		# TODO It would save about 2.7 milliseconds/chunk, which is 97% of the children generation time.
		repeated = np.repeat([self.blocks], len(self.blocks), axis=0)
		diffs = np.abs(repeated - np.transpose(repeated, (1,0,2)))
		diffs_of_half = diffs == 0.5
		diffs_of_zero = diffs == 0
		self.blocks_neighbor = (np.sum(diffs_of_half, axis=2) == 2)*(np.sum(diffs_of_zero, axis=2) == 4)
		#self.blocks_neighbor = np.zeros((len(self.blocks),len(self.blocks)))
		self.offset = offset
		self.level = level
		self.parent = None
		self.children = None

	def get_parent(self):
		if self.parent is None:
			parents = self.network.generate_parents(self.template_index,self.offset,self.level)
			if len(parents) > 1:
				raise Exception("Got more than one parents for a chunk; a way to handle this is currently not implemented.")
			self.parent = Chunk(self.network, parents[0][0], parents[0][1], self.level+1)
			if self.is_topmost:
				print("Adding new top chunk of level "+str(self.level+1))
				self.is_topmost = False
				self.parent.is_topmost = True
				self.network.highest_chunk = self.parent
		return self.parent

	def get_children(self):
		if self.children is None:
			children = self.network.generate_children(self.template_index,self.offset,self.level)
			self.children = [Chunk(self.network, index, offset, self.level-1) for index, offset in children]
			for child in self.children:
				child.parent = self
			if self.level == 2:
				# Very basic terrain generation
				for child in self.children:
					# We'll use a combination of parallel-space positions to determine heightmap.
					child.heightmap = 10*(self.get_parent().offset.dot(self.network.normallel.T)[0]
										+ self.offset.dot(self.network.normallel.T)[1]
										  + child.offset.dot(self.network.normallel.T)[2])
					# Fill in blocks below the chosen height. In a fancier version we'd smooth between neighboring chunks.
					child.block_values = (((child.blocks + child.offset).dot(self.network.normalworld.T)[:,1] < -1)
											* ((child.blocks + child.offset).dot(self.network.normalworld.T)[:,1] > -2.5))#child.heightmap - 20
		return self.children

	def draw_mesh(self, drawp = lambda x: True):
		# TODO Generate mesh in separate thread(s), like the pure-gdscript voxel game demo does.
		st = SurfaceTool()
		if self.level >= 1:
			multiplier = self.network.phi_powers[(self.level-1)*3]
		if self.level < 1:
			multiplier = 1.0/self.network.phi_powers[-((self.level-1)*3)]
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		st.add_color(Color(r.random(), r.random(), r.random()))

		body = StaticBody.new()
		collider = CollisionShape.new()
		body.add_child(collider)
		# TODO If I can write a custom collider shape for the Ammann rhombohedra, there may be ways to make it fast.
		collider.shape = ConcavePolygonShape()
		collider_face_array = PoolVector3Array()

		drew_something = False

		for block_i in range(len(self.blocks)):
			block = self.blocks[block_i] + self.offset
			if self.block_values[block_i] > 0 and drawp(block):
				# Check for exposed faces
				#print(sum(self.blocks_neighbor[block_i]))
				# if sum(self.blocks_neighbor[block_i]) >= 6:
				# 	# Six neighbors lie inside this chunk
				# 	if np.nonzero(self.block_values[np.nonzero(self.blocks_neighbor[block_i])[0]])[0].shape[0] == 6:
				# 		# Skip this block
				# 		print("Skipping a block yay!")
				# 		continue

				drew_something = True

				face_origin = np.floor(block).dot(self.network.worldplane.T) * multiplier
				face_tip = np.ceil(block).dot(self.network.worldplane.T) * multiplier
				dir1, dir2, dir3 = np.eye(6)[np.nonzero(np.ceil(block) - np.floor(block))[0]].dot(
					self.network.worldplane.T) * multiplier
				# Make "right hand rule" apply
				if np.cross(dir1, dir2).dot(dir3) < 0:
					_ = dir1
					dir1 = dir2
					dir2 = _
				corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8 = (
					face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
					face_tip - dir1, face_tip - dir2, face_tip - dir3
				)
				dir1 = Vector3(dir1[0], dir1[1], dir1[2])
				dir2 = Vector3(dir2[0], dir2[1], dir2[2])
				dir3 = Vector3(dir3[0], dir3[1], dir3[2])
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1 + dir2)

				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1 + dir2)
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)

				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2 + dir3)

				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2 + dir3)
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)

				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3 + dir1)

				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3 + dir1)
				st.add_vertex(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)

				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1 - dir2)
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)

				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1 - dir2)

				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2 - dir3)
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)

				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2 - dir3)

				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3 - dir1)
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)

				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)
				st.add_vertex(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3 - dir1)

				# Now create colliders
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1 + dir2)

				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1 + dir2)
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)

				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2)
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2 + dir3)

				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir2 + dir3)
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)

				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3)
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3 + dir1)

				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]))
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir3 + dir1)
				collider_face_array.push_back(Vector3(face_origin[0], face_origin[1], face_origin[2]) + dir1)

				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1 - dir2)
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)

				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1 - dir2)

				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2 - dir3)
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2)

				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir2 - dir3)

				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3 - dir1)
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3)

				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]))
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir1)
				collider_face_array.push_back(Vector3(face_tip[0], face_tip[1], face_tip[2]) - dir3 - dir1)

		# Finalize mesh for the chunk

		if drew_something:
			st.generate_normals()
			new_mesh = ArrayMesh()
			new_mi = MeshInstance.new()
			new_mi.mesh = new_mesh
			st.commit(new_mesh)
			new_mesh.surface_set_material(new_mesh.get_surface_count() - 1, COLOR)
			self.network.add_child(new_mi)

			# Finalize collision shape for the chunk
			collider.shape.set_faces(collider_face_array)
			body.collision_layer = 0xFFFFF
			body.collision_mask = 0xFFFFF
			new_mi.add_child(body)
		else:
			collider.free()
			body.free()

class GoldenField:
	phi = 1.61803398874989484820458683

	def __init__(self, values):
		self.ndarray = np.array(values, dtype=np.int16)
		if self.ndarray.shape[-1] != 2:
			raise Exception("Not a valid golden field array; last axis must be of size 2.")

	def __repr__(self):
		return f"{self.__class__.__name__}({list(self.ndarray)})"

	def __array__(self, dtype=None):
		return self.ndarray[..., 0] + self.phi * self.ndarray[..., 1]

	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		if method == '__call__':
			# Check if all integer
			all_integer = True
			for input in inputs:
				if not isinstance(input, numbers.Integral):
					if isinstance(input, np.ndarray):
						if not (input.dtype.kind in ['u', 'i']):
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
						returnval[..., 0] = returnval[..., 0] + input
				return self.__class__(returnval)
			elif ufunc == np.multiply:
				returnval = self.ndarray.copy()
				for input in inputs:
					intpart = np.zeros(self.ndarray[..., 0].shape)
					phipart = np.zeros(self.ndarray[..., 0].shape)
					if isinstance(input, self.__class__):
						intpart = returnval[..., 0] * input.ndarray[..., 0]
						phipart = returnval[..., 0] * input.ndarray[..., 1] + returnval[..., 1] * input.ndarray[..., 0]
						intpart = intpart + returnval[..., 1] * input.ndarray[..., 1]
						phipart = phipart + returnval[..., 1] * input.ndarray[..., 1]
					elif isinstance(input, np.ndarray):
						# Multiply both parts by the array
						intpart = returnval[..., 0] * input
						phipart = returnval[..., 1] * input
					elif isinstance(input, numbers.Integral):
						intpart = returnval[..., 0] * input
						phipart = returnval[..., 1] * input
					else:
						return NotImplemented
					returnval[..., 0] = intpart
					returnval[..., 1] = phipart
				return self.__class__(returnval)
			else:
				return NotImplemented
		else:
			return NotImplemented
