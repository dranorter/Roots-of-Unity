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
	phi_powers = np.power(np.array([phi]*100), np.arange(100))
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

	rhomb_indices = np.array(
		[0, 2, 7, 0, 7, 3, 0, 3, 5, 0, 5, 4, 0, 4, 6, 0, 6, 2, 1, 4, 5, 1, 6, 4, 1, 2, 6, 1, 7, 2, 1, 3, 7, 1, 5, 3])

	player_pos = np.zeros((3,))
	player_guess = None
	block_highlight = ImmediateGeometry.new()

	def phipow(self, n):
		if n >= 0:
			return self.phi_powers[n]
		if n < 0:
			return 1.0/self.phi_powers[-n]

	def custom_pow(self, matrix, power, safe_amount=7):
		"""
		A safe method for taking positive and negative powers of integer matrices without getting floating point errors.
		When the power is negative, a positive power is taken first before inverting (the opposite of numpy). When the
		power's absolute value is outside the safe_amount,
		:param matrix: The (square) matrix to be exponentiated. Will be converted to numpy array, dtype=int.
		:param power: The exponent, which should be integer.
		:param safe_amount: Amount to exponentiate at once. Default was obtained via testing; 8 seems fine but went with
		 a default of 7. Higher values, perhaps 10 to 12, are fine if exponents will be positive. Higher values will
		 make performance better if very large exponents are being used.
		:return: A numpy matrix, the power of the input matrix.
		"""
		exponent_increment = safe_amount
		remaining_levels = power
		product = np.eye(len(matrix), dtype=int)
		trustworthy_portion = np.array(
			np.round(np.linalg.inv(np.linalg.matrix_power(np.array(matrix, dtype=int), exponent_increment))), dtype=int)
		while remaining_levels < -exponent_increment:
			product = np.array(np.round(product.dot(trustworthy_portion)), dtype=int)
			remaining_levels += exponent_increment
		trustworthy_portion = np.array(
			np.round(np.linalg.matrix_power(np.array(matrix, dtype=int), exponent_increment)), dtype=int)
		while remaining_levels > exponent_increment:
			product = np.array(np.round(product.dot(trustworthy_portion)), dtype=int)
			remaining_levels -= exponent_increment
		remaining_part = np.linalg.matrix_power(np.array(matrix, dtype=int), abs(remaining_levels))
		if remaining_levels < 0:
			remaining_part = np.array(np.round(np.linalg.inv(remaining_part)), dtype=int)
		return np.array(np.round(product.dot(remaining_part)), dtype=int)

	def chunk_center_lookup(self, template_index):
		"""
		Returns the chunk's template's center (as an array of six coordinates) given its index.
		This function is needed because all_chosen_centers stores the centers as strings.
		:param template_index: The index of the chunk template within all_chosen_centers, all_blocks, etc.
		:return: The center of the chunk, as block-level coordinates within the template.
		"""
		# TODO Replace with a pre-calculated lookup table ("all_chosen_centers_live")?
		return self.possible_centers_live[self.possible_centers.index(self.all_chosen_centers[template_index])]

	def convert_chunklayouts(self, filename="res://chunklayouts_perf_14"):
		"""
		Loads a pure-Python repr of the chunk layouts (as generated by
		numpylattice.py), and then saves that as three separate files, one
		of which is in the more compact numpy save format.
		"""
		# TODO Identify all the blocks which are always present,
		# 	then exclude those when saving, so they don't have to be loaded.
		# 	Save them in a separate (and of course human-readable) file, or maybe
		# 	as a first element in the layouts file.
		# TODO Figure out all the rotation matrices mapping between chunk
		# 	orientations, so that I can save 1/10th as many of these templates.
		# 	Also, check if other symmetries apply - individual chunks are
		# 	rotationally symmetrical too.
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
		# TODO Definitely need tests that repeated saving and loading produces identical files. I don't want to change
		#  something here to fix a bug only to find that it's just a particular file that was odd.
		fs = File()
		fs.open("res://" + filename, fs.READ)
		self.all_constraints = eval(str(fs.get_line()))
		# NOTE Chosen centers are stored as strings!
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
			#  constraints, and speed up the constraint search by a bit.
			#  (I'm uncertain now whether it's very much.)
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
							#  to see if we can use its literal value.
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
		# TODO My initial chunk templates, which I was using for months, had a lot of overlap between adjacent
		#  templates. At first, when I had just begun writing any sort of visualization of the chunk hierarchy,
		#  I was frequently getting two parents for a given chunk; however, more recently this has not happened once
		#  in hundreds of tries. Technically this should be happening whenever there are chunks that overlap - so
		#  what's happening? Why is this not finding every parent in inside blocks?
		chosen_center = self.chunk_center_lookup(template_index)
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

		# We need to calculate the chunk axes inverse carefully. When Numpy takes a matrix inverse, it makes it
		# out of floats, and floating point error turns it into junk at around the 13th or 14th power (IE, chunks of
		# level 13 or 14).
		# What we'd like is this simple line:
		#chunk_axes_inv = np.round(np.linalg.matrix_power(np.array(self.deflation_face_axes,dtype=int).T, -level))

		# But we'll break it into smaller steps:
		# TODO The below method creates a hang of about a second if chunk level gets near 1 million. If I pre-calculate
		#  just a few powers of deflation_face_axes, e.g. 10, 100, 1000, 10,000, etc, these very rare hiccups could
		#  be avoided.
		def_face_matrix = np.array(self.deflation_face_axes,dtype=int).T
		# Testing has shown problems cropping up when below increment is equal to 9. Using 7 for safety margin.
		chunk_axes_inv = self.custom_pow(def_face_matrix, -level, safe_amount=7)
		level_sign = pow(-1,level)

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
		#  - I found it was important to do the dot products before the addition (in calculating safer_seed), since
		#    lowered_offset can be a very large number and the seed is a small one; the dot product brings them
		#    into something like a shared range. Especially important since their difference, safer_seed, is supposed
		#    to be quite tiny.
		#  - I found that much floating point error was being created by the matrix power, used to calculate chunk_axes_inv.
		#    So I wrote my own matrix power function, which has been somewhat carefully tested.
		#  - The chunk axis transform is simply a scaling up in the worldplane and simultaneous scaling down in "parallel
		#    space". Maybe it can be calculated with less problems in those terms. Instead of multiplying offset by a matrix
		#    power, multiply it by a power of phi and then add it to the seed, then take the squarallel of that.

		# "offset" starts out at the level "level", and the seed starts out at
		# level 1, so we want to apply deflation_face_axes level-1 times.
		#lowered_offset = np.linalg.matrix_power(np.array(self.deflation_face_axes,dtype=int), (level - 1)).dot(np.round(offset))
		lowered_offset = np.array(np.round(offset),dtype=int).dot(self.custom_pow(def_face_matrix,level - 1))
		# Can we get less floating point problems by just multiplying by a power of phi?
		#lowered_offset = level_sign*np.array(offset,dtype=int)*self.phipow(-3*(level-1))
		# To reduce floating point error, we separately dot "seed" and "lowered_offset" with squarallel, then subtract.
		# TODO Actually, why take the squarallel?
		# But, "seed" doesn't actually need projected, it's already been.
		# (Encountered pretty big floating point errors around 1e-9, or, level-13 chunks.)
		safer_seed = self.seed - lowered_offset.dot(self.squarallel)
		#chunk_axes_inv_seed = chunk_axes_inv.dot(safer_seed)
		# Can we get less floating point problems by just multiplying by a power of phi?
		chunk_axes_inv_seed_2 = level_sign*safer_seed*self.phipow(3*level)
		chunk_axes_inv_seed = level_sign*self.phipow(3*level)*self.seed + self.phipow(3)*np.array(offset,dtype=int)#.dot(self.squarallel)
		print("Calculating seed from safe point " + str(lowered_offset))
		print(str(lowered_offset.dot(self.squarallel)))
		print(str(-level_sign*np.array(offset,dtype=int).dot(self.squarallel)*self.phipow(-3*(level-1))))
		print("Translated seed (should be < " + str(round(math.pow(1 / 4.7, level), 2)) + "):" + str(safer_seed))
		print("Translated Seed: "+str(self.seed + level_sign*np.array(offset,dtype=int).dot(self.squarallel)*self.phipow(-3*(level-1))))
		print("Rescaled seed: " + str(chunk_axes_inv_seed))
		print("Rescaled seed: " + str(chunk_axes_inv_seed_2))
		#print("Directly rescaled: "+ str(-safer_seed*self.phipow(3*level)))

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

		#TODO Low priority, but: should the below be rewritten to use the constraint tree?
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
			#translated_seeds = (chunk_axes_inv_seed - proposed_origins).dot(self.normallel.T)
			# TODO Maybe chunk_axes_inv_seed.dot(self.normallel.T) could be calculated more directly.
			translated_seeds = chunk_axes_inv_seed.dot(self.normallel.T) - proposed_origins.dot(self.normallel.T)
			# Note, these tests used to be strict, but I added an epsilor
			a_hit = np.all([np.all(self.twoface_normals.dot(translated_seeds.T).T - constraint[:, 0] >= -1e-14, axis=1),
							np.all(self.twoface_normals.dot(translated_seeds.T).T - constraint[:, 1] <= 1e-14, axis=1)], axis=0)
			if np.any(a_hit):
				print("We're inside template #" + str(i) + ", at offset" + str(
					block_offsets[np.array(np.nonzero(a_hit))[0, 0]]))
				inside_hits = [(i, proposed_origins[j] + chunk_as_block_origin) for j in np.nonzero(a_hit)[0]]
				break
		# Checking outside blocks takes a lot longer, we only want to  do it if we need to
		if len(inside_hits) == 0:
			print("No inside hits; checking outside")
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
					# TODO I think this is happening because of some glitch with my testpoint scheme for canonical
					#  block assignment. Need to print a lot more test info here.
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
		#  are genuinely all we need. It does look fine, though.
		# (In fact if anything, I still have overlap!)
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

		chunkscaled_positions = np.array(self.deflation_face_axes,dtype=int).dot(np.transpose(unique_points)).T
		chunkscaled_offset = np.array(self.deflation_face_axes,dtype=int).dot(offset)
		# Dot product and then subtract *might* reduce overall floating point error.
		chunk_seeds = current_level_seed.dot(self.squarallel) - chunkscaled_positions.dot(self.squarallel)

		# TODO Occasionally the below line leads to an error, with no valid chunks in one of the find_satisfied calls. Why?
		#  May be this only happens when generating children of fairly high-level chunk (say, 13 or so). There's floating
		#  point error right now plaguing those chunks' offset values.
		found_satisfied = [self.find_satisfied(seed) for seed in chunk_seeds]
		# TODO Nested for loops; optimize?
		# for seed_i in range(len(chunk_seeds)):
		for chunk_i in range(len(chunks)):
			# Index into the computations
			u_i = unique_lookup[chunk_i]
			# children.append((self.find_satisfied(chunk_seeds[seed_i]),chunkscaled_positions[seed_i]+chunkscaled_offset))

			location = chunkscaled_positions[u_i] + chunkscaled_offset
			for template_index in found_satisfied[u_i]:
				template_center = self.chunk_center_lookup(template_index)
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
		#  here for whether the single-hit cases still pass the self.satisfies test.
		# The tree ought to get us 1 to 3 possibilities to sort through.
		hits = self.constraint_tree.find(seed_15)
		# Hits can be spurious if the seed doesn't happen to be separated
		# from the constraint region by any of the planes which the tree
		# made use of. (If the seed is outside of a constraint, there's
		# always some plane the tree *could have* tested in order to see this.
		# But when several constraint regions are separated by no single plane,
		# the tree won't test every plane.)
		# TODO If I ever wanted to remove this call to self.satisfies, I could
		#  have the leaf nodes of the tree do some final tests. In principle
		#  the leaf nodes could know which planes haven't been tested yet and
		#  cross their regions, and then use those to determine which
		#  single template to return. Almost equivalent, but easier on the
		#  lookup tree code, would be cutting up all the constraint regions
		#  so that the overlapping regions are their own separate entries on
		#  the constraints list. I'd have a choice of exactly how to cut
		#  everything up yet leave the regions convex.
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
		raise Exception("No valid hits out of " + str(len(hits)) + " hits."
						+ "\n Distances from validity:\n"
						+ str([self.satisfies_by(seedvalue, self.all_constraints[i]) for i in hits]))

	def update_player_pos(self, pos, transform):
		trans_inv = transform.basis.inverse()
		rotation = np.array([[transform.basis.x.x,transform.basis.x.y,transform.basis.x.z],
							 [transform.basis.y.x,transform.basis.y.y,transform.basis.y.z],
							 [transform.basis.z.x,transform.basis.z.y,transform.basis.z.z]])
		rotation_inv = np.array([[trans_inv.x.x, trans_inv.x.y, trans_inv.x.z],
							 [trans_inv.y.x, trans_inv.y.y, trans_inv.y.z],
							 [trans_inv.z.x, trans_inv.z.y, trans_inv.z.z]])
		position = np.array([pos.x,pos.y,pos.z])
		translation = np.array([transform.origin.x, transform.origin.y, transform.origin.z])
		self.player_pos = position - np.array([2.5, 51, -66.7]) #- translation

	def chunk_at_location(self, target, target_level=0, generate=False, verbose=False):
		"""
		Takes a 3D coordinate and returns the smallest list of already-generated chunks guaranteed to contain that coordinate.
		The search proceeds from the top-level chunk down; if you want to include a guess, call the Chunk.chunk_at_location
		function on your best-guess chunk.
		:param target:
		The coordinates to search for.
		:param target_level:
		The level of chunk desired, with blocks being target_level 0. If chunks of the desired level have not yet been
		generated, by default this will return the closest available level.
		:param generate:
		Set generate=True to force generation of the desired level of chunks. Note, this can either force generation
		down to target_level, or up to target_level, or some combination; for example we can request a chunk of level 10
		(about the scale of Earth's moon), and place it a cauple hundred million blocks away from us (about the distance
		to the moon), and this forces the generation of a highest-level chunk of approximately level 14, subdivided just
		enough to produce the requested level 10 chunk.
		:return: A list of chunks. If target_level < 0, these will be placeholder chunks which represent blocks or sub-blocks.
		"""
		if self.player_guess is None:
			return self.highest_chunk.chunk_at_location(target, target_level, generate, verbose)
		else:
			return self.player_guess.chunk_at_location(target, target_level, generate, verbose)

	def _process(self, delta):
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
		if r.random() < 0.95:
			return
		# For now, we use a loading radius, which cautiously expands.

		self.block_highlight.clear()
		#self.block_highlight.set_color(Color(.1,.1,.1))
		# Is the player contained in a fully generated chunk?
		closest_chunks = self.chunk_at_location(self.player_pos)
		if len(closest_chunks) != 1 or closest_chunks[0].level != 0:
			print(str(len(closest_chunks))+" results from search for player. Level: "+str(min([c.level for c in closest_chunks]+[0])))
		for closest_chunk in closest_chunks:
			self.player_guess = closest_chunk
			#closest_chunk.highlight_block()
			#closest_chunk.get_parent().highlight_block()
			#closest_chunk.get_parent().get_parent().highlight_block()
			#self.block_highlight.show()
			if not closest_chunk.drawn:
				closest_chunk.draw_mesh()
			if closest_chunk.level > 1 or (closest_chunk.level == 1 and not closest_chunk.all_children_generated):
				# If not, generating that chunk is enough for this frame; do it and return.
				# But first, reduce the loading_radius since it obviously failed.
				print("Search found chunk of level "+str(closest_chunk.level)+". Children generated? "+str(closest_chunk.all_children_generated))
				# TODO It would be nice for testing to be able to discover whether anything new generates when generate=True.
				#print("You need some ground to stand on! Trying to draw it in")
				new_ground = closest_chunk.chunk_at_location(self.player_pos, target_level=0, generate=True)
				if closest_chunk in new_ground:
					print("Got same chunk back. Children generated? "+str(closest_chunk.all_children_generated))
					print("Did the search go exactly the same way? "+str(new_ground == closest_chunks))
				return
		if len(closest_chunks) == 0:
			print("\n\tBroader chunk tree needed! Generating...\t")
			closest_chunks = self.chunk_at_location(self.player_pos, target_level=0, generate=True, verbose=True)
			print("\n\tBroadening resulted in "+str(len(closest_chunks))+" search results.\n")
			return

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

		self.load_templates_npy("templates_test_point")
		print("Done loading " + str(len(self.all_constraints)) + " templates")
		print(time.perf_counter() - starttime)

		# Lookup table for Chunk.rhomb_contains_point
		self.axes_matrix_lookup = {center:[np.linalg.inv(
			self.worldplane.T[np.nonzero(
				self.possible_centers_live[self.possible_centers.index(center)]
				- np.floor(self.possible_centers_live[self.possible_centers.index(center)]) - 0.5)[0]]
			* self.phipow(3 * level)) for level in range(10)] for center in self.possible_centers}

		print("Created axes matrix lookup table")
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

		print("Removing duplicate blocks...")

		# Checking for overlap with neighbors!
		# This is doubtless slow, but I have to do it for now.
		# OK, after the below testing, here are the facts:
		# - Sadly, block centers do sometimes fall exactly on chunk boundaries. This is something I'd been hoping
		#   was not true since it makes block ownership very ambiguous.
		# - I tried testing the block's origin to "break the tie", and found that it, too, was sometimes on the
		#   boundary, so no decision could be made.
		# - Additionally, it should be noted that sometimes the origin is the same distance out from each chunk,
		#   but that distance is about 0.236. Surely, this means the origin would fall in some other neighbor; but
		#   I haven't verified that.
		# - I tried introducing a 'test point', specifically (1, 0, 0) in the block's three axes, but this too
		#   turned out to fall on the boundary some of the time, so didn't break all ties.
		# - Introducing a second test point, (0, 1, 0), appears to break all ties.
		def rhomb_contains_point(point, template_index, self = self):
			level = 1
			axes_matrix = self.axes_matrix_lookup[self.all_chosen_centers[template_index]][level]
			target_coords_in_chunk_axes = np.array(point).dot(axes_matrix) - 0.5
			# To measure distance from closest face-plane, we take the minimum of the absolute value.
			dist_from_center = np.abs(target_coords_in_chunk_axes)
			return np.min(0.5 - dist_from_center)

		for t_i in []:#range(len(self.all_blocks)):
			to_skip = []
			blocks = self.all_blocks[t_i][0]
			for block_i in range(len(self.all_blocks[t_i][0])):
				center = blocks[block_i]
				origin = np.floor(blocks[block_i])
				axes = np.nonzero(blocks[block_i] - np.floor(blocks[block_i]))[0]
				testpoint = (np.floor(blocks[block_i])
					+ 0.8 * np.eye(6)[axes[0]]
					+ 0.1 * np.eye(6)[axes[1]]
					+ 0.1 * np.eye(6)[axes[2]])
				testpoint2 = (
					 np.floor(blocks[block_i])
					+ 0.1 * np.eye(6)[axes[0]]
					+ 0.8 * np.eye(6)[axes[1]]
					+ 0.1 * np.eye(6)[axes[2]])
				testpoint3 = (
					np.floor(blocks[block_i])
					+ 0.1 * np.eye(6)[axes[0]]
					+ 0.1 * np.eye(6)[axes[1]]
					+ 0.8 * np.eye(6)[axes[2]])
				testpoint4 = (
					np.floor(blocks[block_i])
					+ 0.8 * np.eye(6)[axes[0]]
					+ 0.8 * np.eye(6)[axes[1]]
					+ 0.1 * np.eye(6)[axes[2]])
				dist_center = rhomb_contains_point(center.dot(self.worldplane.T), t_i)
				dist_origin = rhomb_contains_point(origin.dot(self.worldplane.T), t_i)
				dist_testpoint = rhomb_contains_point(testpoint.dot(self.worldplane.T), t_i)
				dist_testpoint2 = rhomb_contains_point(testpoint2.dot(self.worldplane.T), t_i)
				dist_testpoint3 = rhomb_contains_point(testpoint3.dot(self.worldplane.T), t_i)
				dist_testpoint4 = rhomb_contains_point(testpoint4.dot(self.worldplane.T), t_i)
				#print(dist_center)
				if abs(dist_center) < 1e-15:
					test_value = [0]
					# Block most likely picked up by a neighboring chunk in its template. Decide who the block belongs to.
					# The idea is, the block's origin is canonical (ie, the different templates will agree on it). Whichever
					# chunk contains the origin can take ownership of the point.
					#print("Got to 1st test")
					if abs(dist_origin) < 1e-15:
						#print("Got to 1st test")
						# Origin doesn't work as tiebreak, go to first test point.
						if abs(dist_testpoint) < 1e-15:
							# First testpoint doesn't work as tiebreak; go to second testpoint.
							#print("Got to 2nd test")
							if abs(dist_testpoint2) < 1e-15:
								# Second testpoint doesn't work either!!
								#print("Got to 3rd test")
								if abs(dist_testpoint3) < 1e-15:
									#print("Got to 4th test")
									if abs(dist_testpoint4) < 1e-15:
										raise Exception(
											"Unable to assign block to a specific chunk. Please add 5th test point.")
									else:
										test_value[0] = dist_testpoint4
								else:
									test_value[0] = dist_testpoint3
							else:
								test_value[0] = dist_testpoint2
						else:
							test_value[0] = dist_testpoint
					else:
						test_value[0] = dist_origin
					# If test_value > 0, the block stays.
					if test_value[0] < 0:
						print(test_value[0])
						to_skip.append(block_i)
			print("Removing "+str(len(to_skip))+" blocks out of "+str(len(self.all_blocks[t_i][0])))
			# We just shift it over to the "outside" list.
			self.all_blocks[t_i] = ([self.all_blocks[t_i][0][block_i] for block_i in range(len(self.all_blocks[t_i][0]))
									   if block_i not in to_skip], np.concatenate(
									[self.all_blocks[t_i][1], [self.all_blocks[t_i][0][block_i] for block_i in range(len(self.all_blocks[t_i][0]))
									   if block_i in to_skip]]))

		print("Removed dupes in templates.")
		print(time.perf_counter() - starttime)
		# Now that constraints have considerably less points, we probably need considerably less of them.
		#self.test_templates(True)
		#self.simplify_constraints()
		#self.simplify_constraints()
		#self.simplify_constraints()
		#self.save_templates_npy("templates_test_point")

		# TODO Could still try optimizing generation by pre-generating higher-level chunks --
		#  e.g. making templates for supersuperchunks. But at this point it feels like that
		#  would be premature optimization.

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
		chosen_center = self.chunk_center_lookup(chunk_num)
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
		#  But, should do a more rigorous test, ideally verifying back in
		#  numpylattice.py that this is inevitable.)

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

		#for c in children:
		#	c.draw_mesh()
		print("Done calling draw. Time:")
		print(time.perf_counter() - starttime)
		# if len(children) > 0:
		# 	st.generate_normals()
		# 	st.commit(self.mesh)
		# 	self.mesh.surface_set_material(self.mesh.get_surface_count() - 1, COLOR)
		# print("Mesh updated. Time:")
		# print(time.perf_counter() - starttime)

	def measure_chunk_overlap(self):
		"""
		Checks each chunk template to see how far blocks come into the chunk, which aren't owned by that chunk.
		Distance is a taxi cab distance using the chunk's own axes, as implemented by rhomb_contains_point. I wrote
		this to get a safety margin for the safely_contains_point function. However, it serves as good testing code
		since the result should theoretically not exceed 0.19794.
		:return:
		"""
		# First run got 0.25650393107963126, which is higher than what was in principle expected (0.19793839129906832).
		# Second run, excluded "exterior" blocks which were actually in the interior. Got
		# 0.11524410338055335, which is inside the theoretical bounds. Time to make the search more exhaustive.
		# Third run, back to 0.25650393107963115, and this value was encountered frequently. Ah, I've found a mistake
		# in my distance function. Fourth run, 0.14199511391282338, closer to theoretical and a believable final answer.
		# Fifth run, added just one more point (origin, tip, and one combo point); result unchanged.
		# Sixth run, changing strategy. We ought to get the same measurement using inside blocks as we do with outside,
		# we just have to measure how far they poke *out* instead of *in*. Result: 0.14199511391282327. OK, so we
		# should be able to test more vertices now; iterating through inside blocks saves time. (However, I'm pretty
		# confident in the result.)
		highest = 0
		for template_i in range(len(self.all_blocks)):
			new_chunk = Chunk(self,template_i,[0,0,0,0,0,0],1)
			for block_j in range(len(self.all_blocks[template_i][0])):
				center_3D = self.all_blocks[template_i][0][block_j].dot(self.normalworld.T)
				# If the center is inside the chunk:
				if new_chunk.rhomb_contains_point(center_3D) >= 0:#1e-15:
					origin_3D = np.floor(self.all_blocks[template_i][0][block_j]).dot(self.normalworld.T)
					tip_3D =  np.ceil(self.all_blocks[template_i][0][block_j]).dot(self.normalworld.T)
					for corner_3D in [origin_3D, tip_3D, np.concatenate([origin_3D[:3],tip_3D[3:]])]:
						distance_to_corner = new_chunk.rhomb_contains_point(corner_3D)
						if distance_to_corner < 0:
							print(-distance_to_corner)
							print(highest)
						if -distance_to_corner > highest:
							highest = -distance_to_corner
		return highest

	def test_templates(self, remove_dupes = False):
		starttime = time.perf_counter()
		possible_blocks = set()
		for blocklayout in self.all_blocks:
			# combined = np.concatenate([blocklayout[0], blocklayout[1]])
			# Right now all we care about is the inner blocks
			combined = blocklayout[0]
			combined = combined * 2
			combined = np.array(np.round(combined), dtype=np.int64)
			combined = [repr(list(x)) for x in combined]
			for block in combined:
				possible_blocks.add(block)
		print("Set up possible blocks list. " + str(len(possible_blocks)) + " occur.")  # 4042
		print(time.perf_counter() - starttime)

		possible_layouts = []
		blocklist = [eval(x) for x in possible_blocks]
		novel_indices = []
		for blocklayout_i in range(len(self.all_blocks)):
			blocklayout = self.all_blocks[blocklayout_i]
			#combined = np.concatenate([blocklayout[0], blocklayout[1]])
			# Right now all we care about is the inner blocks
			combined = blocklayout[0]
			combined = np.round(combined * 2)
			layout = np.any(
				np.all(np.repeat(blocklist, len(combined), axis=0).reshape(-1, len(combined), 6) - combined == 0,
					   axis=2), axis=1)
			novel = True
			for poss in possible_layouts:
				if np.all(layout == poss):
					novel = False
					#debugging.breakpoint()
			if novel:
				possible_layouts.append(layout)
				novel_indices.append(blocklayout_i)
		print("Number of unique layouts according to more careful calculation:")
		print(len(possible_layouts))
		print(time.perf_counter() - starttime)
		if remove_dupes:
			self.all_blocks = [self.all_blocks[i] for i in range(len(self.all_blocks)) if i in novel_indices]
			self.all_constraints = [self.all_constraints[i] for i in range(len(self.all_constraints)) if i in novel_indices]
			self.all_chosen_centers = [self.all_chosen_centers[i] for i in range(len(self.all_chosen_centers)) if i in novel_indices]

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
			print(self.chunk_center_lookup(i) - translation)
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
		self.offset = np.array(offset)
		self.level = level
		# TODO Easy to forget whether self.blocks holds straight template coords or coords with offset. Make the
		#  mistake less easily made. Which *should* it be?
		self.blocks = network.all_blocks[template_index][0]
		self.block_values = np.zeros((len(self.blocks)), dtype=np.int)
		# Block centers are neighbors if two dim.s differ by 0.5 and no others differ.
		# TODO I've got to include neighbor information in the templates themselves.
		#  It would save about 2.7 milliseconds/chunk, which is 97% of the children generation time.
		repeated = np.repeat([self.blocks], len(self.blocks), axis=0)
		diffs = np.abs(repeated - np.transpose(repeated, (1,0,2)))
		diffs_of_half = diffs == 0.5
		diffs_of_zero = diffs == 0
		self.blocks_neighbor = (np.sum(diffs_of_half, axis=2) == 2)*(np.sum(diffs_of_zero, axis=2) == 4)
		#self.blocks_neighbor = np.zeros((len(self.blocks),len(self.blocks)))
		self.parent = None
		self.children = [None]*len(self.blocks)
		self.all_children_generated = False
		if level == 0: self.all_children_generated = True
		self.drawn = False
		self.mesh = None
		self.collision_mesh = None

	def get_parent(self):
		if self.parent is None:
			parents = self.network.generate_parents(self.template_index, self.offset, self.level)#self.level
			if len(parents) > 1:
				raise Exception("Got more than one parents for a chunk; a way to handle this is currently not implemented.")
			self.parent = Chunk(self.network, parents[0][0], parents[0][1], self.level+1)
			# We have to set ourselves as a child, in the appropriate place
			# TODO Below line took way to long to get right, for two reasons.
			#  - It's easy to forget Chunk.blocks just stores default template positions, not adding in the chunk's
			#    own offset.
			#  - I should still make more utility functions; e.g., something for getting block axes and converting
			#    arbitrary points between chunk level coords.
			chunk_as_block_center = np.round(
				(np.linalg.inv(self.network.deflation_face_axes)).dot(
					self.network.chunk_center_lookup(self.template_index)
					+ self.offset - self.parent.get_offset(self.level - 1)) * 2) / 2.0
			for i in range(len(self.parent.blocks)):
				if np.all(chunk_as_block_center == self.parent.blocks[i]):
					self.parent.children[i] = self
					self.index_in_parent = i
					break
			#TODO Convert into a test
			print("Did block end up parent's child? "+str(self in self.parent.children))
			if not self in self.parent.children:
				print("Our offset:")
				print(self.get_offset(self.level) - self.parent.get_offset(self.level)
					  + (1 - self.network.chunk_center_lookup(self.template_index)
					  - np.round(self.network.chunk_center_lookup(self.template_index))))
				print(np.round(
					(np.linalg.inv(self.network.deflation_face_axes)).dot(self.network.chunk_center_lookup(self.template_index) + self.offset) * 2) / 2.0)
				print(np.round(
					(np.linalg.inv(self.network.deflation_face_axes)).dot(self.network.chunk_center_lookup(self.template_index) ) * 2) / 2.0)
				print("Estranged sibling offsets:")
				for i in range(len(self.parent.blocks)):
					print(self.parent.blocks[i])
				# Probably, an "outside block" hit got return from generate_parents. We can try to recover by finding
				# a grandparent which can contain both.
				if self.is_topmost:
					print("Adding new top chunk of level " + str(self.level + 1))
					self.is_topmost = False
					self.parent.is_topmost = True
					self.network.highest_chunk = self.parent
				grandparent = self.parent.get_parent()
				old_parent = self.parent
				avunculars = grandparent.get_children()
				found_parent = False
				for a in avunculars:
					#TODO There's always a chance a grandparent won't be enough and we'd have to go to great
					# grandparent etc. I could (a) code a recursive solution, assuring myself it would happen rarely,
					# (b) use neighbor information to see ahead of time if that's what will happen, generating a
					# more minimal set of parents, or (c) figure out why all of this is happening in the first place
					# and prevent it.
					chunk_as_block_center = np.round((np.linalg.inv(self.network.deflation_face_axes)).dot(
							self.network.chunk_center_lookup(self.template_index)
							+ self.offset - a.get_offset(self.level - 1)) * 2) / 2.0
					for i in range(len(a.blocks)):
						if np.all(chunk_as_block_center == a.blocks[i]):
							print("Found our real parent!")
							self.parent = a
							self.parent.children[i] = self
							self.index_in_parent = i
							break
					if found_parent:
						break
				if not found_parent:
					print("Warning: Unable to place chunk amongst parent's siblings.")
			if self.is_topmost:
				print("Adding new top chunk of level "+str(self.level+1))
				self.is_topmost = False
				self.parent.is_topmost = True
				self.network.highest_chunk = self.parent
		return self.parent

	def get_children(self):
		"""
		Returns all direct children, generating any which are not yet generated.
		:return: A list of all children as chunks, in the same order as in the chunk template.
		"""
		if not self.all_children_generated:
			children = self.network.generate_children(self.template_index, self.offset, self.level)
			if len(children) != len(self.blocks):
				print("MISMATCH BETWEEN CHILDREN AND TEMPLATES")
				print("Children: "+str(len(self.blocks)))
				print("Templates:"+str(len(children)))
			for i in range(len(self.children)):
				if self.children[i] is None:
					self.children[i] = Chunk(self.network, children[i][0], children[i][1], self.level-1)
			self.all_children_generated = True
			for child_i in range(len(self.children)):
				self.children[child_i].parent = self
				self.children[child_i].index_in_parent = child_i
			if self.level == 2:
				# Very basic terrain generation
				for child in self.children:
					# We'll use a combination of parallel-space positions to determine heightmap.
					child.heightmap = 10*(self.get_parent().offset.dot(self.network.normallel.T)[0]
										+ self.offset.dot(self.network.normallel.T)[1]
										  + child.offset.dot(self.network.normallel.T)[2])
					# Fill in blocks below the chosen height. In a fancier version we'd smooth between neighboring chunks.
					child.block_values = (((child.blocks + child.offset).dot(self.network.normalworld.T)[:, 1] < -1)
											* ((child.blocks + child.offset).dot(self.network.normalworld.T)[:, 1] > -2.5))#child.heightmap - 20
					# Use below to completely fill chunks
					#if np.any(child.block_values):
					#	child.block_values[:] = True
			# If this is a level 1 chunk, we need to draw the blocks in.
			if self.level == 2:
				#print("Ah, let's draw these in")
				for child in self.children:
					if not child.drawn:
						child.draw_mesh()
			if self.level == 1 and not self.drawn:
				self.draw_mesh()
			# TODO Don't just scrap below code; alter it to detect and store block neighbors across chunk boundaries.
			if self.parent is not None:
				neighbor_indices = np.nonzero(self.parent.blocks_neighbor[self.index_in_parent])[0]
				for neighbor_i in neighbor_indices:
					if self.parent.children[neighbor_i] is not None:
						neighbor = self.parent.children[neighbor_i]
						if neighbor.template_index == self.template_index:
							if np.all(neighbor.offset == self.offset):
								print("Warning: Two children of a chunk are identical.")
						for neighbor_child_i in range(len(neighbor.children)):
							neighbor_child = neighbor.children[neighbor_child_i]
							if neighbor_child is not None:
								for child_i in range(len(self.children)):
									child = self.children[child_i]
									if child is not None:
										if child.template_index == neighbor_child.template_index:
											if np.all(child.offset == neighbor_child.offset):
												print("Warning: Duplicate chunk found. Mending network... This will create a loop...")
												#print("Guilty templates: ")
												#print(self.template_index)
												#print(neighbor.template_index)
												#print("Containment values:")
												lowered_center = self.network.custom_pow(np.array(self.network.deflation_face_axes),
																						self.level - 1).dot(self.get_offset(level=self.level-1)+self.blocks[child_i])
												lowered_origin = self.network.custom_pow(
													np.array(self.network.deflation_face_axes),
													self.level - 1).dot(
													child.get_offset(level=self.level - 1))
												block_axes = np.nonzero(self.blocks[child_i] - np.floor(self.blocks[child_i]))[0]
												lowered_testpoint = self.network.custom_pow(
													np.array(self.network.deflation_face_axes),
													self.level - 1).dot(
													child.get_offset(level=self.level - 1)+1.0*np.eye(6)[block_axes[0]]+0.0*np.eye(6)[block_axes[1]]+0.0*np.eye(6)[block_axes[2]])
												lowered_testpoint2 = self.network.custom_pow(
													np.array(self.network.deflation_face_axes),
													self.level - 1).dot(
													child.get_offset(level=self.level - 1) + 0.0 * np.eye(6)[
														block_axes[0]] + 1.0 * np.eye(6)[block_axes[1]] + 0.0 *
													np.eye(6)[block_axes[2]])
												self_dist_center = self.rhomb_contains_point(lowered_center.dot(self.network.worldplane.T))
												neighbor_dist_center = neighbor.rhomb_contains_point(lowered_center.dot(self.network.worldplane.T))
												self_dist_origin = self.rhomb_contains_point(
													lowered_origin.dot(self.network.worldplane.T))
												neighbor_dist_origin = neighbor.rhomb_contains_point(
													lowered_origin.dot(self.network.worldplane.T))
												self_dist_testpoint = self.rhomb_contains_point(lowered_testpoint.dot(self.network.worldplane.T))
												neighbor_dist_testpoint = neighbor.rhomb_contains_point(
													lowered_testpoint.dot(self.network.worldplane.T))
												self_dist_testpoint2 = self.rhomb_contains_point(
													lowered_testpoint2.dot(self.network.worldplane.T))
												neighbor_dist_testpoint2 = neighbor.rhomb_contains_point(
													lowered_testpoint2.dot(self.network.worldplane.T))
												if self_dist_origin == neighbor_dist_origin:
													#print("Origin distance: "+str(self_dist_origin))
													if self_dist_testpoint == neighbor_dist_testpoint:
														#print("Test1 distance: " + str(self_dist_testpoint))
														if self_dist_testpoint2 == neighbor_dist_testpoint2:
															print("Your scheme will never work!")
												self.children[child_i] = neighbor_child
		return self.children

	def get_existing_children(self):
		"""
		Note; these are typically not in an order matching self.blocks_neighbor.
		:return:
		"""
		return [child for child in self.children if child != None]

	def rhomb_contains_point(self, point):
		"""
		Quick check for point containment using just the rhombohedral shape of the chunk. Bear in mind a chunk's
		sub-chunks or blocks can extend out from the rhombohedron or not cover the entire rhombohedron (where neighbor
		chunks' sub-chunks/blocks extend in). Use safely_contains_point to check for certain.
		:param point: The point to be checked.
		:return: A signed distance function which is positive when the point lies inside the chunk; the distance from
		the point to the nearest face of the rhombohedron (scaled so that the center of the chunk is at distance 0.5).
		Suitable for checking whether a sphere falls fully within the rhombohedron. Not suitable for checking for
		spherical overlap, since the negative values are not accurate Euclidean distances.
		"""
		# TODO I will often want to do this in large contiguous batches of chunks, wherein there are probably good
		# 	search strategies; or at least I could probably feed all the block coordinates to a single Numpy command
		# 	and do the math much more quickly. One search strategy might be to use neighbor relations to move along
		# 	Conway worms, starting perhaps from some good guess; but then I need to store enough structural info to
		# 	make Conway worms easy to traverse. Also it would be nice to have enough structural info to quickly generate
		# 	a good guess - grabbing a lattice point with at least one 6D coordinate already close.
		# 	Hmm, doing the Conway worm traversal requires that the set of chunks being searched be "Conway convex",
		# 	with all shortest Conway worm paths falling entirely in the set. Would be fun to have a "Conway closure"
		# 	function, for making smoothed shapes in terrain gen.
		# Historical note: the above comment led to a mathematical tangent, me emailing Peter Hilgers, and
		# my collaborating with him and Anton Shutov on a paper.

		# If level < 10, we have a lookup table of the matrices.
		if self.level < 10:
			axes_matrix = self.network.axes_matrix_lookup[self.network.all_chosen_centers[self.template_index]][self.level]
		else:
			# Determine orientation via template's center point.
			ch_c = np.array(self.network.chunk_center_lookup(self.template_index))
			# Chunks work opposite to blocks - the axes which are integer at their center are the ones in which the chunk
			# has positive size.
			axes = np.nonzero(ch_c - np.floor(ch_c) - 0.5)[0]
			# Convert point into the rhombohedron's basis for easy math
			# TODO Should use the golden field object to calculate these accurately; matrix inverse will introduce error.
			axes_matrix = np.linalg.inv(self.network.worldplane.T[axes]*self.network.phipow(3*self.level))
		worldplane_chunk_origin = self.get_offset(0).dot(self.network.worldplane.T)
		target_coords_in_chunk_axes = (point - worldplane_chunk_origin).dot(axes_matrix) - 0.5
		# To measure distance from closest face-plane, we take the minimum of the absolute value.
		dist_from_center = np.abs(target_coords_in_chunk_axes)
		return np.min(0.5 - dist_from_center)

	def get_offset(self, level=0):
		"""
		Returns the chunk's offset, scaled appropriately for the requested level.
		:param level: Optional argument, the level to scale the offset to. The default is zero, referring to block level;
		note that this convention means self.offset will differ from self.get_offset(self.level). A chunk's stored offset
		is equal instead to self.get_offset(self.level - 1).
		:return: A numpy array, giving 6D coordinates of the chunk's location.
		"""
		# TODO Add documentation focused on avoiding floating point error.
		#return np.linalg.matrix_power(np.array(self.network.deflation_face_axes), (self.level-1) - level).dot(self.offset)
		return self.network.custom_pow(np.array(self.network.deflation_face_axes), self.level - 1 - level).dot(self.offset)

	def find_triacontahedron(self):
		"""
		Locates a triacontahedron within the chunk.
		:return: None if no triacontahedron is found; otherwise, a set of 20 sub-chunks forming one.
		"""
		# TODO Not the best place for this. Do I want it to operate on arbitrary contiguous clumps of chunks? Or
		# 	maybe return triacontahedra of a target level near a target point?

		# Triacontahedra are sets of tiles which all come from 3-faces of the same hypercube. Hypercubes have centers
		# with all-half-integer values. So the procedure is to look for large sets of tile centers whose half-integer
		# positions all equal those of the same central hypercube, and whose integer positions are only away from that
		# center by 0.5. Speed isn't too important for this - its main use case right now is setting up the starting
		# planet and moon. But if I use it for creating decorative trees and other terrain features later, I'll need
		# to worry a bit more about speed.
		pass

	def safely_contains_point(self, point, block_level=0):
		"""
		Returns true if the point can definitely be said to be inside the chunk - meaning it would be inside some block
		which is a descendant of the chunk. This function generates false negatives, and doesn't actually search
		existing children; it's meant as the check one would run to decide whether to search children.
		:param point: The point to be checked, converted into block-level worldplane coordinates.
		:param block_level: At and below this level, no safety margin is used since blocks genuinely have rhombohedral boundaries.
		:return: Boolean.
		"""
		# TODO Determine what the safe margins are. There are lots of clever things I could do, but this function
		#  mainly has to be fast. Also: the logic of _child_chunk_search is based on sorting children by their
		#  probability of containing a point, and for it to work safely_contains_point and might_contain_point need
		#  to be merely based on cutoffs in the value of some norm. So if I make any improvements here, I may need to
		#  make a new norm function corresponding with that (which also has to be fairly fast).
		# Brainstorming:
		# - Chunk corners always contain blocks aligned with the chunk axes, so those areas are safe. Actually any of the
		# 		"always included" blocks which are part of every chunk could be tested.
		# - I could take an intersection of all chunk templates and then create a convex interior of it with not too many faces.
		# - Function could depend on self.level, using the chunk's template itself when level = 1, and growing more cautious higher up.
		# - I could search through all templates for vertices which are missing a neighbor (ie, one of the blocks touching
		#		that vertex is outside the template), and take note of the distance of that vertex into the chunk as
		#		scored by rhomb_contains_point. Any point further in than that would be safely in the chunk. -Ive done this now
		# - I could do the above, but use two fast but differently-shaped distance metrics. Current rhomb_contains function
		# 		is like L_inf norm, but I could use taxi cab norm as well; combining the two is like using a cube and
		# 		an octahedron together as a nonconvex boundary.
		# - For higher confidence I should do the exhaustive template search thing using, essentially, superchunks
		# 		instead of chunks.
		if self.level <= block_level:
			# At block level, rhomb_contains_point is the true containment. Not sure if any
			return self.rhomb_contains_point(point) > 0
		# self.rhomb_contains_point gives us a value of 0 at rhombus boundary and 0.5 at center.
		# We know that at worst, a prolate rhomb could point directly into a face (but, this never actually happens!) and
		# that rhomb could be almost halfway inside the chunk. It turns out that with side length equal to one, a
		# prolate rhomb has a diagonal of sqrt((1/100.0)*(56*sqrt(5)+156)).
		# Changed from theoretical value 0.1979 to value 0.1420 below based on brute force examination of templates
		# TODO Value of 0.14199511391282338 was not completely reliable; parent would 'safely contain' target but
		#  children may not contain it. As far as I've observed, the closest child has always been very close to target,
		#  for example 0.030, 0.042 or 0.048 away from bounding rhomb. Observed at levels 1 and 2 of search. For now,
		#  I'm changing the value here back to theoretical, but I'd like to figure out why my brute force measurement
		#  was wrong.
		# More apparent exceptions: 0.2504585925389473, 0.2563185433459001, 0.2559212731069851, 0.2060401723750226,
		# 0.25458511670379114. 0.27080803857963665, 0.27049171239589354,
		# The following values are apparently caused by missing level 6 chunks below a level 7 chunk; values were
		# reported as the player traveled through the level 7 chunk, quite obviously inside it the entire time. So,
		# at least some of the problem is caused by this. 0 . 3 3 9 2 4 3 1 6 5 2 6 0 7 2 2 0 7, 0 . 3 3 9 2 4 5 7 3 6 5 9 9 6 4 8 1
		# 0 . 3 3 9 2 4 8 3 0 7 9 3 8 5 7 3 9 7,  0 . 3 3 9 4 5 2 7 2 9 3 8 3 1 9 0 5, 0 . 3 3 9 4 5 5 3 0 0 7 2 2 1 1 6 4,  0 . 3 3 9 4 9 0 6 5 6 6 3 2 3 4 8 8, 0 . 3 3 9 4 9 5 7 9 9 3 1 0 2 0 0 8,
		# 0 . 3 3 9 5 0 5 4 4 1 8 3 1 1 7 3 3 7, 0 . 3 3 9 5 0 8 0 1 3 1 7 0 0 9 9 4, 0 . 3 3 9 6 9 5 7 2 0 9 1 1 6 9 6 8 6, 0 . 3 3 9 6 9 8 2 9 2 2 5 0 6 2 2 8 7, 0 . 3 3 9 7 0 0 8 6 3 5 8 9 5 4 8 9,
		# 0 . 3 3 9 9 7 7 1 6 9 6 5 3 0 1 1 8 3, 0 . 3 3 9 9 7 9 7 3 9 0 8 6 9 4 0 5 7, 0 . 3 4 0 0 3 1 7 7 0 1 2 3 9 9 9 6 5
		# Using a more cautious, less principled value for now.....
		return self.rhomb_contains_point(point) > 0.15#0.271#0.19793839129906832

	def might_contain_point(self, point, block_level = 0):
		"""
		Returns true if the point is inside the rhombohedral boundary, but also returns true if the point falls close
		enough to the chunk that some block belonging to this chunk might jut out of the chunk's rhombohedral boundary
		and turn out to contain the point.
		:param point: A 3D point, given in worldplane coordinates.
		:param block_level: At and below this level, no safety margin is used since blocks genuinely have rhombohedral boundaries.
		:return: Boolean.
		"""
		if self.level <= block_level:
			return self.rhomb_contains_point(point) > 0
		# For now, just using the same quick test as safely_contains_point.
		# TODO safely_contains and might_contain ought to be coupled, to prevent me changing the value in one place and
		#  forgetting the other (which I just now did).
		return self.rhomb_contains_point(point) > -0.15#0.271#-0.19793839129906832#-0.14199511391282338

	def chunk_at_location(self, target, target_level=0, generate=False, verbose=False):
		"""
		Takes a 3D coordinate and returns the smallest list of already-generated chunks guaranteed to contain that coordinate.
		The search proceeds from the present chunk, so it should be treated as a best-guess.
		:param target:
		The coordinates to search for.
		:param target_level:
		The level of chunk desired, with blocks being target_level 0. If chunks of the desired level have not yet been
		generated, by default this will return the closest available level.
		:param generate:
		Set generate=True to force generation of the desired level of chunks. Note, this can either force generation
		down to target_level, or up to target_level, or some combination; for example we can request a chunk of level 10
		(about the scale of Earth's moon), and place it a cauple hundred million blocks away from us (about the distance
		to the moon), and this forces the generation of a highest-level chunk of approximately level 14, subdivided just
		enough to produce the requested level 10 chunk.
		:return: Returns a list of Chunk objects. If these are block-level or below, they are just temporary objects
		for sake of convenience; they know which chunks are their parent but those chunks don't acknowledge them as
		children. If generate=False, the list will be empty if nothing appropriate exists.
		"""
		if verbose: print("Search invoked at level "+str(self.level)+".\nTarget level "+str(target_level)+".")
		# TODO When generate=False and target_level is below what's been generated, should dummy chunks be returned
		#  like in the block case? Or maybe target_level=1 should be the default and the dummy chunks system should
		#  be replaced with an extra function for finding block coordinates within a chunk.
		# To avoid moving up and down and up and down the tree of chunks, we must first move up and then move down.
		# Recursive calls of chunk_at_location will move up only.
		if self.safely_contains_point(target):
			if verbose: print("Level "+str(self.level)+" safely contains target.")
			if self.level == target_level:
				return [self]
			if self.level > target_level:
				# We're far up enough; move downwards.
				return self._child_chunk_search(target, target_level, generate, verbose)
			if self.level < target_level:
				if self.parent != None:
					return self.parent.chunk_at_location(target, target_level, generate, verbose)
				else:
					if generate:
						return self.get_parent().chunk_at_location(target, target_level, generate, verbose)
					else:
						# We're the highest-level chunk available which contains the point, although a higher level
						# was requested for some strange reason. This is odd enough that it should be logged, but,
						# the right behavior is to return ourselves.
						print("WARNING: Call to chunk_at_location probably missing argument 'generate=True'; "
							  + "a chunk of higher level than available was requested.")
						return [self]
		else:
			if self.parent is not None:
				return self.parent.chunk_at_location(target, target_level, generate, verbose)
			else:
				if generate:
					print("Top-level chunk is "+str(self.level)+"; target not safely inside."
						  +"Target distance: "+str(self.rhomb_contains_point(target))+". Generating new parent.")
					return self.get_parent().chunk_at_location(target, target_level, generate, verbose)
				else:
					if self.might_contain_point(target):
						if target_level < self.level:
							# We don't know for sure the point lies within this chunk, but we can check anyway.
							return self._child_chunk_search(target, target_level, generate, verbose)
						else:
							# The point could be here, but just returning [self] wouldn't return a list guaranteed to
							# contain the target, so we have to give an empty return.
							print("Search returning because we can't generate high enough chunk levels for certainty.")
							return []
					else:
						# The point lies outside the generated chunk tree.
						print("Search returning; target lies outside existing tree and generate=False.")
						return []

	def _child_chunk_search(self, target, target_level, generate, verbose=False):
		"""
		This function is meant to be called from within chunk_at_location in order to do the downward half of the
		recursive search. The functionality is the same as chunk_at_location except that this function assumes the
		target lies within this chunk.
		:param target: The coordinates to search for.
		:param target_level: The level of chunk desired, with blocks being target_level 0. If chunks of the desired
		level have not yet been generated, by default this will return the closest available level.
		:param generate: Set generate=True to force generation of the desired level of chunks. Note, this can either
		force generation down to target_level, or up to target_level, or some combination; for example we can request a
		chunk of level 10 (about the scale of Earth's moon), and place it a cauple hundred million blocks away from us
		(about the distance to the moon), and this forces the generation of a highest-level chunk of approximately level
		14, subdivided just enough to produce the requested level 10 chunk.
		:return: Returns a list of Chunk objects. If these are block-level or below, they are just temporary objects
		for sake of convenience; they know which chunks are their parent but those chunks don't acknowledge them as
		children. If generate=False, the list will be empty if nothing appropriate exists.
		"""
		if verbose: print("Search descended to level "+str(self.level))
		if self.level == target_level:
			# We assume the function wouldn't have been called if the point weren't nearby, so, this chunk is the best bet.
			if verbose: print("This is the target level; returning self. Containment: "+str(self.rhomb_contains_point(target)))
			return [self]
		if self.level < target_level:
			# This should be unreachable.
			raise Exception("_child_chunk_search has descended lower than its target level of " + str(target_level)
							+ ".\nCurrent chunk level: " + str(self.level) + "."
							+ "\nWas _child_chunk_search accidentally called instead of chunk_at_location?")
		child_list = []
		if generate or self.all_children_generated:
			child_list = self.get_children()
		else:
			child_list = self.get_existing_children()
			if child_list is None or len(child_list) == 0:
				# We can't eliminate the possibility that the target is here, so we return self.
				if verbose: print("No available children; returning self. Containment: " + str(self.rhomb_contains_point(target)))
				return [self]
		priority_list = [child.rhomb_contains_point(target) for child in child_list]
		sorted_indices = list(range(len(priority_list)))
		sorted_indices.sort(key=lambda x: priority_list[x], reverse=True)
		if child_list[sorted_indices[0]].safely_contains_point(target):
			# No need to search more than one children, this one contains the point.
			if verbose: print("A child safely_contains; returning it. Containment: " + str(self.rhomb_contains_point(target)))
			return child_list[sorted_indices[0]]._child_chunk_search(target, target_level, generate)
		# Handle some special cases where no recursive search is needed
		if self.safely_contains_point(target):
			if verbose: print("Level "+str(self.level)+" safely contains target; checking children. Containment: "+ str(self.rhomb_contains_point(target)))
			if priority_list[sorted_indices[0]] < 0:
				# (One might think the condition here should be based on 'possibly contains', ie, if we safely
				# contain the target at least one child should possibly contain it; but actually 'safely contains'
				# provides a guarantee of the stronger condition above. This is as it should be; safely_contains
				# is extremely cautious.)
				if generate or self.all_children_generated:
					# We 'safely contain' the target but none of our children contain it??
					#raise Exception("Search terminated at level "+str(self.level)
					#				+": This chunk 'safely contains' target point but none of its children do.\n"
					#				+ "Target: "+str(target)
					#				+ "\nChunk offset: "+str(self.get_offset(level=0))
					#				+ "\nClosest child was "+str(-priority_list[sorted_indices[0]])+" away."
					#				+ "\nChild offset: "+str(child_list[sorted_indices[0]].get_offset(level=0)))
					print("Search terminated at level "+str(self.level)
									+": This chunk 'safely contains' target point but none of its children do.\n"
									+ "Target: "+str(target)
									+ "\nChunk offset: "+str(self.get_offset(level=0))
									+ "\nClosest child was "+str(-priority_list[sorted_indices[0]])+" away."
									+ "\nChild offset: "+str(child_list[sorted_indices[0]].get_offset(level=0)))

					return [self]
				else:
					# Some child must contain the point, but it apparently hasn't been generated yet; this chunk is the
					# best to return (assuming I even understand the use cases at all)
					if verbose: print("Correct child not generated; returning self. Containment: " + str(
						self.rhomb_contains_point(target)))
					return [self]
		else:
			# Target not within "safely_contains" margin; may actually belong to a neighbor.
			# However, search process reached this chunk, so this chunk is the guess nearest the point.
			if generate or self.all_children_generated:
				if not child_list[sorted_indices[0]].might_contain_point(target):
					# No point looking further.
					if verbose: print("No child contains target; returning []. Containment: " + str(
						self.rhomb_contains_point(target)))
					return []
		# OK, no one child safely contains the point, and we weren't able to return [self] or [] based on a quick check.
		# So we need to return a list of all children which might contain it, combining output from multiple recursive
		# calls. We still may return [self] in the generate=False case, where we can't be sure our list is complete.
		if generate or self.all_children_generated:
			results = []
			for child_i in sorted_indices:
				if not child_list[child_i].might_contain_point(target):
					# Since children are ordered by distance to target, and that's all that's used in might_contain_point,
					# we know no further children might contain the target.
					# No need to return [self] if 'results' is empty; we then know this chunk doesn't contain the point.
					if verbose: print("No one child confirmed; returning list of "+str(len(results))
									  +" possible children. Containment: " + str(self.rhomb_contains_point(target)))
					break
				child_search_results = child_list[child_i]._child_chunk_search(target, target_level, generate, verbose)
				# If any one search returns something which definitely contains the target, we need to simply
				# return that.
				# TODO Write a test that we always manage to return a list of length one when we find the target inside
				#  some chunk.
				if len(child_search_results) == 1:
					# We only need to check when the length is 1, since the recursive call would return just one result
					# if that result actually contained the point.
					if child_search_results[0].safely_contains_point(target):
						if verbose: print("Encountered child safely containing point; returning it. Containment: " + str(
							self.rhomb_contains_point(target)))
						return child_search_results
				results += child_search_results
			if len(results) == 0 and self.safely_contains_point(target):
				#raise Exception("Chunk at level "+str(self.level)+"'safely contains point', yet recursive search turned up empty."
				#				+"\nContainment: "+str(self.rhomb_contains_point(target))
				#				+"\nClosest child: "+str(priority_list[sorted_indices[0]]))
				print("Chunk at level "+str(self.level)+"'safely contains point', yet recursive search turned up empty."
								+"\nContainment: "+str(self.rhomb_contains_point(target))
								+"\nClosest child: "+str(priority_list[sorted_indices[0]]))
				return [self]

			return results
		else:
			# Now we need to check if any one child definitely contains the point. If not, the target could be in
			# a non-generated child, or could be in a neighboring chunk. So, this specific case is likely to cause
			# a lot of fruitless searching.
			# TODO This could be abbreviated by generating and returning 'temporary' chunks.
			# Note: at the moment, I have never seen this warning get printed, which is actually a bit odd.
			print(
				"Warning: Potentially expensive search is occurring at edge of generated grid. If this message"
				+ " is printed many times, consider using generate=True in the code responsible for the warning.")
			for child_i in sorted_indices:
				if not child_list[child_i].might_contain_point(target):
					# We've gone far enough in the list with no results.
					return [self]
				child_search_results = child_list[child_i]._child_chunk_search(target, target_level, generate)
				# We only care if one of the results definitely contains the target.
				# We only need to check when the length is 1, since the recursive call would return just one result
				# if that result actually contained the point.
				if len(child_search_results) == 1:
					# TODO In some cases we call safely_contains_point or a related function 3 times per chunk
					#  (before the recursive search call, during it, and after it returns). Do one call and just
					#  pass around the resulting information?
					if child_search_results[0].safely_contains_point(target):
						return child_search_results
			# If we didn't return during the loop, there are no children definitely containing the target.
			# We don't want to return a list of all children that might contain the target, because it might
			# be incomplete. We shouldn't return an empty list, because the search process reached this far so
			# the present chunk is a possible location of target (it might even be a sure location).
			return [self]

	def highlight_block(self):
		"""
		Draws an outline around a block.
		:return: None.
		"""
		# TODO This should make use of self.network.draw_block_wireframe.
		template_center = self.network.chunk_center_lookup(self.template_index)
		# We use get_offset's default level (zero) so we don't need scale multipliers.
		block = self.get_offset() + 0.5 - (template_center - np.floor(template_center))
		block = np.array(np.round(block * 2),dtype=np.float) / 2.0
		#print("hilite "+str(block))
		face_origin = np.floor(block).dot(self.network.worldplane.T)
		face_tip = np.ceil(block).dot(self.network.worldplane.T)
		dir1, dir2, dir3 = np.eye(6)[np.nonzero(np.ceil(block) - np.floor(block))[0]].dot(
			self.network.worldplane.T)
		face_origin = Vector3(face_origin[0],face_origin[1],face_origin[2])
		face_tip = Vector3(face_tip[0],face_tip[1],face_tip[2])
		dir1 = Vector3(dir1[0], dir1[1], dir1[2])
		dir2 = Vector3(dir2[0], dir2[1], dir2[2])
		dir3 = Vector3(dir3[0], dir3[1], dir3[2])
		dumb_highlight = MeshInstance.new()
		dumb_highlight.mesh = SphereMesh()
		dumb_highlight.translation = face_origin
		self.network.add_child(dumb_highlight)
		dh_pos = dumb_highlight.global_transform.origin
		dh_pos = np.array([dh_pos.x, dh_pos.y, dh_pos.z])
		player_pos = self.network.get_node("../../Player").transform.origin
		player_pos = np.array([player_pos.x, player_pos.y, player_pos.z])
		print(dh_pos - player_pos)
		self.network.block_highlight.begin(Mesh.PRIMITIVE_LINES)
		self.network.block_highlight.add_vertex(face_origin)
		self.network.block_highlight.add_vertex(face_origin+dir1)
		self.network.block_highlight.add_vertex(face_origin)
		self.network.block_highlight.add_vertex(face_origin + dir2)
		self.network.block_highlight.add_vertex(face_origin)
		self.network.block_highlight.add_vertex(face_origin + dir3)
		self.network.block_highlight.add_vertex(face_origin)
		self.network.block_highlight.end()

	def draw_mesh(self, drawp = lambda x: True):
		"""
		Creates a mesh consisting of the chunk's children (ie, blocks, if the chunk is level 1); then adds that mesh
		as a child of self.network so that it will be drawn.
		:param drawp: Optional function which can take a block and returns true or false. If false is returned, we
		won't draw that block.
		:return: none
		"""
		starttime = time.perf_counter()

		self.drawn = True
		# TODO Generate mesh in separate thread(s), like the pure-gdscript voxel game demo does.
		#st = SurfaceTool()
		if self.level >= 1:
			multiplier = self.network.phi_powers[(self.level-1)*3]
		if self.level < 1:
			multiplier = 1.0/self.network.phi_powers[-((self.level-1)*3)]
		#st.begin(Mesh.PRIMITIVE_TRIANGLES)
		#st.add_color(Color(r.random(), r.random(), r.random()))

		vertices = PoolVector3Array()
		indices = PoolIntArray()
		normals = PoolVector3Array()

		body = StaticBody.new()
		collider = CollisionShape.new()
		body.add_child(collider)
		# TODO If I can write a custom collider shape for the Ammann rhombohedra, there may be ways to make it fast.
		collider.shape = ConcavePolygonShape()
		collider_face_array = PoolVector3Array()

		preliminaries = time.perf_counter()

		# The block corner indices are added in a predictable order, so we can construct our index array all at once.
		# The pattern for one block is stored in the Chunk_Network as rhomb_indices. Here's how it looks.
		# rhomb_indices = np.array([0, 2, 7, 0, 7, 3, 0, 3, 5, 0, 5, 4, 0, 4, 6, 0, 6, 2, 1, 4, 5, 1, 6, 4, 1, 2, 6, 1, 7, 2, 1, 3, 7, 1, 5, 3])
		# Now we just need to add the offset of 8 for each new block.
		to_draw = np.array([self.block_values[block_i] > 0 and drawp(self.blocks[block_i]) for block_i in range(len(self.blocks))])
		blocks_to_draw = [self.blocks[i] + self.offset for i in range(len(to_draw)) if to_draw[i]]
		# Check for exposed faces
		# print(sum(self.blocks_neighbor[block_i]))
		# if sum(self.blocks_neighbor[block_i]) >= 6:
		# 	# Six neighbors lie inside this chunk
		# 	if np.nonzero(self.block_values[np.nonzero(self.blocks_neighbor[block_i])[0]])[0].shape[0] == 6:
		# 		# Skip this block
		# 		print("Skipping a block yay!")
		# 		continue
		build_draw_list = time.perf_counter()

		num_to_draw = sum(to_draw)

		if num_to_draw == 0:
			collider.free()
			body.free()
			return

		index_data = np.tile(self.network.rhomb_indices,num_to_draw) \
					 + 8*np.repeat(np.arange(num_to_draw),len(self.network.rhomb_indices))
		# Careful now!
		indices.resize(36 * num_to_draw)
		with indices.raw_access() as indices_dump:
			for i in range(36 * num_to_draw):
				indices_dump[i] = index_data[i]

		index_precalc = time.perf_counter()

		block_origins = np.floor(blocks_to_draw).dot(self.network.worldplane.T) * multiplier
		block_axes = ((np.array([np.eye(6)]*len(blocks_to_draw))[np.ceil(blocks_to_draw)
				- np.floor(blocks_to_draw) != 0]).dot(self.network.worldplane.T) * multiplier
					  ).reshape((len(blocks_to_draw),3,3))
		# right hand rule
		block_axes_flipped = block_axes[:,[1,0,2],:]
		block_axes = np.where(
			np.repeat(np.diag(np.cross(block_axes[:,0],block_axes[:,1]).dot(block_axes[:,2].T)) < 0, 9).reshape(-1,3,3),
							  block_axes_flipped, block_axes)
		corner1 = block_origins
		corner2 = block_origins + np.sum(block_axes, axis=1)
		corner3 = block_origins + block_axes[:,0]
		corner4 = block_origins + block_axes[:,1]
		corner5 = block_origins + block_axes[:,2]
		corner6 = corner2 - block_axes[:,0]
		corner7 = corner2 - block_axes[:,1]
		corner8 = corner2 - block_axes[:,2]

		center = (block_origins + np.sum(block_axes, axis=1)/2).repeat(8,axis=0)

		vertices_data = np.array([corner1, corner2, corner3, corner4,
								  corner5, corner6, corner7, corner8]).transpose((1,0,2)).reshape((-1,3))

		vertices.resize(8 * num_to_draw)
		with vertices.raw_access() as vertices_dump:
			for i in range(8 * num_to_draw):
				vertices_dump[i] = Vector3(vertices_data[i][0], vertices_data[i][1], vertices_data[i][2])

		normal_data = (vertices_data - center)/np.linalg.norm(vertices_data - center,axis=1).repeat(3).reshape((-1,3))
		normals.resize(8 * num_to_draw)
		with normals.raw_access() as normals_dump:
			for i in range(8 * num_to_draw):
				normals[i] = Vector3(normal_data[i][0], normal_data[i][1], normal_data[i][2])

		vertex_precalc = time.perf_counter()

		# Now that we've got the vertices, let's try and calculate the colliders in one step too.
		if len(vertices) > 0:
			# Converting the PoolVector3Array to a numpy array simply flattens, for 3x the length
			collider_data = np.array(vertices)[index_data]
			# Careful now!
			collider_face_array.resize(36 * 3 * num_to_draw)
			with collider_face_array.raw_access() as collider_dump:
				for i in range(36*num_to_draw):
					collider_dump[i] = collider_data[i]
		collision_precalc = time.perf_counter()
		# Finalize mesh for the chunk

		new_mesh = ArrayMesh()
		arrays = Array()
		arrays.resize(ArrayMesh.ARRAY_MAX)
		arrays[ArrayMesh.ARRAY_VERTEX] = vertices
		arrays[ArrayMesh.ARRAY_INDEX] = indices
		arrays[ArrayMesh.ARRAY_NORMAL] = normals
		arrays[ArrayMesh.ARRAY_COLOR] = PoolColorArray(Array([Color(r.random(),r.random(),r.random())]*len(vertices)))
		new_mesh.add_surface_from_arrays(ArrayMesh.PRIMITIVE_TRIANGLES,arrays)
		new_mi = MeshInstance.new()
		new_mi.mesh = new_mesh
		new_mesh.surface_set_material(new_mesh.get_surface_count() - 1, COLOR)

		self.network.add_child(new_mi)
		new_mi.show()

		add_surface = time.perf_counter()

		# Finalize collision shape for the chunk
		collider.shape.set_faces(collider_face_array)
		body.collision_layer = 0xFFFFF
		body.collision_mask = 0xFFFFF
		new_mi.add_child(body)

		add_collider = time.perf_counter()

		self.mesh = new_mi
		self.collision_mesh = body
		final_time = time.perf_counter()
		#print("\nPreliminaries:     "+str(preliminaries - starttime))
		#print("Build draw list:   "+str(build_draw_list - preliminaries))
		#print("Index building:    "+str(index_precalc - build_draw_list))
		#print("Vertex building:   "+ str(vertex_precalc - index_precalc))
		#print("Collision precalc: "+str(collision_precalc - vertex_precalc))
		#print("Add surface:       "+str(add_surface - collision_precalc))
		#print("Add Collider:      "+str(add_collider - add_surface))
		#print("Final time:        "+ str(final_time - add_collider))

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
