from godot import exposed, export
from godot import *
import numpy as np
from debugging import debugging
import random as r
import time

COLOR = ResourceLoader.load("res://new_spatialmaterial.tres")

@exposed(tool=True)
class numpylattice(MeshInstance):
	
	def chunk_test(self, a = None, chosen_center = None, constraints = None):
		if a is None:
			a = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
		starttime = time.perf_counter()
		phi = 1.61803398874989484820458683
		# Set up a 6D array holding 6-vectors which are their own coords
		# w/in the array
		esize = 10
		offset = -5
		embedding_space = np.indices((esize,esize,esize,esize,esize,esize),dtype=np.int8).T[...,-1::-1] + offset
		# The basis for the worldplane
		worldplane = np.array([[phi,0,1,phi,0,-1],[1,phi,0,-1,phi,0],[0,1,phi,0,-1,phi]])
		normalworld = worldplane / np.linalg.norm(worldplane[0])
		squareworld = normalworld.transpose().dot(normalworld)
		
		# I think "parallel space" is the right term here - parallel to
		# the projection to the worldplane, so of course perpendicular to
		# the worldplane.
		parallelspace = np.array([[-1/phi,0,1,-1/phi,0,-1],
								  [1,-1/phi,0,-1,-1/phi,0],
								  [0,1,-1/phi,0,-1,-1/phi]])
		normallel = parallelspace / np.linalg.norm(parallelspace[0])
		squarallel = normallel.T.dot(normallel)
		
		# Slide worldplane origin to a point where it's perpendicular to the worldplane
		#a = a - a.dot(squarallel)
		a = a.dot(squarallel)
		print(a)
		
		deflation = np.array([[2, 1,-1,-1, 1, 1],
							  [1, 2, 1,-1,-1, 1],
							  [-1,1, 2, 1, -1,1],
							  [-1,-1,1, 2, 1, 1],
							  [1,-1,-1, 1, 2, 1],
							  [1, 1, 1, 1, 1, 2]])
		deflation_face_axes = [ [ 2, 1, 1, 1, 1,-1],
								[ 1, 2, 1,-1, 1, 1],
								[ 1, 1, 2, 1,-1, 1],
								[ 1,-1, 1, 2,-1,-1],
								[ 1, 1,-1,-1, 2, 1],
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
		dists = np.sum(np.stack(np.repeat([embedding_space
			.reshape((-1,6))-a],15,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals])
			.reshape((1,15,3))),axis=-1)
		distance_scores = np.max(np.abs(dists),axis=1)
		included = distance_scores < np.linalg.norm(np.array([0,0,1,-1,-1,-1]).dot(normallel.T))/2
		included = included.reshape((esize,esize,esize,esize,esize,esize))
		deflation_included = distance_scores < np.linalg.norm(np.array([0,0,1,-1,-1,-1]).dot(normallel.T))/(2*phi*phi*phi)
		deflation_included = deflation_included.reshape((esize,esize,esize,esize,esize,esize))
		
		# Need to know which lines are included, not which points; so we shift
		# the "included" array over by one in each dimension to compare it
		# with itself.
		
		# Each point present in the list line<n> represents a line starting at that
		# point and proceeding 1 unit in the nth positive direction.
		lines0 = embedding_space[np.nonzero(np.all([included[:esize-1],included[1:]],axis=0))]
		lines1 = embedding_space[np.nonzero(np.all([included[:,:esize-1],included[:,1:]],axis=0))]
		lines2 = embedding_space[np.nonzero(np.all([included[:,:,:esize-1],included[:,:,1:]],axis=0))]
		lines3 = embedding_space[np.nonzero(np.all([included[:,:,:,:esize-1],included[:,:,:,1:]],axis=0))]
		lines4 = embedding_space[np.nonzero(np.all([included[...,:esize-1,:],included[...,1:,:]],axis=0)) ]
		lines5 = embedding_space[np.nonzero(np.all([included[...,:esize-1],included[...,1:]],axis=0))]
		
		all_lines = [lines0,lines1,lines2,lines3,lines4,lines5]
		# Deflated lines are of distance 3 apart, having a distance of 1 in five
		# of their coordinates and a distance of 2 in one of them.
		deflated_lines = [[],[],[],[],[],[]]
		for i in embedding_space[deflation_included]:
			for j in embedding_space[deflation_included]:
				#print(np.linalg.norm(i - j))
				#print(np.sum(np.abs(i - j)))
				#print(np.where(i - j == 2)[0].shape == (1,))
				if np.linalg.norm(i - j) == 3:
					if np.sum(np.abs(i - j)) == 7:
						if np.where(i - j == 2)[0].shape == (1,):
							deflated_lines[np.where(i - j == 2)[0][0]].append(j)
		
		if chosen_center is None:
			chunks = []
			for point in embedding_space[deflation_included]:
				# Find dimensions in which we have a neighbor in the positive direction
				pos_lines = set()
				for dim in range(6):
					if np.any(np.all(np.array(deflated_lines[dim]) - point == 0,axis=1)):
						pos_lines.add(dim)
				# Consider all sets of 3 such dimensions
				if len(pos_lines) >= 3:
					for i in pos_lines:
						for j in pos_lines - set([i]):
							for k in pos_lines - set([i,j]):
								# Check whether all eight cube corners exist with these axes
								try:
									if (deflation_included[tuple((point - offset)+deflation_face_axes[i]
															+deflation_face_axes[j])] and
										deflation_included[tuple((point - offset)+ deflation_face_axes[i]
															+deflation_face_axes[k])] and
										deflation_included[tuple((point - offset)+ deflation_face_axes[j]
															+deflation_face_axes[k])] and
										deflation_included[tuple((point - offset)+ deflation_face_axes[i]
															+deflation_face_axes[j]
															+deflation_face_axes[k])]):
										chunks.append(point+(np.array(deflation_face_axes[i])
											+np.array(deflation_face_axes[j])+np.array(deflation_face_axes[k]))/2)
								except IndexError:
									# If one of the indices was out of bound, it's not a chunk, so do nothing.
									pass
			print("Chunks computed. t="+str(time.perf_counter()-starttime)+" seconds")
			# TODO It seems that very occasionally, deflation_faces is empty at this
			# point; probably just means esize has to be greater than 8 to guarantee there are chunks.
			# Recording a-values where this happens: [ 0.03464434, -0.24607234,
			# -0.05386369,  0.2771699,   0.39596138,  0.45066235]
			
			# Choose a chunk near origin
			deflation_faces = np.array(chunks)
			chosen_center = deflation_faces[
				np.where(np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1)
				== np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1).min())[0][0]]
		print("Chose chunk "+str(chosen_center)+" second="+str(time.perf_counter()-starttime))
		chosen_axes = 1-np.array(chosen_center - np.floor(chosen_center))*2
		chosen_origin = chosen_center - np.array(deflation_face_axes).T.dot(1-chosen_axes)/2
		chosen_axis1 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][0]])
		chosen_axis2 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][1]])
		chosen_axis3 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][2]])
		
		blocks = np.zeros((0,6))
		for axes in ch3:
			ax1, ax2, ax3 = np.eye(6,dtype=np.int64)[np.nonzero(axes)[0]]
			#print([ax1,ax2,ax3])
			ax12 = ax1 + ax2
			ax13 = ax1 + ax3
			ax23 = ax2 + ax3
			ax123 = ax12 + ax3
			r1,r2,r3,r4,r5,r6,r7,r8 = (included[:esize-1,:esize-1,:esize-1,:esize-1,:esize-1,:esize-1],
				included[ax1[0]:esize-1+ax1[0],ax1[1]:esize-1+ax1[1],ax1[2]:esize-1+ax1[2],
					ax1[3]:esize-1+ax1[3],ax1[4]:esize-1+ax1[4],ax1[5]:esize-1+ax1[5]],
				included[ax2[0]:esize-1+ax2[0],ax2[1]:esize-1+ax2[1],ax2[2]:esize-1+ax2[2],
					ax2[3]:esize-1+ax2[3],ax2[4]:esize-1+ax2[4],ax2[5]:esize-1+ax2[5]],
				included[ax3[0]:esize-1+ax3[0],ax3[1]:esize-1+ax3[1],ax3[2]:esize-1+ax3[2],
					ax3[3]:esize-1+ax3[3],ax3[4]:esize-1+ax3[4],ax3[5]:esize-1+ax3[5]],
				included[ax12[0]:esize-1+ax12[0],ax12[1]:esize-1+ax12[1],ax12[2]:esize-1+ax12[2],
					ax12[3]:esize-1+ax12[3],ax12[4]:esize-1+ax12[4],ax12[5]:esize-1+ax12[5]],
				included[ax13[0]:esize-1+ax13[0],ax13[1]:esize-1+ax13[1],ax13[2]:esize-1+ax13[2],
					ax13[3]:esize-1+ax13[3],ax13[4]:esize-1+ax13[4],ax13[5]:esize-1+ax13[5]],
				included[ax23[0]:esize-1+ax23[0],ax23[1]:esize-1+ax23[1],ax23[2]:esize-1+ax23[2],
					ax23[3]:esize-1+ax23[3],ax23[4]:esize-1+ax23[4],ax23[5]:esize-1+ax23[5]],
				included[ax123[0]:esize-1+ax123[0],ax123[1]:esize-1+ax123[1],ax123[2]:esize-1+ax123[2],
					ax123[3]:esize-1+ax123[3],ax123[4]:esize-1+ax123[4],ax123[5]:esize-1+ax123[5]])
			nonzero = np.nonzero(np.all([r1,r2,r3,r4,r5,r6,r7,r8],axis=0))
			blocks = np.concatenate((blocks,embedding_space[nonzero]+np.array(ax123,dtype=np.float)/2))
		
		multiplier = 4
		array_mesh = ArrayMesh()
		self.mesh = array_mesh
		
		
		st = SurfaceTool()
		#print("Drawing neighbor blocks next. seconds="+str(time.perf_counter()-starttime))
		st.begin(Mesh.PRIMITIVE_LINES)
		st.add_color(Color(0,.5,0))
		neighbor_blocks = []
		for block in blocks:
			face_origin = np.floor(block).dot(worldplane.T)*multiplier
			face_tip = np.ceil(block).dot(worldplane.T)*multiplier
			dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(worldplane.T)*multiplier
			corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = (
				face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
				face_tip - dir1, face_tip - dir2, face_tip - dir3
			)
			if np.any(np.abs((np.array(block).dot(worldplane.T)*multiplier - (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) > 0.5) and np.any(np.all(np.abs((np.array([
								corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8]) 
								- (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) < 0.5000001,axis=-1)):
				neighbor_blocks.append(block)
				# Represents a voxel on the boundary of our chosen chunk
				dir1 = Vector3(dir1[0],dir1[1],dir1[2])
				dir2 = Vector3(dir2[0],dir2[1],dir2[2])
				dir3 = Vector3(dir3[0],dir3[1],dir3[2])
		inside_blocks = []
		for block in blocks:
			if np.all(np.abs((np.array(block).dot(worldplane.T)*multiplier - (chosen_center)
								.dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) <= 0.5):
				# Represents a voxel inside our chosen chunk
				inside_blocks.append(block)
				
				face_origin = np.floor(block).dot(worldplane.T)*multiplier
				face_tip = np.ceil(block).dot(worldplane.T)*multiplier
				dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(worldplane.T)*multiplier
				dir1 = Vector3(dir1[0],dir1[1],dir1[2])
				dir2 = Vector3(dir2[0],dir2[1],dir2[2])
				dir3 = Vector3(dir3[0],dir3[1],dir3[2])
		all_owned_blocks = neighbor_blocks + inside_blocks
		# Now we want to calculate the validity bounds for this chunk.
		# Tentatively, only including vertices inside the chunk. I might want
		# to include all vertices of blocks which overlap the chunk, but it
		# seems like they'll be determined by the interior vertices.
		# OK, my constraints don't seem narrow enough to work so I'm guessing
		# this is the issue - for now I'm just going to widen the distance in
		# order to catch points nearby. Adding too many constraints doesn't seem
		# like it will do much harm - I will just end up with multiple identical
		# chunk templates. I can "glue together" the constraint regions.
		relevance = np.all(np.abs((embedding_space[included].dot(worldplane.T)*multiplier 
								- (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) < 1.5,axis=1)# Was "< 0.501"
		relevant_points = embedding_space[included][relevance]
		# Recalculating just because it's fast and easier than trying to get the
		# right numbers, arranged properly, out of the old "constraints" variable.
		dists = np.sum(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		# The intersection of all the constraints takes the minima along each of the 30 vectors.
		#if constraints is None:
		constraints = 0.9732489894677302 - np.max(dists,axis=0)
		print("Wiggle room in all 30 directions:")
		print(constraints)
		print("Wiggle room along the 15 axes:")
		print(constraints[:15]+constraints[15:])
		print("Proposed new point inside the constraints:")
		b = np.array([a[0]+r.random()/2-0.25,a[1]+r.random()/2-0.25,a[2]+r.random()/2-0.25,
					a[3]+r.random()/2-0.25,a[4]+r.random()/2-0.25,a[5]+r.random()/2-0.25])
		while np.any(np.concatenate([twoface_normals,-twoface_normals]).dot((a-b).dot(normallel.T)) > constraints):
			b = np.array([a[0]+r.random()/2-0.25,a[1]+r.random()/2-0.25,a[2]+r.random()/2-0.25,
					a[3]+r.random()/2-0.25,a[4]+r.random()/2-0.25,a[5]+r.random()/2-0.25])
			b = b.dot(squarallel)
		print(b)
		new_constraints = np.sum(np.stack(np.repeat([relevant_points - b],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		print("New max constraint distance: "+str(np.max(np.abs(new_constraints))))
		print("Predicted new max: "+str(np.max(
			-0.9732489894677302 + np.max(np.sum(
				(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T) )
				* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3)))
			,axis=2),axis=0)
			+ 0.9732489894677302 + np.dot(
				np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),
				(a-b).dot(normallel.T)
			)
			)))
		print("Predicted to pass: "+str(np.all(
			+ np.dot(
				np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),
				(a-b).dot(normallel.T))
			< 0.9732489894677302 - np.max(np.sum(
				(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T) )
				* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3)))
			,axis=2),axis=0)
			)))
		return (b, constraints, chosen_center, all_owned_blocks)
	
	def _ready(self):
		# TODO This needs to be rewritten to generate integer 3D coordinates
		# by using the golden field (R adjoin phi) rather than just R.
		starttime = time.perf_counter()
		phi = 1.61803398874989484820458683
		# Set up a 6D array holding 6-vectors which are their own coords
		# w/in the array
		esize = 10
		offset = -5
		#embedding_space = np.zeros((esize,esize,esize,esize,esize,esize,6),dtype=np.int8)
		print("Creating embedding space; t="+str(time.perf_counter()-starttime))
#		for index,x in np.ndenumerate(embedding_space[...,0]):
#			embedding_space[index] = np.array(index) + offset
		embedding_space = np.indices((esize,esize,esize,esize,esize,esize),dtype=np.int8).T[...,-1::-1] + offset
		print("Done creating embedding space; t="+str(time.perf_counter()-starttime))
		# "a" is the origin of the worldplane. Just want it to be nonsingular,
		# ie, worldplane shouldn't intersect any 6D integer lattice points.
		# Also want worldplane to intersect at least some of the values we
		# actually check (so, inside the range of the embedding_space).
		#a = np.array([5.1+phi/8,5.2+phi/8,5.3,5.4,5.5,5.6])
		a = np.array([0.1201,0.61102,0.1003,0.60904,0.0805,0.60706])
		a -= np.array([0.26201,0.0611,0.15003,0.16094,0.12805,0.20706])
		a = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
		a = np.array([-0.16913145,  0.04060133, -0.33081354,  0.76832666,  0.53877964,  0.63870467])
		a = np.array([-0.0441522,  -0.09743448, -0.38699097,  0.79503878,  0.61608302,  0.82796904])
		# The basis for the worldplane
		worldplane = np.array([[phi,0,1,phi,0,-1],[1,phi,0,-1,phi,0],[0,1,phi,0,-1,phi]])
		normalworld = worldplane / np.linalg.norm(worldplane[0])
		squareworld = normalworld.transpose().dot(normalworld)
		
		# I think "parallel space" is the right term here - parallel to
		# the projection to the worldplane, so of course perpendicular to
		# the worldplane.
		parallelspace = np.array([[-1/phi,0,1,-1/phi,0,-1],
								  [1,-1/phi,0,-1,-1/phi,0],
								  [0,1,-1/phi,0,-1,-1/phi]])
		normallel = parallelspace / np.linalg.norm(parallelspace[0])
		squarallel = normallel.T.dot(normallel)
		
		# Slide worldplane origin to a point where it's perpendicular to the worldplane
		#a = a - a.dot(squarallel)
		a = a.dot(squarallel)
		print(a)
		
		deflation = np.array([[2, 1,-1,-1, 1, 1],
							  [1, 2, 1,-1,-1, 1],
							  [-1,1, 2, 1, -1,1],
							  [-1,-1,1, 2, 1, 1],
							  [1,-1,-1, 1, 2, 1],
							  [1, 1, 1, 1, 1, 2]])
		deflation_face_axes = [ [ 2, 1, 1, 1, 1,-1],
								[ 1, 2, 1,-1, 1, 1],
								[ 1, 1, 2, 1,-1, 1],
								[ 1,-1, 1, 2,-1,-1],
								[ 1, 1,-1,-1, 2, 1],
								[-1, 1, 1,-1,-1, 2]]
		
		
		# What we want is a list of every point in the 6D lattice which, in all
		# six dimensions, comes within a distance of 0.5 of the worldplane. In
		# other words, does the hypercube centered there intersect the worldplane?
		# My intuition fought this conclusion for awhile, but it seems to do this 
		# we have to  check for intersections with individual 3-faces. No one point
		# can be projected and checked, since there are arrangements which could make
		# the intersection occur elsewhere. Even just checking all corners doesn't work.
		# If there is a better solution than checking all faces like this, it would 
		# have to exploit the known 'angle' of the worldplane to our hypercube to
		# reduce the number of cases to check.
		
		ch3 = [[1,1,1,0,0,0],[1,1,0,1,0,0],[1,1,0,0,1,0],[1,1,0,0,0,1],[1,0,1,1,0,0],[1,0,1,0,1,0],
						[1,0,1,0,0,1],[1,0,0,1,1,0],[1,0,0,1,0,1],[1,0,0,0,1,1],[0,1,1,1,0,0],[0,1,1,0,1,0],
						[0,1,1,0,0,1],[0,1,0,1,1,0],[0,1,0,1,0,1],[0,1,0,0,1,1],[0,0,1,1,1,0],[0,0,1,1,0,1],
						[0,0,1,0,1,1],[0,0,0,1,1,1]]
		
		# A point is in our grid if the hypercube centered on that point intersects
		# the world-plane. We calculate this by projecting the hypercube into
		# parallel space, so it becomes a rhombic triacontahedron. 30 of the
		# hypercube's 480 2-faces form the outer boundary of this shape, and
		# the distance to the planes of these faces become the constraints for
		# inclusion.
		# Using the 30 constraints like that isn't the fastest algorithm, but
		# has huge simplicity advantages and can make good use of numpy.
		# If I had some reason to optimize this step, one approach would be to
		# eliminate and/or include a lot of points using faster checks, so that
		# this 30-fold computation would only run on what's left.
		print("Setting up 2-face geometry t="+str(time.perf_counter()-starttime))
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
		print("Ready to compute lattice at t="+str(time.perf_counter()-starttime))
		constraints = np.sum(np.stack(np.repeat([embedding_space
			.reshape((-1,6))-a],15,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals])
			.reshape((1,15,3))),axis=-1)
		distance_scores = np.max(np.abs(constraints),axis=1)
		# The expression at the end represents the distante to just one of the 
		# many 2-faces which fall on the boundary when projected.
		# It evaluates to 0.9732489894677302.
		# This seems to be correct after checking various a-values, but
		# geometrically I don't understand the division by 2.
		# Explanation: "constraints" computes distance from the origin to each
		# of the 6D points, within the parallel space, along the 30 directions
		# perpendicular to faces of a rhombic triacontahedron (since this is 
		# the shape of a hypercube projected into the 3D "parallel space").
		# This is half as much as the possible distance from the projected
		# 6D point to the plane representing one of these 30 faces, which is what
		# I was picturing.
		included = distance_scores < np.linalg.norm(np.array([0,0,1,-1,-1,-1]).dot(normallel.T))/2
		included = included.reshape((esize,esize,esize,esize,esize,esize))
		print("Latticepoints found in "+str(time.perf_counter()-starttime))
		deflation_included = distance_scores < np.linalg.norm(np.array([0,0,1,-1,-1,-1]).dot(normallel.T))/(2*phi*phi*phi)
		deflation_included = deflation_included.reshape((esize,esize,esize,esize,esize,esize))
		print("Chunks found in "+str(time.perf_counter()-starttime))
		
		# Need to know which lines are included, not which points; so we shift
		# the "included" array over by one in each dimension to compare it
		# with itself.
		
		# Each point present in the list line<n> represents a line starting at that
		# point and proceeding 1 unit in the nth positive direction.
		lines0 = embedding_space[np.nonzero(np.all([included[:esize-1],included[1:]],axis=0))]
		lines1 = embedding_space[np.nonzero(np.all([included[:,:esize-1],included[:,1:]],axis=0))]
		lines2 = embedding_space[np.nonzero(np.all([included[:,:,:esize-1],included[:,:,1:]],axis=0))]
		lines3 = embedding_space[np.nonzero(np.all([included[:,:,:,:esize-1],included[:,:,:,1:]],axis=0))]
		lines4 = embedding_space[np.nonzero(np.all([included[...,:esize-1,:],included[...,1:,:]],axis=0)) ]
		lines5 = embedding_space[np.nonzero(np.all([included[...,:esize-1],included[...,1:]],axis=0))]
		
		all_lines = [lines0,lines1,lines2,lines3,lines4,lines5]
		print("Lines found in "+str(time.perf_counter()-starttime))
		# Deflated lines are of distance 3 apart, having a distance of 1 in five
		# of their coordinates and a distance of 2 in one of them.
		deflated_lines = [[],[],[],[],[],[]]
		for i in embedding_space[deflation_included]:
			for j in embedding_space[deflation_included]:
				#print(np.linalg.norm(i - j))
				#print(np.sum(np.abs(i - j)))
				#print(np.where(i - j == 2)[0].shape == (1,))
				if np.linalg.norm(i - j) == 3:
					if np.sum(np.abs(i - j)) == 7:
						if np.where(i - j == 2)[0].shape == (1,):
							deflated_lines[np.where(i - j == 2)[0][0]].append(j)
		
		#Want to identify the blocks and chunks
		print("Deflated lines found in "+str(time.perf_counter()-starttime))
		blocks = np.zeros((0,6))
		for axes in ch3:
			ax1, ax2, ax3 = np.eye(6,dtype=np.int64)[np.nonzero(axes)[0]]
			#print([ax1,ax2,ax3])
			ax12 = ax1 + ax2
			ax13 = ax1 + ax3
			ax23 = ax2 + ax3
			ax123 = ax12 + ax3
			r1,r2,r3,r4,r5,r6,r7,r8 = (included[:esize-1,:esize-1,:esize-1,:esize-1,:esize-1,:esize-1],
				included[ax1[0]:esize-1+ax1[0],ax1[1]:esize-1+ax1[1],ax1[2]:esize-1+ax1[2],
					ax1[3]:esize-1+ax1[3],ax1[4]:esize-1+ax1[4],ax1[5]:esize-1+ax1[5]],
				included[ax2[0]:esize-1+ax2[0],ax2[1]:esize-1+ax2[1],ax2[2]:esize-1+ax2[2],
					ax2[3]:esize-1+ax2[3],ax2[4]:esize-1+ax2[4],ax2[5]:esize-1+ax2[5]],
				included[ax3[0]:esize-1+ax3[0],ax3[1]:esize-1+ax3[1],ax3[2]:esize-1+ax3[2],
					ax3[3]:esize-1+ax3[3],ax3[4]:esize-1+ax3[4],ax3[5]:esize-1+ax3[5]],
				included[ax12[0]:esize-1+ax12[0],ax12[1]:esize-1+ax12[1],ax12[2]:esize-1+ax12[2],
					ax12[3]:esize-1+ax12[3],ax12[4]:esize-1+ax12[4],ax12[5]:esize-1+ax12[5]],
				included[ax13[0]:esize-1+ax13[0],ax13[1]:esize-1+ax13[1],ax13[2]:esize-1+ax13[2],
					ax13[3]:esize-1+ax13[3],ax13[4]:esize-1+ax13[4],ax13[5]:esize-1+ax13[5]],
				included[ax23[0]:esize-1+ax23[0],ax23[1]:esize-1+ax23[1],ax23[2]:esize-1+ax23[2],
					ax23[3]:esize-1+ax23[3],ax23[4]:esize-1+ax23[4],ax23[5]:esize-1+ax23[5]],
				included[ax123[0]:esize-1+ax123[0],ax123[1]:esize-1+ax123[1],ax123[2]:esize-1+ax123[2],
					ax123[3]:esize-1+ax123[3],ax123[4]:esize-1+ax123[4],ax123[5]:esize-1+ax123[5]])
			nonzero = np.nonzero(np.all([r1,r2,r3,r4,r5,r6,r7,r8],axis=0))
			blocks = np.concatenate((blocks,embedding_space[nonzero]+np.array(ax123,dtype=np.float)/2))
#			for block in embedding_space[nonzero]:
#				blocks.append(block+np.array(ax123,dtype=np.float)/2)
		print("Found "+str(len(blocks))+" blocks. t="+str(time.perf_counter()-starttime))
		
		chunks = []
		for point in embedding_space[deflation_included]:
			# Find dimensions in which we have a neighbor in the positive direction
			pos_lines = set()
			for dim in range(6):
				if np.any(np.all(np.array(deflated_lines[dim]) - point == 0,axis=1)):
					pos_lines.add(dim)
			# Consider all sets of 3 such dimensions
			if len(pos_lines) >= 3:
				for i in pos_lines:
					for j in pos_lines - set([i]):
						for k in pos_lines - set([i,j]):
							# Check whether all eight cube corners exist with these axes
							try:
								if (deflation_included[tuple((point - offset)+deflation_face_axes[i]
														+deflation_face_axes[j])] and
									deflation_included[tuple((point - offset)+ deflation_face_axes[i]
														+deflation_face_axes[k])] and
									deflation_included[tuple((point - offset)+ deflation_face_axes[j]
														+deflation_face_axes[k])] and
									deflation_included[tuple((point - offset)+ deflation_face_axes[i]
														+deflation_face_axes[j]
														+deflation_face_axes[k])]):
									chunks.append(point+(np.array(deflation_face_axes[i])
										+np.array(deflation_face_axes[j])+np.array(deflation_face_axes[k]))/2)
							except IndexError:
								# If one of the indices was out of bound, it's not a chunk, so do nothing.
								pass
		print("Chunks computed. t="+str(time.perf_counter()-starttime)+" seconds")
		# TODO It seems that very occasionally, deflation_faces is empty at this
		# point; probably just means esize has to be greater than 8 to guarantee there are chunks.
		# Recording a-values where this happens: [ 0.03464434, -0.24607234,
		# -0.05386369,  0.2771699,   0.39596138,  0.45066235]
		
		# Choose a chunk near origin
		deflation_faces = np.array(chunks)
		chosen_center = deflation_faces[
			np.where(np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1)
			== np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1).min())[0][0]]
		#Rigging the process
		#chosen_center = np.array([0.5, 1.5, 0.,  0.,  1.5, 1. ])
		print("Chose chunk "+str(chosen_center)+" second="+str(time.perf_counter()-starttime))
		chosen_axes = 1-np.array(chosen_center - np.floor(chosen_center))*2
		chosen_origin = chosen_center - np.array(deflation_face_axes).T.dot(1-chosen_axes)/2
		chosen_axis1 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][0]])
		chosen_axis2 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][1]])
		chosen_axis3 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][2]])
		
		
		multiplier = 4
		array_mesh = ArrayMesh()
		self.mesh = array_mesh
		
		
		st = SurfaceTool()
		print("Drawing neighbor blocks next. seconds="+str(time.perf_counter()-starttime))
		st.begin(Mesh.PRIMITIVE_LINES)
		st.add_color(Color(0,.5,0))
		for block in blocks:
			face_origin = np.floor(block).dot(worldplane.T)*multiplier
			face_tip = np.ceil(block).dot(worldplane.T)*multiplier
			dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(worldplane.T)*multiplier
			corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = (
				face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
				face_tip - dir1, face_tip - dir2, face_tip - dir3
			)
			if np.any(np.abs((np.array(block).dot(worldplane.T)*multiplier - (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) > 0.5) and np.any(np.all(np.abs((np.array([
								corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8]) 
								- (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) < 0.5000001,axis=-1)):
				# Represents a voxel on the boundary of our chosen chunk
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
		print("Drawing inner blocks. "+str(time.perf_counter()-starttime))
		st.begin(Mesh.PRIMITIVE_LINES)
		st.add_color(Color(0,1,1))
		for block in blocks:
			if np.all(np.abs((np.array(block).dot(worldplane.T)*multiplier - (chosen_center)
								.dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) <= 0.5):
				# Represents a voxel inside our chosen chunk
				face_origin = np.floor(block).dot(worldplane.T)*multiplier
				face_tip = np.ceil(block).dot(worldplane.T)*multiplier
				dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(worldplane.T)*multiplier
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
#		st.begin(Mesh.PRIMITIVE_LINES)
#
#		for dim in range(6):
#			basis_element = np.zeros(6)
#			basis_element[dim] = 1
#			offset = basis_element.dot(worldplane.T)
#			for line in all_lines[dim]:
#				# Test if inside chosen chunk
#				point1 = line.dot(worldplane.T)
#				point2 = point1 + offset
#				point1 *= multiplier
#				point2 *= multiplier
#				if np.all(np.abs((np.array([point1,point2]) - (chosen_center).dot(worldplane.T)*multiplier)
#								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
#								*(phi*phi*phi*multiplier) ))) < 0.501):
#					st.add_vertex(Vector3(point1[0],point1[1],point1[2]))
#					st.add_vertex(Vector3(point2[0],point2[1],point2[2]))
#
#		st.commit(self.mesh)
		
		st.begin(Mesh.PRIMITIVE_LINES)
		
		st.add_color(Color(1,0.2,1))
		for dim in range(6):
			basis_element = np.zeros(6)
			basis_element[dim] = phi*phi*phi
			offset = basis_element.dot(worldplane.T)#*(-phi*phi*phi)
			for line in deflated_lines[dim]:
				point1 = line.dot(worldplane.T)
				point2 = point1 + offset
				point1 *= multiplier#*(-phi*phi*phi)
				point2 *= multiplier#*(-phi*phi*phi)
				if np.all(np.abs((np.array([point1,point2]) - (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) < 0.501):
					st.add_vertex(Vector3(point1[0],point1[1],point1[2]))
					st.add_vertex(Vector3(point2[0],point2[1],point2[2]))
		st.commit(self.mesh)
		
		self.mesh.surface_set_material(2,COLOR)
		
		latticepoints = embedding_space[included]
		print(latticepoints.shape)
		print("Done with rendering. t="+str(time.perf_counter()-starttime))
		
		# Now we want to calculate the validity bounds for this chunk.
		# Tentatively, only including vertices inside the chunk. I might want
		# to include all vertices of blocks which overlap the chunk, but it
		# seems like they'll be determined by the interior vertices.
		# OK, my constraints don't seem narrow enough to work so I'm guessing
		# this is the issue - for now I'm just going to widen the distance in
		# order to catch points nearby. Adding too many constraints doesn't seem
		# like it will do much harm - I will just end up with multiple identical
		# chunk templates. I can "glue together" the constraint regions.
		relevance = np.all(np.abs((embedding_space[included].dot(worldplane.T)*multiplier 
								- (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) < 1.5,axis=1)# Was "< 0.501"
		relevant_points = embedding_space[included][relevance]
		# Recalculating just because it's fast and easier than trying to get the
		# right numbers, arranged properly, out of the old "constraints" variable.
		constraints = np.sum(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		#print(np.max(constraints))
		print("Max constraint distance: "+str(np.max(np.abs(constraints))))#0.9731908764184711 max so far
		print("t="+str(time.perf_counter()-starttime))
		# The intersection of all the constraints takes the minima along each of the 30 vectors.
		overall_constraints = 0.9732489894677302 - np.max(constraints,axis=0)
		print("Wiggle room in all 30 directions:")
		print(overall_constraints)
		print("Wiggle room along the 15 axes:")
		print(overall_constraints[:15]+overall_constraints[15:])
		print("Proposing new point inside the constraints:")
		b = np.array([a[0]+r.random()/2-0.25,a[1]+r.random()/2-0.25,a[2]+r.random()/2-0.25,
					a[3]+r.random()/2-0.25,a[4]+r.random()/2-0.25,a[5]+r.random()/2-0.25])
		# TODO is it b-a or a-b?
		while np.any(np.concatenate([twoface_normals,-twoface_normals]).dot((a-b).dot(normallel.T)) > overall_constraints):
			b = np.array([a[0]+r.random()/2-0.25,a[1]+r.random()/2-0.25,a[2]+r.random()/2-0.25,
					a[3]+r.random()/2-0.25,a[4]+r.random()/2-0.25,a[5]+r.random()/2-0.25])
			b = b.dot(squarallel)
		print(b)
		constraints = np.sum(np.stack(np.repeat([relevant_points - b],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		print("New max constraint distance: "+str(np.max(np.abs(constraints))))
		print("Predicted new max: "+str(np.max(
			-0.9732489894677302 + np.max(np.sum(
				(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T) )
				* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3)))
			,axis=2),axis=0)
			+ 0.9732489894677302 + np.dot(
				np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),
				(a-b).dot(normallel.T)
			)
			)))
		print("Predicted to pass: "+str(np.all(
			+ np.dot(
				np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),
				(a-b).dot(normallel.T))
			< 0.9732489894677302 - np.max(np.sum(
				(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T) )
				* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3)))
			,axis=2),axis=0)
			)))
		#print(np.concatenate([twoface_normals,-twoface_normals]).dot((b-a).dot(normallel.T)))
		print("t="+str(time.perf_counter()-starttime))
		
		# Run a bunch of wiggle room tests
		# TODO The constraints seem correct in a few senses:
		# 1) They are the exact same numbers which get used to define the lattice
		# 	in the first place.
		# 2) They successfully constrain the points in "relevant_points" to occur
		#	in the lattice, in all tests so far.
		# 3) They identify the same set of chunks, in all tests so far.
		# Despite this, new offset parameters ("a") within the constraints 
		# sometimes identify _new points_ within the region used by "relevant_points".
		# In the loops below, where I repeatedly randomize within current constraints,
		# then derive new constraints, then randomize within those, etc, this causes
		# the constraints to become narrower over time as the set of points expands.
		# It seems as if some starting offsets never have this occur, while others
		# quickly find these new points. Note, as far as I've seen these extra
		# points only occur when certain values are used for the comparison which
		# defines "relevant_points" - I've used "0.5", "0.501", "1.0", and "1.5",
		# and all values besides "1.0" produced the anomolous extra points.
		# Conceptually, the new points make some sense since we only constrain all
		# present points to be in the new lattice. However, I'd thought it shouldn't
		# occur inside the chunk, since an old point has to fall off in order for a
		# new point to enter. (Between the two "a" values should lie some single
		# value where the new point is on the boundary between inclusion and exclusion;
		# and this is equivalent to some currently present point also being on
		# the boundary between inclusion and exclusion.) Maybe it could occur
		# closer to the edge. Yet I find that examining these new points, they
		# apparently fall mostly inside our chosen chunk. Sometimes they occur
		# exactly on the boundary, on a face of the chunk.
		# This seems to contradict claim (3) above. Any new point inside the chunk
		# would be a corner of several blocks, and so would change which blocks
		# were present.
		# THEREFORE If I ever suspect this to be a real problem, the first step
		# I reccommend is adding constraints which keep new points from entering
		# the grid. This shouldn't be difficult; excluded points generate one
		# constraint instead of two, but they're of the same form as the existing
		# constraints so they just need to be part of the existing selection of
		# closest constraining plane for a given direction.
		# If I want to fully diagnose this oddity: my best guess would be that
		# when I encode points as strings in order to throw them in sets,
		# something about the mapping is not 1-to-1. But if that were right, I
		# think I'd end up with "missing" points, so IDK. This theory also doesn't
		# line up with the constraints' being visibly narrower after the new point
		# is discovered; the new point has unique constraints and therefore a
		# unique location.
		# In principle, each constraint on a point comes with information about
		# what happens if we cross; EG, cross this plane and [0,-2,1,1,5,4] leaves
		# the grid, but [0,-2,1,2,5,4] arrives. This means we could check which 
		# point was "supposed to" be removed when a given novel point gets added.
		# Obviously we also have all the information needed to figure out which
		# chunks the new point associates with.
		# The most direct contradiction would seem to be from these points often
		# falling inside the chosen chunk, which makes the calculation of position
		# relative to the chunk suspect. But, it was directly pasted from working
		# code.
		# See also the commented printouts copied below the test.
		
		print()
		print()
		clean_test = True
		failures = 0
		seeds = 10
		repetitions = 4
		for i in range(seeds):
			print()
			print("New seed:")
			#b = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
			b, overall_constraints, chosen_center, resulting_blocks = self.chunk_test()
			for j in range(repetitions):
				wiggle_room = overall_constraints[:15]+overall_constraints[15:]
				b, new_constraints, new_center, new_blocks = self.chunk_test(b,chosen_center,constraints=overall_constraints)
				old_block_set = set()
				new_block_set = set()
				for point in resulting_blocks:
					old_block_set.add(repr(list(point)))
				for point in new_blocks:
					new_block_set.add(repr(list(point)))
				if old_block_set != new_block_set:
					# Yes, I refer to nonexistent variables in this block.
					# Block seems unreachable at present, and I may switch
					# back to testing points instead of blocks if I decide
					# to figure out why that test was failing.
					clean_test = False
					print("Warning, new points!")
					new_points_ndarrays = []
					for string in list(new_point_set - new_point_set.intersection(old_point_set)):
						new_points_ndarrays.append(np.array(eval(string)))
					print("Closest new point to chunk: "
						+str(np.min(np.abs((
							np.array(new_points_ndarrays)
								.dot(worldplane.T)*multiplier 
								- (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))))))
					print("Novel points:"+str(len(new_point_set - new_point_set.intersection(old_point_set)))+"/"+str(len(new_point_set)))
					print(new_point_set - new_point_set.intersection(old_point_set))
					if len(old_point_set - old_point_set.intersection(new_point_set)) > 0:
						failures += 1
						print("Missing points: "+str(len(old_point_set 
							- old_point_set.intersection(new_point_set)))+"/"+str(len(old_point_set)))
				else:
					# We want to save the chunk.
		if clean_test:
			print("Constraints seem very solid!")
		if not clean_test and failures == 0:
			print("Constraints were OK on existing points, but didn't constrain possible new points")
		if failures > 1:
			print("Test failed "+str(failures)+" times out of "+str(seeds*repetitions)+".")
		print("t="+str(time.perf_counter()-starttime))
		
		# We now have all the constraints for identifying when to use this chunk template.
		# (It might also be handy that the constraints are overly fine-grained; we're practically
		# constraining entire neighboring chunks too.) The next thing is to package the chunk
		# (so, the full block arrangement) in some convenient format along with the constraints,
		# and then systematically generate every possible chunk, so that the union of our constraint
		# regions covers the whole 6D cube of possible a-values (well, actually the constraint
		# regions are in parallel space, so a triacontahedron.). They can then serve as a lookup table
		# for hierarchical generation. The chunk hierarchy itself will be tricky to program, but
		# the remaining math is just the conversion between block-coordinates and chunk-coordinates
		# (so that we know which a-value to use for generation when cutting a large chunk into
		# sub-chunks).
		# Hrm, the a-value alone isn't enough information. To speficy a specific chunk
		# layout you need to know the a-value and which chunk (IE, the chosen_center value).
		# This means the resulting table needs entries for all chosen_center values which might
		# occur; specifically, I should apply the block-to-chunk mapping to every block in
		# every chunk, in order to extract its equivalent chosen_center. However, applied
		# recursively, this would just result in endless expansion outwards; and obviously
		# these new chunks don't have novel configurations. So then, what's the right 
		# mapping from arbitrary (a, chosen_center) pairs to a canonical set of
		# chosen_center values? Well - starting with truly arbitrary pairs, we can
		# reduce each coordinate of a to be within a unit interval centered on the
		# origin; adding or subtracting 1 from both a and chosen_center at the same
		# time doesn't affect the shape. However, in order for our chunk finding 
		# algorithm to work, we restrict a to be perpendicular to the worldplane.
		# This means that rather than just moving a by unit amounts, we actually
		# must first move it by the unit and then slide it along the worldplane
		# to the new closest point to the origin. This changes 3D coordinates of
		# blocks and chunks, but not their 6D coordinates, so the appropriate
		# chosen_center value is only changed by the first part of the motion.
		# The best way of thinking of this combined motion is, we move the whole
		# worldplane by an integer amount, and this changes the value of "a" in
		# accordance with the worldplane's change in distance from the origin.
		# If we have integer vectors which are approximately parallel to the 
		# worldplane, we can move the worldplane by that amount with relatively
		# little change in the value of a. But, the lattice itself is a rich
		# source of such vectors. This means that if we restrict a to fall in a
		# particular unit cube, we still can translate most (or at least many)
		# points included in our lattice between one another. So a good 
		# canonical choice of chunk templates might, say, ensure that [0,0,0,
		# 0,0,0] is a corner of every chunk, or maybe arrange the chunk centers
		# around the origin in a specific symmetrical way. And really, we can
		# just move the chosen chunk to its assigned place, make the necessary
		# adjustment to a, and then whatever range of a-values results can be
		# our set of canonical values for a. (Hopefully there's a good way to
		# keep them in one cube.)
"""
[-0.09034036  0.04535047 -0.11384384  0.28502651  0.18663901  0.20116496]
Chose chunk [0.  0.5 0.  0.  0.5 0.5] second=2.5996314999999868
Wiggle room in all 30 directions:
[0.0218368  0.00485282 0.02454847 0.0140518  0.04457309 0.03391155
 0.03469994 0.00142113 0.00048725 0.04176359 0.02920158 0.05972287
 0.04987174 0.00627395 0.00936308 0.03240051 0.04938448 0.02968883
 0.04018551 0.00966422 0.02032575 0.01953736 0.03209937 0.03303325
 0.01247372 0.02503572 0.02803493 0.00436557 0.02724654 0.01135373]
Wiggle room along the 15 axes:
[0.05423731 0.05423731 0.05423731 0.05423731 0.05423731 0.05423731
 0.05423731 0.0335205  0.0335205  0.05423731 0.05423731 0.08775781
 0.05423731 0.0335205  0.02071681]
Proposed new point inside the constraints:
[-0.10429039  0.0408611  -0.10620435  0.29972654  0.20883509  0.21001799]
New max constraint distance: 0.973120320270533
Predicted new max: 0.9731203202705331
Predicted to pass: True
[-0.10429039  0.0408611  -0.10620435  0.29972654  0.20883509  0.21001799]
Chose chunk [0.  0.5 0.  0.  0.5 0.5] second=2.5782736999999827
Wiggle room in all 30 directions:
[0.00225823 0.03066252 0.00236608 0.0199209  0.03449092 0.00324151
 0.05264632 0.00041238 0.03053385 0.05285451 0.02133738 0.0359074
 0.05410864 0.0310749  0.00087543 0.05197908 0.02357479 0.05187123
 0.03431641 0.01974639 0.0509958  0.00159099 0.03310812 0.00298665
 0.0013828  0.03289993 0.01832991 0.00012867 0.0024456  0.01984138]
Wiggle room along the 15 axes:
[0.05423731 0.05423731 0.05423731 0.05423731 0.05423731 0.05423731
 0.05423731 0.0335205  0.0335205  0.05423731 0.05423731 0.05423731
 0.05423731 0.0335205  0.02071681]
Proposed new point inside the constraints:
[-0.09973368  0.04201567 -0.10863546  0.29513268  0.20202516  0.20752676]
New max constraint distance: 0.9727866410073203
Predicted new max: 0.9727866410073205
Predicted to pass: True
Warning, new points!
Closest new point to chunk: 0.5
Novel points:4/2032
{'[-2, -2, -1, 2, -1, 0]', '[-2, -1, 1, 2, -2, 1]', '[-1, -1, 1, 3, -2, 1]', '[-3, -2, -1, 1, -1, 0]'}

[-0.04214969 -0.0804242  -0.40199395  0.7990637   0.60032266  0.82271865]
Chunks computed. t=2.554683600000004 seconds
Chose chunk [ 0.5  1.  -0.5  0.   1.5  1. ] second=2.5550733999999977
Wiggle room in all 30 directions:
[0.05112147 0.034144   0.03562749 0.02513483 0.03755318 0.03975657
 0.01838879 0.0094392  0.02031448 0.00724492 0.03181584 0.04423418
 0.00688729 0.0228664  0.00412909 0.00311584 0.02009331 0.01860982
 0.00838566 0.01668413 0.01448073 0.06936902 0.0447981  0.06744333
 0.04699238 0.02242147 0.01000312 0.04735002 0.06489141 0.05010822]
Wiggle room along the 15 axes:
[0.05423731 0.05423731 0.05423731 0.0335205  0.05423731 0.05423731
 0.08775781 0.05423731 0.08775781 0.05423731 0.05423731 0.05423731
 0.05423731 0.08775781 0.05423731]
Proposed new point inside the constraints:
[-0.04329696 -0.08588073 -0.40043335  0.80385145  0.60944723  0.83016967]
New max constraint distance: 0.9716017171127801
Predicted new max: 0.9716017171127801
Predicted to pass: True
[-0.04329696 -0.08588073 -0.40043335  0.80385145  0.60944723  0.83016967]
Chose chunk [ 0.5  1.  -0.5  0.   1.5  1. ] second=2.5473968000000013
Wiggle room in all 30 directions:
[0.04094795 0.04792252 0.02350689 0.02781741 0.03172016 0.0281919
 0.02063971 0.00454277 0.02885299 0.01797437 0.00187743 0.03930068
 0.00164727 0.03174849 0.00468501 0.01328936 0.00631478 0.03073042
 0.00570308 0.02251714 0.0260454  0.06711809 0.04969454 0.05890482
 0.03626294 0.01883938 0.01493663 0.05259003 0.05600932 0.04955229]
Wiggle room along the 15 axes:
[0.05423731 0.05423731 0.05423731 0.0335205  0.05423731 0.05423731
 0.08775781 0.05423731 0.08775781 0.05423731 0.02071681 0.05423731
 0.05423731 0.08775781 0.05423731]
Proposed new point inside the constraints:
[-0.0379087  -0.08075999 -0.4057313   0.7985846   0.59774128  0.82506816]
New max constraint distance: 0.9718400752149063
Predicted new max: 0.9718400752149063
Predicted to pass: True
Warning, new points!
Closest new point to chunk: 0.4999999999999998
Novel points:2/1897
{'[2, -4, -1, 4, 1, -3]', '[3, -3, -1, 4, 1, -3]'}
[-0.0379087  -0.08075999 -0.4057313   0.7985846   0.59774128  0.82506816]
Chose chunk [ 0.5  1.  -0.5  0.   1.5  1. ] second=2.520566599999995
Wiggle room in all 30 directions:
[0.04956669 0.03395126 0.03749414 0.02784327 0.04038074 0.04230061
 0.01175676 0.00437948 0.01464336 0.00947709 0.00209981 0.04815777
 0.00140891 0.01761393 0.00480647 0.00467062 0.02028605 0.01674317
 0.00567723 0.01385657 0.0119367  0.07600104 0.04985783 0.07311444
 0.04476022 0.018617   0.00607953 0.05282839 0.07014388 0.04943084]
Wiggle room along the 15 axes:
[0.05423731 0.05423731 0.05423731 0.0335205  0.05423731 0.05423731
 0.08775781 0.05423731 0.08775781 0.05423731 0.02071681 0.05423731
 0.05423731 0.08775781 0.05423731]
Proposed new point inside the constraints:
[-0.0441522  -0.09743448 -0.38699097  0.79503878  0.61608302  0.82796904]
New max constraint distance: 0.9706209514212343
Predicted new max: 0.9706209514212343
Predicted to pass: True
Warning, new points!
Closest new point to chunk: 0.4999999999999998
Novel points:2/1897
{'[2, -4, -1, 4, 1, -3]', '[3, -3, -1, 4, 1, -3]'}
[-0.0441522  -0.09743448 -0.38699097  0.79503878  0.61608302  0.82796904]
Chose chunk [ 0.5  1.  -0.5  0.   1.5  1. ] second=2.5837237000000073
Wiggle room in all 30 directions:
[0.0501028  0.05160927 0.00838677 0.00931782 0.01094204 0.03000145
 0.03252467 0.01754608 0.00155944 0.02574919 0.0107706  0.04591532
 0.00418748 0.01491804 0.02161468 0.00413451 0.00262804 0.04585054
 0.02420268 0.04329527 0.02423586 0.05523314 0.03669123 0.05267787
 0.02848812 0.0099462  0.00832199 0.05004983 0.03931926 0.03262263]
Wiggle room along the 15 axes:
[0.05423731 0.05423731 0.05423731 0.0335205  0.05423731 0.05423731
 0.08775781 0.05423731 0.05423731 0.05423731 0.02071681 0.05423731
 0.05423731 0.05423731 0.05423731]
Proposed new point inside the constraints:
[-0.04737544 -0.09025126 -0.39071531  0.79909699  0.61339999  0.8255957 ]
New max constraint distance: 0.9712404992064727
Predicted new max: 0.9712404992064722
Predicted to pass: True
Warning, new points!
Closest new point to chunk: 0.027864045000420522
Novel points:18/1913
{'[-3, 2, -1, -3, 2, 4]', '[-3, 0, -2, -2, 2, 2]', '[-1, 0, -3, -1, 4, -1]', '[1, 4, 2, -2, 3, 3]',
'[-2, 3, -1, -3, 3, 3]', '[2, -4, -1, 4, 1, -3]', '[1, 4, 3, -1, 2, 4]', '[-1, 2, -2, -2, 4, 1]',
'[-1, 3, -1, -3, 4, 3]', '[2, 3, 3, 1, 2, 2]', '[2, 3, 1, -1, 4, 1]', '[-2, -1, -3, -1, 3, 0]',
'[3, 2, 1, 1, 4, -1]', '[2, 3, 2, 0, 3, 2]', '[-2, 1, -2, -2, 3, 2]', '[3, 2, 2, 2, 3, 0]',
'[3, -3, -1, 4, 1, -3]', '[1, 4, 1, -2, 4, 3]'}

"""
