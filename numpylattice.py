from godot import exposed, export
from godot import *
import numpy as np
from debugging import debugging
import random as r

COLOR = ResourceLoader.load("res://new_spatialmaterial.tres")

@exposed(tool=True)
class numpylattice(MeshInstance):
#	phi = 1.61803398874989484820458683
#	parallelspace = np.array([[-1/phi,0,1,-1/phi,0,-1],
#								  [1,-1/phi,0,-1,-1/phi,0],
#								  [0,1,-1/phi,0,-1,-1/phi]])
#	normallel = parallelspace / np.linalg.norm(parallelspace[0])
#	squarallel = normallel.T.dot(normallel)
	
#	def existence_boundaries(points):
#		""" Takes an ndarray of points in R^6 and returns an ndarray of bounds on
#		 the worldplane offset which make each point part of the lattice.
#		"""
#		# We assume each point is currently part of the lattice, and give
#		# bounds in both positive and negative direction.
#		pass
	
	def _ready(self):
		phi = 1.61803398874989484820458683
		# Set up a 6D array holding 6-vectors which are their own coords
		# w/in the array
		esize = 10
		offset = -5
		embedding_space = np.zeros((esize,esize,esize,esize,esize,esize,6),dtype=np.int8)
		for index,x in np.ndenumerate(embedding_space[...,0]):
			embedding_space[index] = np.array(index) + offset
		# "a" is the origin of the worldplane. Just want it to be nonsingular,
		# ie, worldplane shouldn't intersect any 6D integer lattice points.
		# Also want worldplane to intersect at least some of the values we
		# actually check (so, inside the range of the embedding_space).
		#a = np.array([5.1+phi/8,5.2+phi/8,5.3,5.4,5.5,5.6])
		a = np.array([0.1201,0.61102,0.1003,0.60904,0.0805,0.60706])
		a -= np.array([0.26201,0.0611,0.15003,0.16094,0.12805,0.20706])
		a = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
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
		
		
#		included = np.zeros_like(embedding_space[...,0],dtype=bool)
#		deflation_included = np.zeros_like(embedding_space[...,0],dtype=bool)
		
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
		# There are twenty 3-face orientations, corresponding to the list above.
		# Each 6-cube has eight 3-faces of a given orientation, for the eight
		# possible boundary settings of its three fixed dimensions. But each
		# 3-face also bounds eight different hypercubes, so we can check
		# every hypercube by systematically choosing one of those eight
		# 3-faces.
#		intersections = []
#		faces = []
#		deflation_faces = []
#		for axes in ch3:
#			# Project axes with 1s into parallel space
#			#facevectors = parallelspace.T[np.nonzero(axes)[0]].T
#			# To make this a basis transform, scale correctly
#			#facevectors = facevectors / np.linalg.norm(facevectors[0])
#			# Then just represent the chosen starting corner in that basis
#			# and check the distance to origin in all 3 directions.
#			# (embedding_space-a-np.ones(6)/2) is the chosen corner, represented
#			# by its vector from the worldplane center a.
#			# (embedding_space-a-np.ones(6)/2).dot(normallel.T) is the parallel-space
#			# component of this vector, IE, the vector to the chosen corner from
#			# the nearest point on the worldplane. Check: It can be added to the 
#			# projection onto the worldplane to recover the original.
#			# (embedding_space-a-np.ones(6)/2).dot(normallel.T)
#			#			.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]))
#			# represents the chosen corner's position in terms of the three vectors
#			# which define our 3-face. Check: moving the original point by 1 along 
#			# an axis of our current 3-face moves the result by 1 in the corresponding
#			# axis.
#			corners = np.all(np.abs((embedding_space-a-np.ones(6)/2).dot(normallel.T)
#							.dot( np.linalg.inv(normallel.T[np.nonzero(axes)[0]]) )+np.ones(3)/2)<0.5,axis=6)
#
#			deflation_corners1 = np.all(np.abs((embedding_space-a-np.ones(6)/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			deflation_corners2 = np.all(np.abs((embedding_space-a-(np.ones(6)
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]])/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			deflation_corners3 = np.all(np.abs((embedding_space-a-(np.ones(6)
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]])/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			deflation_corners4 = np.all(np.abs((embedding_space-a-(np.ones(6)
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			deflation_corners5 = np.all(np.abs((embedding_space-a-(np.ones(6)
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]]
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]])/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			deflation_corners6 = np.all(np.abs((embedding_space-a-(np.ones(6)
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]]
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			deflation_corners7 = np.all(np.abs((embedding_space-a-(np.ones(6)
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]]
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			deflation_corners8 = np.all(np.abs((embedding_space-a-(np.ones(6)
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]]
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]]
#							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
#							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
#			included = included + corners
#			deflation_included = (deflation_included + deflation_corners1 + deflation_corners2 + deflation_corners3
#								+ deflation_corners4 + deflation_corners5 + deflation_corners6
#								+ deflation_corners7 + deflation_corners8)
#
#			fixedvectors = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
#									[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
#			fixedvectors[[0,1,2],np.nonzero(1-np.array(axes))[0]] = 1
#			fixedvectors[3] = fixedvectors[0] + fixedvectors[1]
#			fixedvectors[4] = fixedvectors[0] + fixedvectors[2]
#			fixedvectors[5] = fixedvectors[1] + fixedvectors[2]
#			fixedvectors[6] = fixedvectors[0] + fixedvectors[1] + fixedvectors[2]#1-np.array(axes)
#			for fv in fixedvectors:
#				included = included + np.pad(corners[fv[0]:,fv[1]:,
#							fv[2]:,fv[3]:,fv[4]:,fv[5]:],((0,fv[0]),(0,fv[1]),(0,fv[2]),(0,fv[3]),(0,fv[4]),(0,fv[5])),
#							'constant',constant_values=(False))
		
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
		# TODO The expression at the end evaluates to 0.9732489894677302
		# This seems to be correct after checking various a-values, but
		# geometrically I don't understand the division by 2. 
		included = np.max(np.abs(np.sum(np.stack(np.repeat([embedding_space
			.reshape((-1,6))-a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals])
			.reshape((1,30,3))),axis=-1)),axis=1) <  np.linalg.norm(np.array([0,0,1,-1,-1,-1])
			.dot(normallel.T))/2
		included = included.reshape((esize,esize,esize,esize,esize,esize))
		
		deflation_included = np.max(np.abs(np.sum(np.stack(np.repeat([embedding_space
			.reshape((-1,6))-a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals])
			.reshape((1,30,3))),axis=-1)),axis=1) <  np.linalg.norm(np.array([0,0,1,-1,-1,-1])
			.dot(normallel.T))/(2*phi*phi*phi)
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
		
		#Want to identify the blocks and chunks
		
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
		
		# TODO It seems that very occasionally, deflation_faces is empty at this
		# point; probably just means esize has to be greater than 8 to guarantee there are chunks.
		# Recording a-values where this happens: [ 0.03464434, -0.24607234,
		# -0.05386369,  0.2771699,   0.39596138,  0.45066235]

		
		# Choose a chunk near origin
		deflation_faces = np.array(chunks)
		chosen_center = deflation_faces[
			np.where(np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1)
			== np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1).min())[0][0]]
		print("Chose chunk "+str(chosen_center))
		chosen_axes = 1-np.array(chosen_center - np.floor(chosen_center))*2
		chosen_origin = chosen_center - np.array(deflation_face_axes).T.dot(1-chosen_axes)/2
		chosen_axis1 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][0]])
		chosen_axis2 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][1]])
		chosen_axis3 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][2]])
		
		
		multiplier = 4
		array_mesh = ArrayMesh()
		self.mesh = array_mesh
		
		
		st = SurfaceTool()
		
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
		
		st.begin(Mesh.PRIMITIVE_LINES)
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
		
		st.add_color(Color(1,0,1))
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
		"""
		# Now we want to calculate the validity bounds for this chunk.
		"""
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
		relevance = np.all(np.abs((embedding_space[included].dot(worldplane.T)*multiplier 
								- (chosen_center).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*multiplier) ))) < 0.501,axis=1)
		relevant_points = embedding_space[included]#[relevance]
		constraints = np.sum(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		#print(np.max(constraints))
		print(np.max(np.abs(constraints)))#0.9731908764184711 max so far
		
		# Checking an alternate inclusion criterion
#		# TODO The expression at the end evaluates to 0.9732489894677302
#		# This seems to be correct after checking various a-values, but
#		# geometrically I don't understand the division by 2. 
#		included_test = np.max(np.abs(np.sum(np.stack(np.repeat([embedding_space
#			.reshape((-1,6))-a],30,axis=0),axis=1).dot(normallel.T)
#			* np.concatenate(np.array([twoface_normals,-twoface_normals])
#			.reshape((1,30,3))),axis=-1)),axis=1) <  np.linalg.norm(np.array([0,0,1,-1,-1,-1]).dot(normallel.T))/2
#		print(str(included_test.shape) + " vs " + str(included.flatten().shape))
#		print(embedding_space.reshape((-1,6))[included_test].shape)
#		print("Missing "+str(embedding_space[included].shape[0] - 
#			embedding_space.reshape((-1,6))[included_test].shape[0])+" points captured by old algorithm.")
#		print(np.array(np.nonzero(included_test == included.flatten())).shape[1])
#		print("Disagreeing on " + str(
#			included_test.shape[0] - np.array(np.nonzero(included_test == included.flatten())).shape[1])+" points.")
#		if (included_test.shape[0] - np.array(np.nonzero(included_test == included.flatten())).shape[1]) > 0:
#			# Calculate exact gap for points of difference
#			full_constraints = np.max(np.abs(np.sum(np.stack(np.repeat([embedding_space
#				.reshape((-1,6))-a],30,axis=0),axis=1).dot(normallel.T)
#				* np.concatenate(np.array([twoface_normals,-twoface_normals])
#				.reshape((1,30,3))),axis=-1)),axis=1)
#			print("Points novel to new test:")
#			print(full_constraints[np.logical_and(included_test != included.flatten(),included_test)])
#			print(np.array(np.nonzero(np.logical_and(included_test.reshape((esize,esize,esize,esize,esize,esize))
#				 != included,included_test.reshape((esize,esize,esize,esize,esize,esize)) ))).T)
#			print("Old points missing in new test:")
#			print(full_constraints[np.logical_and(included_test != included.flatten(),included.flatten())])
