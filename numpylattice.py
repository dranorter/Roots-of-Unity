from godot import exposed, export
from godot import *
import numpy as np
from debugging import debugging

COLOR = ResourceLoader.load("res://new_spatialmaterial.tres")

@exposed(tool=True)
class numpylattice(MeshInstance):
	
	def _ready(self):
		phi = 1.61803398874989484820458683
		# Set up a 6D array holding 6-vectors which are their own coords
		# w/in the array
		esize = 8
		offset = -4
		embedding_space = np.zeros((esize,esize,esize,esize,esize,esize,6),dtype=np.int8)
		for index,x in np.ndenumerate(embedding_space[...,0]):
			embedding_space[index] = np.array(index) + offset
		# "a" is the origin of the worldplane. Just want it to be nonsingular,
		# ie, worldplane shouldn't intersect any 6D integer lattice points.
		# Also want worldplane to intersect at least some of the values we
		# actually check (so, inside the range of the embedding_space).
		#a = np.array([5.1+phi/8,5.2+phi/8,5.3,5.4,5.5,5.6])
		a = np.array([0.1201,0.61102,0.1003,0.60904,0.0805,0.60706])
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
		
		
		included = np.zeros_like(embedding_space[...,0],dtype=bool)
		deflation_included = np.zeros_like(embedding_space[...,0],dtype=bool)
		
		# What we want is a list of every point in the 6D lattice which, in all
		# six dimensions, comes within a distance of 0.5 of the worldplane. In
		# other words, does the hypercube centered there intersect the worldplane?
		# My intuition fought this conclusion for awhile, but it seems to do this 
		# we have to  check for intersections with individual 3-faces.
		# (One more scheme for getting around this: The problem seems to be that
		# projecting the hypercube into parallel space, the metric of the 
		# embedding space is lost, and the hypercube forms a complex shape (a
		# triacontahedron) rather than a nice easy shape. What about instead
		# finding the closest point in the world-plane, but then projecting that
		# up? I already tried something where I projected the corners down and then
		# up, but this would be slightly different.)
		
		
		
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
		# For debugging:
		intersections = []
		faces = []
		deflation_faces = []
		for axes in ch3:
			# Project axes with 1s into parallel space
			#facevectors = parallelspace.T[np.nonzero(axes)[0]].T
			# To make this a basis transform, scale correctly
			#facevectors = facevectors / np.linalg.norm(facevectors[0])
			# Then just represent the chosen starting corner in that basis
			# and check the distance to origin in all 3 directions.
			# (embedding_space-a-np.ones(6)/2) is the chosen corner, represented
			# by its vector from the worldplane center a.
			# (embedding_space-a-np.ones(6)/2).dot(normallel.T) is the parallel-space
			# component of this vector, IE, the vector to the chosen corner from
			# the nearest point on the worldplane. Check: It can be added to the 
			# projection onto the worldplane to recover the original.
			# (embedding_space-a-np.ones(6)/2).dot(normallel.T)
			#			.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]))
			# represents the chosen corner's position in terms of the three vectors
			# which define our 3-face. Check: moving the original point by 1 along 
			# an axis of our current 3-face moves the result by 1 in the corresponding
			# axis.
			corners = np.all(np.abs((embedding_space-a-np.ones(6)/2).dot(normallel.T)
							.dot( np.linalg.inv(normallel.T[np.nonzero(axes)[0]]) )+np.ones(3)/2)<0.5,axis=6)
			deflation_corners1 = np.all(np.abs((embedding_space-a-np.ones(6)/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			deflation_corners2 = np.all(np.abs((embedding_space-a-(np.ones(6)
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]])/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			deflation_corners3 = np.all(np.abs((embedding_space-a-(np.ones(6)
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]])/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			deflation_corners4 = np.all(np.abs((embedding_space-a-(np.ones(6)
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			deflation_corners5 = np.all(np.abs((embedding_space-a-(np.ones(6)
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]]
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]])/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			deflation_corners6 = np.all(np.abs((embedding_space-a-(np.ones(6)
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]]
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			deflation_corners7 = np.all(np.abs((embedding_space-a-(np.ones(6)
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]]
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			deflation_corners8 = np.all(np.abs((embedding_space-a-(np.ones(6)
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][0]]
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][1]]
							-2*np.eye(6)[np.nonzero(1-np.array(axes))[0][2]])/(2*phi*phi*phi)).dot(normallel.T)
							.dot(np.linalg.inv(normallel.T[np.nonzero(axes)[0]]/(phi*phi*phi)))+np.ones(3)/2)<0.5,axis=6)
			included = included + corners
			deflation_included = (deflation_included + deflation_corners1 + deflation_corners2 + deflation_corners3
								+ deflation_corners4 + deflation_corners5 + deflation_corners6
								+ deflation_corners7 + deflation_corners8)
			# for debugging
			#intersections.append( (embedding_space-a-np.ones(6)/2).dot(normallel.T)
			#				.dot( np.linalg.inv(normallel.T[np.nonzero(axes)[0]]))
			#				.dot(np.eye(6)[np.nonzero(np.array(axes))[0]]))
			
			fixedvectors = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
									[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
			fixedvectors[[0,1,2],np.nonzero(1-np.array(axes))[0]] = 1
			fixedvectors[3] = fixedvectors[0] + fixedvectors[1]
			fixedvectors[4] = fixedvectors[0] + fixedvectors[2]
			fixedvectors[5] = fixedvectors[1] + fixedvectors[2]
			fixedvectors[6] = 1-np.array(axes)
			for fv in fixedvectors:
				included = included + np.pad(corners[fv[0]:,fv[1]:,
							fv[2]:,fv[3]:,fv[4]:,fv[5]:],((0,fv[0]),(0,fv[1]),(0,fv[2]),(0,fv[3]),(0,fv[4]),(0,fv[5])),
							'constant',constant_values=(False))
			for face in (embedding_space[corners] + np.array(1-np.array(axes))/2):
				faces.append(face)
			for face in (embedding_space[deflation_corners1] + np.array(1-np.array(axes))*(3.0/2)):
				deflation_faces.append(face)
		
		
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
				
		#Want to identify the chunks
		
		
		
		# Choose a chunk near origin
		deflation_faces = np.array(deflation_faces)
		chosen_center = deflation_faces[
			np.where(np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1)
			== np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1).min())[0][0]]
		chosen_origin = np.floor(chosen_center)
		chosen_axes = np.array((chosen_center - chosen_origin)*2)
		chosen_origin = chosen_origin - chosen_axes
		chosen_axis1 = np.array(np.eye(6)[np.nonzero(chosen_axes)[0][0]])
		chosen_axis2 = np.array(np.eye(6)[np.nonzero(chosen_axes)[0][1]])
		chosen_axis3 = np.array(np.eye(6)[np.nonzero(chosen_axes)[0][2]])
		print(chosen_center)
		
		multiplier = 4
		array_mesh = ArrayMesh()
		self.mesh = array_mesh
		st = SurfaceTool()
#		st.begin(Mesh.PRIMITIVE_LINES)
#
#		multiplier = 4
#		st.add_color(Color(1,0,1))
#		for dim in range(6):
#			basis_element = np.zeros(6)
#			basis_element[dim] = 1
#			offset = basis_element.dot(worldplane.T)#*(-phi*phi*phi)
#			for line in all_lines[dim]:
#				point1 = line.dot(worldplane.T)
#				point2 = point1 + offset
#				point1 *= multiplier*(-phi*phi*phi)
#				point2 *= multiplier*(-phi*phi*phi)
#				#if True:#np.linalg.norm(point1 - np.array([-27.41640786 -16.94427191  -0.])) < 20 or 
#				#           np.linalg.norm(point2 - np.array([-27.41640786 -16.94427191  -0.])) < 20:
#				if np.all(np.abs((np.array([point1,point2])/(multiplier*np.linalg.norm(worldplane[1])))
#									.dot(np.linalg.inv(normalworld.T[np.nonzero(chosen_axes)[0]]*
#									(-phi*phi*phi))) - 0.5) < .501):
#					st.add_vertex(Vector3(point1[0],point1[1],point1[2]))
#					st.add_vertex(Vector3(point2[0],point2[1],point2[2]))
#		st.commit(self.mesh)
		
		
		st = SurfaceTool()
		st.begin(Mesh.PRIMITIVE_LINES)
		
		for dim in range(6):
			basis_element = np.zeros(6)
			basis_element[dim] = 1
			offset = basis_element.dot(worldplane.T)
			for line in all_lines[dim]:
				# Test if inside chosen chunk
				point1 = line.dot(worldplane.T)
				point2 = point1 + offset
				point1 *= multiplier
				point2 *= multiplier
				#np.linalg.norm(point1 - np.array([-27.41640786 -16.94427191  -0.])) < 20 or 
				#          np.linalg.norm(point2 - np.array([-27.41640786 -16.94427191  -0.])) < 20:
				#True:#5*multiplier < point1[1] < 10*multiplier:
				if np.all(np.abs((np.array([point1,point2]) - (chosen_center+chosen_axes*.67).dot(worldplane.T)*multiplier)
								.dot(np.linalg.inv(normalworld.T[np.nonzero(chosen_axes)[0]]
								*(phi*phi*phi*np.linalg.norm(worldplane[1])) ))) < 2.351):
					st.add_vertex(Vector3(point1[0],point1[1],point1[2]))
					st.add_vertex(Vector3(point2[0],point2[1],point2[2]))
		
		st.commit(self.mesh)
		
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
				#if True:#np.linalg.norm(point1 - np.array([-27.41640786 -16.94427191  -0.])) < 20 or 
				#           np.linalg.norm(point2 - np.array([-27.41640786 -16.94427191  -0.])) < 20:
				if True:#np.all(np.abs((np.array([point1,point2])/(multiplier*np.linalg.norm(worldplane[1])))
					#				.dot(np.linalg.inv(normalworld.T[np.nonzero(chosen_axes)[0]]*
					#				(-phi*phi*phi))) - 0.5) < .8):
					st.add_vertex(Vector3(point1[0],point1[1],point1[2]))
					st.add_vertex(Vector3(point2[0],point2[1],point2[2]))
		st.commit(self.mesh)
		
		self.mesh.surface_set_material(1,COLOR)
		
		debugging.breakpoint()
		
		latticepoints = embedding_space[included]
#		st = SurfaceTool()
#		st.begin(Mesh.PRIMITIVE_POINTS)
#		for point in latticepoints:
#			worldpoint = multiplier*point.dot(worldplane.T)
#			st.add_vertex(Vector3(worldpoint[0],worldpoint[1],worldpoint[2]))
			
		#array_mesh = ArrayMesh()
		#st.commit(array_mesh)
		#self.mesh = array_mesh
		print(latticepoints.shape)
