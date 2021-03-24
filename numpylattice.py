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
		esize = 9
		offset = -5
		embedding_space = np.zeros((esize,esize,esize,esize,esize,esize,6),dtype=np.int8)
		for index,x in np.ndenumerate(embedding_space[...,0]):
			embedding_space[index] = np.array(index) + offset
		# "a" is the origin of the worldplane. Just want it to be nonsingular,
		# ie, worldplane shouldn't intersect any 6D integer lattice points.
		# Also want worldplane to intersect at least some of the values we
		# actually check (so, inside the range of the embedding_space).
		#a = np.array([5.1+phi/8,5.2+phi/8,5.3,5.4,5.5,5.6])
		a = np.array([5.12,5.12,5.12,5.12,5.12,5.12])
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
		a = a - a.dot(squarallel)
		
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
			included = included + corners
			fixedvectors = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
			fixedvectors[[0,1,2],np.nonzero(1-np.array(axes))[0]] = 1
			fixedvectors[3] = fixedvectors[0] + fixedvectors[1]
			fixedvectors[4] = fixedvectors[0] + fixedvectors[2]
			fixedvectors[5] = fixedvectors[1] + fixedvectors[2]
			fixedvectors[6] = 1-np.array(axes)
			for fv in fixedvectors:
				included = included + np.pad(corners[fv[0]:,fv[1]:,
							fv[2]:,fv[3]:,fv[4]:,fv[5]:],((0,fv[0]),(0,fv[1]),(0,fv[2]),(0,fv[3]),(0,fv[4]),(0,fv[5])),
							'constant',constant_values=(False))
			
			
			# Deflation version
#			deflation_corners = np.all(np.abs((embedding_space-deflation.dot(a.T)-np.ones(6)/2).dot(normallel.T)
#							.dot( np.linalg.inv(normallel.T[np.nonzero(axes)[0]]) )+np.ones(3)/2)<0.5,axis=6)
#			deflation_included = deflation_included + deflation_corners
#			fixedvectors = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
#			fixedvectors[[0,1,2],np.nonzero(1-np.array(axes))[0]] = 1
#			fixedvectors[3] = fixedvectors[0] + fixedvectors[1]
#			fixedvectors[4] = fixedvectors[0] + fixedvectors[2]
#			fixedvectors[5] = fixedvectors[1] + fixedvectors[2]
#			fixedvectors[6] = 1-np.array(axes)
#			for fv in fixedvectors:
#				included = included + np.pad(corners[fv[0]:,fv[1]:,
#							fv[2]:,fv[3]:,fv[4]:,fv[5]:],((0,fv[0]),(0,fv[1]),(0,fv[2]),(0,fv[3]),(0,fv[4]),(0,fv[5])),
#							'constant',constant_values=(False))
			
			
		# This variable holds the actual included points.
		# Commented out since we need lines instead.
		#latticepoints = embedding_space[included]
		#lattice3d = latticepoints.dot(worldplane.transpose())
		#smaller_lattice3d = lattice3d[np.all(np.abs(lattice3d[11222]) < 4,axis=1)]
		
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
		
		# Choose a chunk near origin
		#print(included[5,5,5,5,5,5])
		#print(included[6,5,5,5,5,5])
		#print(included[5,6,5,5,5,5])
		#print(included[5,5,6,5,5,5])
		#print(included[6,6,6,5,5,5])
		# Well, I choose the one from [5,5,5,5,5,5] to [6,6,6,5,5,5]
		chosen_center = np.array([5.5,5.5,5.5,5,5,5])
		chosen_origin = np.array([5,5,5,5,5,5])
		chosen_axes = np.array([1,1,1,0,0,0])
		chosen_axis1 = np.array([1,0,0,0,0,0])
		chosen_axis2 = np.array([0,1,0,0,0,0])
		chosen_axis3 = np.array([0,0,1,0,0,0])
		
		st = SurfaceTool()
		st.begin(Mesh.PRIMITIVE_LINES)
		
		multiplier = 4
		st.add_color(Color(1,0,1))
		for dim in range(6):
			basis_element = np.zeros(6)
			basis_element[dim] = 1
			offset = basis_element.dot(worldplane.transpose())#*(-phi*phi*phi)
			for line in all_lines[dim]:
				point1 = line.dot(worldplane.transpose())
				point2 = point1 + offset
				point1 *= multiplier*(-phi*phi*phi)
				point2 *= multiplier*(-phi*phi*phi)
				#if True:#np.linalg.norm(point1 - np.array([-27.41640786 -16.94427191  -0.])) < 20 or np.linalg.norm(point2 - np.array([-27.41640786 -16.94427191  -0.])) < 20:
				if np.all(np.abs((np.array([point1,point2])/(multiplier*np.linalg.norm(worldplane[1]))).dot(np.linalg.inv(normalworld.T[np.nonzero(chosen_axes)[0]]*(-phi*phi*phi))) - 0.5) < .801):
					st.add_vertex(Vector3(point1[0],point1[1],point1[2]))
					st.add_vertex(Vector3(point2[0],point2[1],point2[2]))
		
		
		array_mesh = ArrayMesh()
		st.commit(array_mesh)
		self.mesh = array_mesh
		
		
		st = SurfaceTool()
		st.begin(Mesh.PRIMITIVE_LINES)
		
		for dim in range(6):
			basis_element = np.zeros(6)
			basis_element[dim] = 1
			offset = basis_element.dot(worldplane.transpose())
			for line in all_lines[dim]:
				# Test if inside chosen chunk
				point1 = line.dot(worldplane.transpose())
				point2 = point1 + offset
				point1 *= multiplier
				point2 *= multiplier
				#np.linalg.norm(point1 - np.array([-27.41640786 -16.94427191  -0.])) < 20 or np.linalg.norm(point2 - np.array([-27.41640786 -16.94427191  -0.])) < 20:#True:#5*multiplier < point1[1] < 10*multiplier:
				if np.all(np.abs((np.array([point1,point2])/(multiplier*np.linalg.norm(worldplane[1]))).dot(np.linalg.inv(normalworld.T[np.nonzero(chosen_axes)[0]]*(-phi*phi*phi))) - 0.5) < .801):
					st.add_vertex(Vector3(point1[0],point1[1],point1[2]))
					st.add_vertex(Vector3(point2[0],point2[1],point2[2]))
		
		st.commit(self.mesh)
		
		self.mesh.surface_set_material(0,COLOR)
		
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
