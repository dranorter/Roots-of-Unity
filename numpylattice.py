from godot import exposed, export
from godot.bindings import _File as File
from godot import *
import numpy as np
from debugging import debugging
import traceback
import random as r
import time
import numbers

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
#		dists = np.sum(np.stack(np.repeat([embedding_space
#			.reshape((-1,6))-a],15,axis=0),axis=1).dot(normallel.T)
#			* np.concatenate(np.array([twoface_normals])
#			.reshape((1,15,3))),axis=-1)
#
#		self.es_dists = (np.stack(np.repeat([embedding_space
#			.reshape((-1,6)) ],15,axis=0),axis=1).dot(normallel.T)
#			* np.concatenate(np.array([twoface_normals])
#			.reshape((1,15,3))) )
		dists = np.sum( self.es_dists
			 - a.dot(normallel.T)
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
#		deflated_lines = [[],[],[],[],[],[]]
#		for i in embedding_space[deflation_included]:
#			for j in embedding_space[deflation_included]:
#				if np.linalg.norm(i - j) == 3:
#					if np.sum(np.abs(i - j)) == 7:
#						if np.where(i - j == 2)[0].shape == (1,):
#							deflated_lines[np.where(i - j == 2)[0][0]].append(j)
#
#		chunks = []
#		for point in embedding_space[deflation_included]:
#			# Find dimensions in which we have a neighbor in the positive direction
#			pos_lines = set()
#			for dim in range(6):
#				if np.any(np.all(np.array(deflated_lines[dim]) - point == 0,axis=1)):
#					pos_lines.add(dim)
#			# Consider all sets of 3 such dimensions
#			if len(pos_lines) >= 3:
#				for i in pos_lines:
#					for j in pos_lines - set([i]):
#						for k in pos_lines - set([i,j]):
#							# Check whether all eight cube corners exist with these axes
#							try:
#								if (deflation_included[tuple((point - offset)+deflation_face_axes[i]
#														+deflation_face_axes[j])] and
#									deflation_included[tuple((point - offset)+ deflation_face_axes[i]
#														+deflation_face_axes[k])] and
#									deflation_included[tuple((point - offset)+ deflation_face_axes[j]
#														+deflation_face_axes[k])] and
#									deflation_included[tuple((point - offset)+ deflation_face_axes[i]
#														+deflation_face_axes[j]
#														+deflation_face_axes[k])]):
#									chunks.append(point+(np.array(deflation_face_axes[i])
#										+np.array(deflation_face_axes[j])+np.array(deflation_face_axes[k]))/2)
#							except IndexError:
#								# If one of the indices was out of bound, it's not a chunk, so do nothing.
#								pass
#		print("Chunks computed. t="+str(time.perf_counter()-starttime)+" seconds")
		# TODO It seems that very occasionally, deflation_faces is empty at this
		# point; probably just means esize has to be greater than 8 to guarantee there are chunks.
		# Recording a-values where this happens: [ 0.03464434, -0.24607234,
		# -0.05386369,  0.2771699,   0.39596138,  0.45066235]
		
		if chosen_center is None:
			# Choose a chunk near origin
			deflation_faces = np.array(chunks)
			chosen_center = deflation_faces[
				np.where(np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1)
				== np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1).min())[0][0]]
		else:
			# Verify that chosen_center is within the existing chunks
			if False:#not np.any(np.all(np.array(chunks) - chosen_center == 0,axis=1)):
				print("Intended chunk not present!")
				possible_centers_live = np.array([[0.5, 0.5, 0.5,0.,0.,0.],[0.5,0.5,2., 1.,-1.5, 1.], [ 0.5,1.,1.5, 0., -0.5, 1.],
				[ 0.5,  1.5,  1.  ,-0.5,  0. ,  1. ], [ 0.5,  2. ,  0.5, -1.5,  1. ,  1. ], [ 0.5 , 2. ,  2. , -0.5, -0.5,  2. ], 
				[ 1. ,  0.5,  1.5 , 1. , -0.5,  0. ], [ 1. ,  1.5,  2. ,  0.5, -0.5,  1. ], [ 1.  , 1.5,  0.5, -0.5,  1.,   0. ], 
				[ 1. ,  2. ,  1.5 ,-0.5,  0.5,  1. ], [ 1.5,  0.5,  1. ,  1. ,  0. , -0.5], [ 1.5 , 1. ,  0.5,  0. ,  1.,  -0.5], 
				[ 1.5,  1. ,  2.  , 1. , -0.5,  0.5], [ 1.5,  2. ,  1. , -0.5,  1. ,  0.5], [ 2.  , 0.5,  0.5,  1. ,  1.,  -1.5], 
				[ 2. ,  0.5,  2.  , 2. , -0.5, -0.5], [ 2. ,  1. ,  1.5,  1. ,  0.5, -0.5], [ 2.  , 1.5,  1. ,  0.5,  1.,  -0.5], 
				[ 2. ,  2. ,  0.5 ,-0.5,  2. , -0.5], [2.,  2.,  2.,  0.5, 0.5, 0.5]])
				center_guarantee = dict()
				for center in possible_centers_live:
					center_axes = 1-np.array(center - np.floor(center))*2
					center_origin = center - np.array(deflation_face_axes).T.dot(center_axes)/2
					print("Origin (should be zeros): "+str(center_origin))
					center_axis1 = np.array(deflation_face_axes[np.nonzero(center_axes)[0][0]])
					center_axis2 = np.array(deflation_face_axes[np.nonzero(center_axes)[0][1]])
					center_axis3 = np.array(deflation_face_axes[np.nonzero(center_axes)[0][2]])
					chunk_corners = np.array([center_origin,
						center_origin+center_axis1,center_origin+center_axis2,center_origin+center_axis3,
						center_origin+center_axis1+center_axis2,center_origin+center_axis1+center_axis3,center_origin+center_axis2+center_axis3,
						center_origin+center_axis1+center_axis2+center_axis3])
					center_constraints = np.sum(np.stack(np.repeat([chunk_corners - a],30,axis=0),axis=1).dot(normallel.T)
							* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
					overall_center_constraints = 0.9732489894677302/(phi*phi*phi) - np.max(center_constraints,axis=0)
					translated_constraints = (overall_center_constraints*np.concatenate([-np.ones(15),np.ones(15)]) 
							+ np.concatenate([twoface_normals,twoface_normals]).dot(a.dot(normallel.T)))
					translated_constraints = (translated_constraints).reshape((2,15)).T
					center_guarantee[str(center)] = translated_constraints
				generates_correct_chunk = (np.all(twoface_normals.dot(a.dot(normallel.T)) 
							> center_guarantee[str(chosen_center)][:,0] )
						and np.all(twoface_normals.dot(a.dot(normallel.T)) < center_guarantee[str(chosen_center)][:,1]))
				print("The test that got us here says: "+generates_correct_chunk)
				raise Exception("Intended chunk not present!")
		print("Chose chunk "+str(chosen_center)+" second="+str(time.perf_counter()-starttime))
		chosen_axes = 1-np.array(chosen_center - np.floor(chosen_center))*2
		chosen_origin = chosen_center - np.array(deflation_face_axes).T.dot(chosen_axes)/2
		chosen_axis1 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][0]])
		chosen_axis2 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][1]])
		chosen_axis3 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][2]])
		
		# Now move the chosen chunk to center stage
		embedding_space = embedding_space - chosen_origin
		chosen_center = chosen_center - chosen_origin
		#deflated_lines = [[l - chosen_origin for l in ll] for ll in deflated_lines]
		all_lines = [[l - chosen_origin for l in ll] for ll in all_lines]
		a = (a - chosen_origin).dot(squarallel)
		chosen_origin = np.zeros(6)
		print("Corrected offset:")
		print(a)

		# TODO: This part is a little slow
		blocks = np.zeros((0,6))
		for axes in ch3:
			ax1, ax2, ax3 = np.eye(6,dtype=np.int64)[np.nonzero(axes)[0]]
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
		print("Found "+str(len(blocks))+" blocks in the lattice")
		multiplier = 4
		
		neighbor_blocks = []
		inside_blocks = []
		axes_matrix = np.linalg.inv(worldplane.T[np.nonzero(chosen_axes)[0]]
									*(phi*phi*phi*multiplier) )
		worldplane_chunk_center = (chosen_center).dot(worldplane.T)*multiplier
		block_center_in_chunk = np.all(np.abs((np.array(blocks).dot(worldplane.T)*multiplier 
							- worldplane_chunk_center).dot( axes_matrix )) <= 0.5,axis=1)
		for i in np.nonzero(block_center_in_chunk)[0]:
			inside_blocks.append(blocks[i])
		
		# TODO This is one of the slowest parts.
		# Each corner is being checked many times, once for 
		# each neighboring block. Also maybe I could remove
		# " * multiplier" everywhere.
		face_origins = np.floor(blocks).dot(worldplane.T)*multiplier
		face_tips = np.ceil(blocks).dot(worldplane.T)*multiplier
		dirNs = np.eye(6)[np.nonzero(np.ceil(blocks)-np.floor(blocks))[1].reshape((-1,3))].dot(worldplane.T)*multiplier
		dir1s, dir2s, dir3s = (dirNs[:,0], dirNs[:,1], dirNs[:,2])#.dot(np.array([[1.6,0,1,1.6,0,-1],[1,1.6,0,-1,1.6,0],[0,1,1.6,0,-1,1.6]]).T )
		corner1s,corner2s,corner3s,corner4s,corner5s,corner6s,corner7s,corner8s = (
				face_origins, face_tips, face_origins + dir1s, face_origins + dir2s, face_origins + dir3s,
				face_tips - dir1s, face_tips - dir2s, face_tips - dir3s)
		some_block_corner_in_chunk = np.any(np.all(np.abs((np.array([
								corner1s, corner2s, corner3s, corner4s, corner5s, corner6s, corner7s, corner8s]) 
								- worldplane_chunk_center).dot(axes_matrix)) < 0.5000001,axis=-1),axis=0)
		for i in np.nonzero(1 - block_center_in_chunk)[0]:
#			block = blocks[i]
#			face_origin = np.floor(block).dot(worldplane.T)*multiplier
#			face_tip = np.ceil(block).dot(worldplane.T)*multiplier
#			dir1,dir2,dir3 = np.eye(6)[np.nonzero(np.ceil(block)-np.floor(block))[0]].dot(worldplane.T)*multiplier
#			corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = (
#				face_origin, face_tip, face_origin + dir1, face_origin + dir2, face_origin + dir3,
#				face_tip - dir1, face_tip - dir2, face_tip - dir3)
			if some_block_corner_in_chunk[i]:#np.any(np.all(np.abs((np.array([
				#				corner1s[i], corner2s[i], corner3s[i], corner4s[i], corner5s[i], corner6s[i], corner7s[i], corner8s[i]]) 
				#				- worldplane_chunk_center).dot(axes_matrix)) < 0.5000001,axis=-1)):
				neighbor_blocks.append(blocks[i])
				# Represents a voxel on the boundary of our chosen chunk
				#dir1 = Vector3(dir1[0],dir1[1],dir1[2])
				#dir2 = Vector3(dir2[0],dir2[1],dir2[2])
				#dir3 = Vector3(dir3[0],dir3[1],dir3[2])
		print("Found "+str(len(neighbor_blocks))+" blocks neighboring chunk")
		print("Found "+str(len(inside_blocks))+" blocks inside the chunk")
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
								*(phi*phi*phi*multiplier) ))) < 1.0,axis=1)# Was "< 0.501"
		relevant_points = embedding_space[included][relevance]
		# Recalculating just because it's fast and easier than trying to get the
		# right numbers, arranged properly, out of the old "constraints" variable.
		dists = np.sum(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		# The intersection of all the constraints takes the minima along each of the 30 vectors.
		#if constraints is None:
		constraints = 0.9732489894677302 - np.max(dists,axis=0)
		
		# Need to do a similar calculation for chunk corners, but with tighter constraints.
		chunk_corners = np.array([chosen_origin,
			chosen_origin+chosen_axis1,chosen_origin+chosen_axis2,chosen_origin+chosen_axis3,
			chosen_origin+chosen_axis1+chosen_axis2,chosen_origin+chosen_axis1+chosen_axis3,chosen_origin+chosen_axis2+chosen_axis3,
			chosen_origin+chosen_axis1+chosen_axis2+chosen_axis3])
		chunk_dists = np.sum(np.stack(np.repeat([chunk_corners - a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		print("Max chunk constraint distance: "+str(np.max(np.abs(chunk_dists))))
		overall_chunk_constraints = 0.9732489894677302/(phi*phi*phi) - np.max(chunk_dists,axis=0)
		
		if np.any(overall_chunk_constraints <= 0):
			print("Bad chunk constraints. Hopefully just means the chunk isn't really present.")
			raise Exception("Bad chunk constraints.")
		else:
			constraints = np.min([constraints,overall_chunk_constraints],axis=0)
		
		#print("Wiggle room in all 30 directions:")
		#print(constraints)
		print("Wiggle room along the 15 axes:")
		print(constraints[:15]+constraints[15:])
		
		print("Proposed new point inside the constraints:")
		b = np.array([a[0]+r.random()/2-0.25,a[1]+r.random()/2-0.25,a[2]+r.random()/2-0.25,
					a[3]+r.random()/2-0.25,a[4]+r.random()/2-0.25,a[5]+r.random()/2-0.25])
#		loop_counter = 0
#		prop_search_limit = 1000
#
#		while np.any(np.concatenate([twoface_normals,-twoface_normals]).dot((a-b).dot(normallel.T)) > 
#							constraints)and(loop_counter<prop_search_limit):
#			# Move the generated point toward the constraints by a random amount
#			# Get the min and max distances we need to move to get in this axis' constraints
#			divergence = b - a
#			rand_pos = r.random()
#			move = divergence * rand_pos
#			b = b - move
#
#			if not np.any(np.concatenate([twoface_normals,-twoface_normals]).dot((a-b).dot(normallel.T)) > constraints):
#				# Break early before we mess it up
#				break
#			loop_counter = loop_counter + 1
#		if loop_counter >= prop_search_limit:
#			# We failed to find one.
#			raise Exception("Overran loop_counter searching for an offset satisfying the constraints.")
#		print(b)
#		new_constraints = np.sum(np.stack(np.repeat([relevant_points - b],30,axis=0),axis=1).dot(normallel.T)
#			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
#		print("New max constraint distance: "+str(np.max(np.abs(new_constraints))))
#		print("Predicted new max: "+str(np.max(
#			-0.9732489894677302 + np.max(np.sum(
#				(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T) )
#				* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3)))
#			,axis=2),axis=0)
#			+ 0.9732489894677302 + np.dot(
#				np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),
#				(a-b).dot(normallel.T)
#			)
#			)))
#		print("Predicted to pass: "+str(np.all(
#			+ np.dot(
#				np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),
#				(a-b).dot(normallel.T))
#			< 0.9732489894677302 - np.max(np.sum(
#				(np.stack(np.repeat([relevant_points - a],30,axis=0),axis=1).dot(normallel.T) )
#				* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3)))
#			,axis=2),axis=0)
#			)))
		return (b, constraints, chosen_center, neighbor_blocks, inside_blocks)
	
	def _ready(self):
		debugging.breakpoint()
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
		a = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])*2 -1
		#a = np.array([-0.16913145,  0.04060133, -0.33081354,  0.76832666,  0.53877964,  0.63870467])
		#a = np.array([-0.0441522,  -0.09743448, -0.38699097,  0.79503878,  0.61608302,  0.82796904])
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
								[ 1, 1,-1,-1, 2, -1],
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
		
		# TODO This could be made faster if I would take an initial 
		# product of twoface_normals with the parallel space basis vectors - 
		# then I could just have a matrix transformation from a parallel space
		# point to its distance along all 15 constraint directions, rather than 
		# doing 15 separate dot products.
		print("Ready to compute lattice at t="+str(time.perf_counter()-starttime))
		self.es_dists = (np.stack(np.repeat([embedding_space
			.reshape((-1,6)) ],15,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals])
			.reshape((1,15,3))) )
		constraints = np.sum( self.es_dists
			 - a.dot(normallel.T)
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
		# TODO Finding deflated lines takes perhaps 100 times as long
		# as finding the lines.
		deflated_lines = [[],[],[],[],[],[]]
		nonstandard_deflated_lines = set()
		standard_deflated_lines = set()
		for i in embedding_space[deflation_included]:
			for j in embedding_space[deflation_included]:
				#print(np.linalg.norm(i - j))
				#print(np.sum(np.abs(i - j)))
				#print(np.where(i - j == 2)[0].shape == (1,))
				if np.linalg.norm(i - j) == 3:
					if np.sum(np.abs(i - j)) == 7:
						if np.where(i - j == 2)[0].shape == (1,):
							deflated_lines[np.where(i - j == 2)[0][0]].append(j)
							standard_deflated_lines.add(str(i-j))
						else:
							nonstandard_deflated_lines.add(str(i-j))
		print("Might-be lines: "+str(nonstandard_deflated_lines)+"\n"+str(standard_deflated_lines))
		
		#Want to identify the blocks and chunks
		print("Deflated lines found in "+str(time.perf_counter()-starttime))
		blocks = None#np.zeros((0,6))
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
			if blocks is None:
				blocks = embedding_space[nonzero]+np.array(ax123,dtype=np.float)/2
			else:
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
		near_centers = deflation_faces[
			np.where(np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1)
			< 8)[0]]
		chosen_center = near_centers[r.randint(0,near_centers.shape[0]-1)]
#		chosen_center = deflation_faces[
#			np.where(np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1)
#			== np.linalg.norm(deflation_faces.dot(worldplane.T),axis=1).min())[0][0]]
		#Rigging the process
		#chosen_center = np.array([0.5, 1.5, 0.,  0.,  1.5, 1. ])
		print("Chose chunk "+str(chosen_center)+" second="+str(time.perf_counter()-starttime))
		chosen_axes = 1-np.array(chosen_center - np.floor(chosen_center))*2
		chosen_origin = chosen_center - np.array(deflation_face_axes).T.dot(chosen_axes)/2
		chosen_axis1 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][0]])
		chosen_axis2 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][1]])
		chosen_axis3 = np.array(deflation_face_axes[np.nonzero(chosen_axes)[0][2]])
		
		# Now move the chosen chunk to center stage
		print("Chosen chunk corner (will be sent to origin)"+str(chosen_origin))
		print("First element of lines list:"+str(all_lines[0][0]))
		print("First element of chunk lines list:"+str(deflated_lines[0][0]))
		embedding_space = embedding_space - chosen_origin
		chosen_center = chosen_center - chosen_origin
		chunks = np.array(chunks) - chosen_origin
		blocks = blocks - chosen_origin
		deflated_lines = [[l - chosen_origin for l in ll] for ll in deflated_lines]
		all_lines = [[l - chosen_origin for l in ll] for ll in all_lines]
		a = (a - chosen_origin).dot(squarallel)
		chosen_origin = np.zeros(6)
		print("Corrected offset:")
		print(a)
		print("New first element of lines list:"+str(all_lines[0][0]))
		print("New first element of chunk lines list:"+str(deflated_lines[0][0]))
		
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
		
		# Draw the boundaries of the chunk itself.
		
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
								*(phi*phi*phi*multiplier) ))) < 1.0,axis=1)# Was "< 0.501"
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
		
		# Need to do a similar calculation for chunk corners, but with tighter constraints.
		chunk_corners = np.array([chosen_origin,
			chosen_origin+chosen_axis1,chosen_origin+chosen_axis2,chosen_origin+chosen_axis3,
			chosen_origin+chosen_axis1+chosen_axis2,chosen_origin+chosen_axis1+chosen_axis3,chosen_origin+chosen_axis2+chosen_axis3,
			chosen_origin+chosen_axis1+chosen_axis2+chosen_axis3])
		chunk_constraints = np.sum(np.stack(np.repeat([chunk_corners - a],30,axis=0),axis=1).dot(normallel.T)
			* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
		print("Max chunk constraint distance: "+str(np.max(np.abs(chunk_constraints))))
		overall_chunk_constraints = 0.9732489894677302/(phi*phi*phi) - np.max(chunk_constraints,axis=0)
		
		overall_constraints = np.min([overall_constraints,overall_chunk_constraints],axis=0)
		
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
		repetitions = 0
		if False:#not Engine.editor_hint:
			for i in range(seeds):
				print()
				print("New seed:")
				b, overall_constraints, chosen_center, resulting_neighbor_blocks, resulting_interior_blocks = self.chunk_test()
				resulting_blocks = resulting_neighbor_blocks + resulting_interior_blocks
				for j in range(repetitions):
					wiggle_room = overall_constraints[:15]+overall_constraints[15:]
					b, new_constraints, new_center, new_neighbor_blocks, new_interior_blocks = self.chunk_test(
						b,chosen_center,constraints=overall_constraints)
					new_blocks = new_neighbor_blocks + new_interior_blocks
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
						pass
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
		possible_centers = {'[0.5 0.5 0.5 0.  0.  0. ]', '[ 0.5  0.5  2.   1.  -1.5  1. ]', '[ 0.5  1.   1.5  0.  -0.5  1. ]', 
			'[ 0.5  1.5  1.  -0.5  0.   1. ]', '[ 0.5  2.   0.5 -1.5  1.   1. ]', '[ 0.5  2.   2.  -0.5 -0.5  2. ]', 
			'[ 1.   0.5  1.5  1.  -0.5  0. ]', '[ 1.   1.5  2.   0.5 -0.5  1. ]', '[ 1.   1.5  0.5 -0.5  1.   0. ]', 
			'[ 1.   2.   1.5 -0.5  0.5  1. ]', '[ 1.5  0.5  1.   1.   0.  -0.5]', '[ 1.5  1.   0.5  0.   1.  -0.5]', 
			'[ 1.5  1.   2.   1.  -0.5  0.5]', '[ 1.5  2.   1.  -0.5  1.   0.5]', '[ 2.   0.5  0.5  1.   1.  -1.5]', 
			'[ 2.   0.5  2.   2.  -0.5 -0.5]', '[ 2.   1.   1.5  1.   0.5 -0.5]', '[ 2.   1.5  1.   0.5  1.  -0.5]', 
			'[ 2.   2.   0.5 -0.5  2.  -0.5]', '[2.  2.  2.  0.5 0.5 0.5]'}
		possible_centers_live = np.array([[0.5, 0.5, 0.5,0.,0.,0.],[0.5,0.5,2., 1.,-1.5, 1.], [ 0.5,1.,1.5, 0., -0.5, 1.],
			[ 0.5,  1.5,  1.  ,-0.5,  0. ,  1. ], [ 0.5,  2. ,  0.5, -1.5,  1. ,  1. ], [ 0.5 , 2. ,  2. , -0.5, -0.5,  2. ], 
			[ 1. ,  0.5,  1.5 , 1. , -0.5,  0. ], [ 1. ,  1.5,  2. ,  0.5, -0.5,  1. ], [ 1.  , 1.5,  0.5, -0.5,  1.,   0. ], 
			[ 1. ,  2. ,  1.5 ,-0.5,  0.5,  1. ], [ 1.5,  0.5,  1. ,  1. ,  0. , -0.5], [ 1.5 , 1. ,  0.5,  0. ,  1.,  -0.5], 
			[ 1.5,  1. ,  2.  , 1. , -0.5,  0.5], [ 1.5,  2. ,  1. , -0.5,  1. ,  0.5], [ 2.  , 0.5,  0.5,  1. ,  1.,  -1.5], 
			[ 2. ,  0.5,  2.  , 2. , -0.5, -0.5], [ 2. ,  1. ,  1.5,  1. ,  0.5, -0.5], [ 2.  , 1.5,  1. ,  0.5,  1.,  -0.5], 
			[ 2. ,  2. ,  0.5 ,-0.5,  2. , -0.5], [2.,  2.,  2.,  0.5, 0.5, 0.5]])
		
		center_guarantee = dict()
		for center in possible_centers_live:
			center_axes = 1-np.array(center - np.floor(center))*2
			center_origin = center - np.array(deflation_face_axes).T.dot(center_axes)/2
			print("Origin (should be zeros): "+str(center_origin))
			center_axis1 = np.array(deflation_face_axes[np.nonzero(center_axes)[0][0]])
			center_axis2 = np.array(deflation_face_axes[np.nonzero(center_axes)[0][1]])
			center_axis3 = np.array(deflation_face_axes[np.nonzero(center_axes)[0][2]])
			chunk_corners = np.array([center_origin,
				center_origin+center_axis1,center_origin+center_axis2,center_origin+center_axis3,
				center_origin+center_axis1+center_axis2,center_origin+center_axis1+center_axis3,center_origin+center_axis2+center_axis3,
				center_origin+center_axis1+center_axis2+center_axis3])
			# TODO I'm subtracting "a" and then adding it in again; should remove it from the calculation.
			center_constraints = np.sum(np.stack(np.repeat([chunk_corners - a],30,axis=0),axis=1).dot(normallel.T)
					* np.concatenate(np.array([twoface_normals,-twoface_normals]).reshape((1,30,3))),axis=2)
			overall_center_constraints = 0.9732489894677302/(phi*phi*phi) - np.max(center_constraints,axis=0)
			translated_constraints = (overall_center_constraints*np.concatenate([-np.ones(15),np.ones(15)]) 
					+ np.concatenate([twoface_normals,twoface_normals]).dot(a.dot(normallel.T)))
			translated_constraints = (translated_constraints).reshape((2,15)).T
			center_guarantee[str(center)] = translated_constraints
		print(center_guarantee.keys())
		#'[ 1.   0.5  1.5  1.  -0.5  0. ]'
		constraints_sorted = dict()
		for center in possible_centers:
			constraints_sorted[center] = []
		guarantee_scale = np.max(np.abs(center_guarantee[str(chosen_center)][0] - center_guarantee[str(chosen_center)][1]))
		#debugging.breakpoint()
		# Repetition counts above 10 mysteriously take forever.
		# At 10, takes like 51 seconds. At 11, I haven't waited long enough.
		# Yet it doesn't cover all the possible centers even...
		repetitions = 20000
		time_limit_seconds = 12*60*60
		inner_loop_upper_limit = 20000
		next_save_time = 1*60*60
		all_constraints = []
		all_blocks = []
		all_chunks = []
		all_counters = []
		all_chosen_centers = []
		all_block_axes = []
		all_sorted_constraints = []
		if not Engine.editor_hint:
			bb = a
			fs = File()
			
#			try:
#				fs.open("res://chunklayouts",fs.READ)
#				while not fs.eof_reached():
#					# relevant chunk as chosen_center string
#					ch_c = fs.get_line()
#					# Constraint is 30 floats
#					cstts = np.zeros((30))
#					for i in range(30):
#						cstts[i] = fs.get_real()
#					# Numbers of inside blocks and outside blocks
#					inside_ct = fs.get_32()
#					outside_ct = fs.get_32()
#					# Then retrieve the strings representing the blocks
#					is_blocks = []
#					os_blocks = []
#					for i in range(inside_ct):
#						is_blocks.append(fs.get_line())
#					for i in range(outside_ct):
#						fs.get_line()
#			except Exception as e:
#				print("Encountered some sort of problem saving.")
#				print(e)
#			fs.close()
			
			for i in range(repetitions):
				b = bb.copy()
				try:
					(bb, relative_constraints, chosen_center, neighbor_blocks, interior_blocks) = self.chunk_test(bb,chosen_center)

					# We can't use the raw constraints since they're centered on b; we need a shape
					# relative to the origin.
					# Simply translating them by the proper amount is a bit confusing;
					# in a given direction, starting from a pair like (0.1, 0.1) 
					# which means we've got a margin of 0.1 on both sides, we might 
					# translate it to something like (5.1,-4.9), which would mean
					# we need to be between 5.1 in the positive direction and -4.9 in
					# the negative direction -- IE, between 5.1 and 5.9 overall. So
					# we negate the "negative direction" values and then arrange
					# the whole thing in 15 pairs.
					constraints = (relative_constraints*np.concatenate([-np.ones(15),np.ones(15)]) 
						+ np.concatenate([twoface_normals,twoface_normals]).dot(b.dot(normallel.T)))
					constraints = (constraints).reshape((2,15)).T
					
					print(str(time.perf_counter()-starttime))
					weirdness = False
					if np.any(relative_constraints < 0):
						print("Some of the relative constraints were negative!")
						weirdness = True
					if np.any(np.concatenate([twoface_normals,-twoface_normals]).dot(np.zeros(6).dot(normallel.T)) > relative_constraints):
						print("Relative constraints didn't contain the point that generated them; some must've been negative.")
						weirdness = True
					if not np.all(constraints[:,0] < constraints[:,1]):
						print("Seeing some translated constraints with zero or negative width!")
						print(constraints[:,1] - constraints[:,0])
						weirdness = True
					if (not np.all(twoface_normals.dot(b.dot(normallel.T)) > constraints[:,0] )
									and np.all(twoface_normals.dot(b.dot(normallel.T)) < constraints[:,1])):
						print("Translated constraints didn't contain the point that generated them!!")
						weirdness = True
					if not (np.all(twoface_normals.dot(b.dot(normallel.T)) > center_guarantee[str(chosen_center)][:,0] )
									and np.all(twoface_normals.dot(b.dot(normallel.T)) < center_guarantee[str(chosen_center)][:,1])):
						print("Chunk constraint doesn't contain the point that generated it!")
						weirdness = True
						# Values which have caused this:
						# [ 0.01094418  0.04052633 -0.09285401  0.09200656  0.00957298  0.07372379]
					if time.perf_counter()-starttime > time_limit_seconds:
						print("Taking more than overall time limit...")
						weirdness = True
					if weirdness:
						raise Exception("Going to skip this loop then")
					
					all_constraints.append(constraints)
					constraints_sorted[str(chosen_center)].append(constraints)
					all_sorted_constraints.append(str((chosen_center,constraints)))
					
					# Save the chunk info
					all_chunks.append(str((chosen_center, interior_blocks, neighbor_blocks)))
					all_blocks.append((interior_blocks,neighbor_blocks))
					all_chosen_centers.append(str(chosen_center))
					for block in interior_blocks:
						all_block_axes.append(str(block - np.floor(block)))
				except Exception as e:
					print("Weirdness, skipping this loop")
					print(e)
					traceback.print_exc()
				# Testing that we know how to match constraints properly
				# The suggested value, bb, should fall within the constraints.
#				print("Did the suggested value fall in its constraints? "+str(not
#					np.any(np.concatenate([twoface_normals,-twoface_normals]).dot((b-bb).dot(normallel.T)) > relative_constraints)))
#				print("Was my algebra correct? "+str(
#					np.all(twoface_normals.dot(bb.dot(normallel.T)) > -relative_constraints[:15] + twoface_normals.dot(b.dot(normallel.T))) and
#					np.all(twoface_normals.dot(bb.dot(normallel.T)) < relative_constraints[15:] + twoface_normals.dot(b.dot(normallel.T)))))
#				print(str(constraints[:,0] + relative_constraints[:15] - twoface_normals.dot(b.dot(normallel.T))))
#				print("Does our while loop use a valid test? "+
#					str(np.all(twoface_normals.dot(bb.dot(normallel.T)) > constraints[:,0] )
#								and np.all(twoface_normals.dot(bb.dot(normallel.T)) < constraints[:,1])))
				bb = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
				bb = bb*2 - 1
				bb = bb.dot(squarallel)
				chosen_center = r.choice(possible_centers_live)
				print("...")
				print("New target chunk: "+str(chosen_center))
				in_existing_constraint = np.any([(np.all(twoface_normals.dot(bb.dot(normallel.T)) > constraint[:,0] )
								and np.all(twoface_normals.dot(bb.dot(normallel.T)) < constraint[:,1])) 
								for constraint in constraints_sorted[str(chosen_center)]])
				generates_correct_chunk = (np.all(twoface_normals.dot(bb.dot(normallel.T)) > center_guarantee[str(chosen_center)][:,0] )
								and np.all(twoface_normals.dot(bb.dot(normallel.T)) < center_guarantee[str(chosen_center)][:,1]))
				
				counter = 0
				upper_limit = inner_loop_upper_limit
				
				while ((not generates_correct_chunk) or in_existing_constraint) and counter < upper_limit:
					print(str(time.perf_counter()-starttime))
					bb = np.array([r.random(),r.random(),r.random(),r.random(),r.random(),r.random()])
					bb = bb*2 - 1
					bb = bb.dot(squarallel)
					chosen_center = r.choice(possible_centers_live)
					generates_correct_chunk = (np.all(twoface_normals.dot(bb.dot(normallel.T)) > center_guarantee[str(chosen_center)][:,0] )
								and np.all(twoface_normals.dot(bb.dot(normallel.T)) < center_guarantee[str(chosen_center)][:,1]))
					# It appears this inner loop is the slow part. Would be well worth it to 
					# come up with these in a more thorough manner.
					while (not generates_correct_chunk) and (counter < upper_limit):
						_ = list(range(15))
						r.shuffle(_)
						for axis in _:
							# Move the generated point toward the constraints by a random amount
							#axis = r.randint(0,5)
							# Get the min and max distances we need to move to get in this axis' constraints
							divergence = center_guarantee[str(chosen_center)][axis] - twoface_normals[axis].dot(bb.dot(normallel.T))
							# Is it outside the constraints in this direction?
							if divergence[0]*divergence[1] >= 0:
								rand_pos = r.random()
								move = (divergence[0]*rand_pos + divergence[1]*(1-rand_pos))*twoface_normals[axis]
								bb = bb + move.dot(normallel)
								
								generates_correct_chunk = (np.all(twoface_normals.dot(bb.dot(normallel.T)) 
											> center_guarantee[str(chosen_center)][:,0] )
										and np.all(twoface_normals.dot(bb.dot(normallel.T)) < center_guarantee[str(chosen_center)][:,1]))
								if generates_correct_chunk:
									# Break early before we mess it up
									break
						counter = counter + 1
						if time.perf_counter()-starttime > time_limit_seconds:
							print("Exceeded overall time limit while in inner while loop.")
							break
					if generates_correct_chunk:
						print("Found offset with same chunk; counter="+str(counter)+". Distance from old value: "+str((bb-b)/guarantee_scale))
					else:
						print("Procedure failed to get the intended chunk.")
					counter = counter + 1
					in_existing_constraint = np.any([(np.all(twoface_normals.dot(bb.dot(normallel.T)) > constraint[:,0] )
								and np.all(twoface_normals.dot(bb.dot(normallel.T)) < constraint[:,1])) 
								for constraint in constraints_sorted[str(chosen_center)]])
					if time.perf_counter()-starttime > time_limit_seconds:
						print("Exceeded time limit in outer while loop.")
						break
				all_counters.append(counter)
				if counter >= upper_limit:
					print("Loop overran its limit at attempt #"+str(i+1))
					break
				if time.perf_counter()-starttime > time_limit_seconds:
					print("Exceeded time limit in outer loop.")
					break
				if time.perf_counter()-starttime > next_save_time:
					fs = File()
					fs.open("res://chunklayouts_hourly"+str(round((time.perf_counter()-starttime)/(60*60),2)),fs.WRITE)
					# For each successful loop:
					for i in range(len(all_chunks)):
						try:
							# Store the relevant chunk as chosen_center string
							fs.store_line(all_chosen_centers[i])
							# Store the 30 floats
							for f in all_constraints[i].flatten():
								fs.store_real(f)
							# First store the number of inside blocks and outside blocks
							fs.store_line(str(len(all_blocks[i][0])))
							fs.store_line(str(len(all_blocks[i][1])))
							# Then store the blocks as strings
							for block in all_blocks[i][0]:
								fs.store_line(repr(list(block)))
							for block in all_blocks[i][1]:
								fs.store_line(repr(list(block)))
						except Exception as e:
							print("Encountered some sort of problem saving.")
							print(e)
							break
					fs.close()
					next_save_time = next_save_time + 60*60
			#TODO Would be nice to combine constraint regions which produce the same
			# chunk layout (some of which will be overlapping).
			print("t="+str(time.perf_counter()-starttime))
			print("After "+str(len(all_counters))+" tries,")
			print(str(len(all_chunks))+" chunk layouts generated. Repeats:")
			print(len(all_chunks)-len(set(all_chunks)))
			print(str(len(set([str(tup) for tup in all_blocks])))+" unique layouts.")#2298 unique layouts.
			print(str(len(all_constraints))+" constraints generated. Repeats:")
			print(len(all_constraints)-len(set([str(x) for x in all_constraints])))
			print(str(len(set([str(x) for x in all_constraints])))+" unique constraints. (No check for overlap or subset/superset.)")
			
			print("Max loops required: "+str(max(all_counters)))
			all_counters = np.array(all_counters)
			print("Unique sorted constraints: "+str(len(all_sorted_constraints)))
			print("Average number of loops: "+str(all_counters.mean()))
			print("Constraint counts for each possible chunk: "+str([len(x) for x in constraints_sorted.values()]))
			print("Constraint counts minus repeats: "+str([len(set([str(member) for member in x])) for x in constraints_sorted.values()]))
			print("All counters:")
			print(all_counters)
			
			fs = File()
			fs.open("res://chunklayouts",fs.WRITE)
			# For each successful loop:
			for i in range(len(all_chunks)):
				try:
					# Store the relevant chunk as chosen_center string
					fs.store_line(all_chosen_centers[i])
					# Store the 30 floats
					for f in all_constraints[i].flatten():
						fs.store_real(f)
					# First store the number of inside blocks and outside blocks
					fs.store_line(str(len(all_blocks[i][0])))
					fs.store_line(str(len(all_blocks[i][1])))
					# Then store the blocks as strings
					for block in all_blocks[i][0]:
						fs.store_line(repr(list(block)))
					for block in all_blocks[i][1]:
						fs.store_line(repr(list(block)))
				except Exception as e:
					print("Encountered some sort of problem saving.")
					print(e)
					break
			fs.close()
			
			self.constraints_sorted = constraints_sorted
			self.all_chunks = all_chunks
			self.all_constraints = all_constraints
			self.all_blocks = all_blocks
			self.all_chunks = all_chunks
			self.all_sorted_constraints = all_sorted_constraints
			self.all_chosen_centers = all_chosen_centers
			#print(set(all_chosen_centers))
			#print(set(all_block_axes))
		# There are a couple ways I could search more systematically.
		# If I can list all the blocks which occur in at least one chunk layout,
		# I could proceed in something like a tree: the root node would have
		# no commitments, and each child would represent the presence of a certain
		# block in the chunk. This would be redundant, so I'd put the blocks in
		# some order and require children to be greater than their parent within
		# that order. The tree would be traversed depth-first, and reaching a 
		# given node would mean calculating the constraints for the corresponding
		# block and finding that they're compatible with the parent constraints.
		# (I suppose there could be child nodes representing dead ends, IE 
		# labeled with a given block and the fact that it's incompatible.)
		# When no more children can be added, the chunk is either complete, or
		# it's incomplete (due to the ordering requirement excluding blocks 
		# which are compatible with all constraints so far). We can avoid
		# generating incomplete leaf nodes as follows. We traverse using the
		# ordering of course, checking possible children from low to high. A
		# new child can either make the constraints narrower than the parent,
		# or keep them the same. If they're kept the same, this means the 
		# presence of all blocks from ancestor nodes guaranteed the presence of
		# this block. This needs to be the last block (according to the ordering,
		# and the traversal order) added as a child for the parent node. Any
		# additional siblings would be missing that logically implied block,
		# and therefore could only end up incomplete.
		# So, avoiding invalid children and incomplete children, whenever we
		# run out of possible children it just means we've checked all blocks
		# and no more can be added to the chunk. The process as described, 
		# then, would just produce maximal chunks, and wouldn't cover regions
		# of the constraint space where moving over slightly would add blocks
		# without removing any. If we're really talking about blocks which
		# overlap the chunk, this shouldn't be an issue, but for completeness
		# nodes could be added which state the exclusion of a block rather
		# than its inclusion. This implies we'll end up with every block, either
		# excluded or included, as a parent of every leaf node; so each node
		# has a maximum of two children, corresponding to the inclusion or
		# exclusion of the very next block in the chosen ordering.
	
	def _process(self,delta):
		if not Engine.editor_hint:
			debugging.breakpoint()
			pass

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
