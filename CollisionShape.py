from godot import exposed, export
from godot import *
import debugging

@exposed(tool=False)
class CollisionShape(CollisionShape):
	
	_render = False
	phi = 1.61803398874989484820458683
	
	@export(bool)
	@property
	def render(self):
		return self._render
	
	@render.setter
	def render(self,do_render):
		self._render = do_render
		if not Engine.is_editor_hint():
			return
		self.generate()
	
	def generate(self):
		phi = self.phi
		vertices = PoolVector3Array()
		vertices.push_back(Vector3(0,0,0))
		vertices.push_back(Vector3(0,-2-3*phi,1+2*phi))
		vertices.push_back(Vector3(2+3*phi,-3-5*phi,1+2*phi))
		vertices.push_back(Vector3(2+3*phi,-1-2*phi,0))
		vertices.push_back(Vector3(3+5*phi,-1-2*phi,2+3*phi))
		vertices.push_back(Vector3(1+2*phi,0,2+3*phi))
		vertices.push_back(Vector3(1+2*phi,-2-3*phi,3+5*phi))
		vertices.push_back(Vector3(0,-2-3*phi,1+2*phi))
		
		#array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)
		
		vertices.push_back(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		#vertices.push_back(Vector3(-1-2*phi,0,2+3*phi))
		#vertices.push_back(Vector3(2+3*phi,-1-2*phi,0))
		#vertices.push_back(Vector3(0,-2+-3*phi,-1-2*phi))
		#vertices.push_back(Vector3(0,2+3*phi,1+2*phi))
		#vertices.push_back(Vector3(2+3*phi,1+2*phi,0))
		#vertices.push_back(Vector3(0,2+3*phi,-1-2*phi))
		#vertices.push_back(Vector3(-2-3*phi,1+2*phi,0))
		#vertices.push_back(Vector3(-1-2*phi,0,-2-3*phi))
		#vertices.push_back(Vector3(1+2*phi,0,-2-3*phi))
		array_mesh = ArrayMesh()
		arrays = builtins.Array()
		arrays.resize(ArrayMesh.ARRAY_MAX)
		arrays[ArrayMesh.ARRAY_VERTEX] = vertices
		array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)
		self.shape=array_mesh.create_convex_shape()
	
	def _ready(self):
		phi = self.phi
		vertices = PoolVector3Array()
		vertices.push_back(Vector3(0,0,0))
		vertices.push_back(Vector3(0,-2-3*phi,1+2*phi))
		vertices.push_back(Vector3(2+3*phi,-1-2*phi,0))
		vertices.push_back(Vector3(1+2*phi,0,2+3*phi))
		
		vertices.push_back(Vector3(2+3*phi,-3-5*phi,1+2*phi))
		vertices.push_back(Vector3(3+5*phi,-1-2*phi,2+3*phi))
		vertices.push_back(Vector3(1+2*phi,-2-3*phi,3+5*phi))
		
		vertices.push_back(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		#vertices.push_back(Vector3(-1-2*phi,0,2+3*phi))
		#vertices.push_back(Vector3(2+3*phi,-1-2*phi,0))
		#vertices.push_back(Vector3(0,-2+-3*phi,-1-2*phi))
		#vertices.push_back(Vector3(0,2+3*phi,1+2*phi))
		#vertices.push_back(Vector3(2+3*phi,1+2*phi,0))
		#vertices.push_back(Vector3(0,2+3*phi,-1-2*phi))
		#vertices.push_back(Vector3(-2-3*phi,1+2*phi,0))
		#vertices.push_back(Vector3(-1-2*phi,0,-2-3*phi))
		#vertices.push_back(Vector3(1+2*phi,0,-2-3*phi))
		array_mesh = ArrayMesh()
		arrays = builtins.Array()
		arrays.resize(ArrayMesh.ARRAY_MAX)
		arrays[ArrayMesh.ARRAY_VERTEX] = vertices
		array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)
		self.shape=array_mesh.create_convex_shape()
		
