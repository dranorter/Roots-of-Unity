from godot import exposed, export
from godot import *
import debugging

@exposed(tool=True)
class CSGMesh(CSGMesh):
	
	_render = False
	
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
		vertices = PoolVector3Array()
		vertices.push_back(Vector3(0,1,0))
		vertices.push_back(Vector3(1,0,0))
		vertices.push_back(Vector3(0,0,1))
		vertices.push_back(Vector3(1,0,0))
		vertices.push_back(Vector3(0,1,0))
		vertices.push_back(Vector3(1,1,2))
		array_mesh = ArrayMesh()
		arrays = builtins.Array()
		arrays.resize(ArrayMesh.ARRAY_MAX)
		arrays[ArrayMesh.ARRAY_VERTEX] = vertices
		array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
		self.mesh=array_mesh
	
	def _ready(self):
		self.generate()
		#packed_scene = PackedScene()
		#print(self.get_tree().get_current_scene())
		#packed_scene.pack(self.mesh)#self.get_tree().get_current_scene())
		
