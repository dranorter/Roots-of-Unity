from godot import exposed, export
from godot import *

COLOR = ResourceLoader.load("res://new_spatialmaterial.tres")

@exposed(tool=True)
class OblateMultiMesh(MultiMeshInstance):
	
	_render = False
	phi = 1.61803398874989484820458683
	pi = 3.14159265358979323846264338
	
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
		self._ready()
	
	def _ready(self):
		phi = self.phi
		pi = self.pi
		
		st = SurfaceTool()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		
		# Add a dummy normal, just to set up the format
		st.add_normal(Vector3(1,1,1))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(1+1*phi,1+2*phi,2+3*phi))
		st.add_vertex(Vector3(-1-2*phi,0,2+3*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(-1-2*phi,0,2+3*phi))
		st.add_vertex(Vector3(-1-2*phi,-2-3*phi,1+1*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(-1-2*phi,-2-3*phi,1+1*phi))
		st.add_vertex(Vector3(0,-2-3*phi,-1-2*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(0,-2-3*phi,-1-2*phi))
		st.add_vertex(Vector3(2+3*phi,-1-1*phi,-1-2*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(2+3*phi,-1-1*phi,-1-2*phi))
		st.add_vertex(Vector3(2+3*phi,1+2*phi,0))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(2+3*phi,1+2*phi,0))
		st.add_vertex(Vector3(1+1*phi,1+2*phi,2+3*phi))
		
		
		st.add_vertex(Vector3(1+1*phi,-1-1*phi,1+1*phi))
		st.add_vertex(Vector3(-1-2*phi,0,2+3*phi))
		st.add_vertex(Vector3(1+1*phi,1+2*phi,2+3*phi))
		
		st.add_vertex(Vector3(1+1*phi,-1-1*phi,1+1*phi))
		st.add_vertex(Vector3(-1-2*phi,-2-3*phi,1+1*phi))
		st.add_vertex(Vector3(-1-2*phi,0,2+3*phi))
		
		st.add_vertex(Vector3(1+1*phi,-1-1*phi,1+1*phi))
		st.add_vertex(Vector3(0,-2-3*phi,-1-2*phi))
		st.add_vertex(Vector3(-1-2*phi,-2-3*phi,1+1*phi))
		
		st.add_vertex(Vector3(1+1*phi,-1-1*phi,1+1*phi))
		st.add_vertex(Vector3(2+3*phi,-1-1*phi,-1-2*phi))
		st.add_vertex(Vector3(0,-2-3*phi,-1-2*phi))
		
		st.add_vertex(Vector3(1+1*phi,-1-1*phi,1+1*phi))
		st.add_vertex(Vector3(2+3*phi,1+2*phi,0))
		st.add_vertex(Vector3(2+3*phi,-1-1*phi,-1-2*phi))
		
		st.add_vertex(Vector3(1+1*phi,-1-1*phi,1+1*phi))
		st.add_vertex(Vector3(1+1*phi,1+2*phi,2+3*phi))
		st.add_vertex(Vector3(2+3*phi,1+2*phi,0))
		
		st.generate_normals()
		
		array_mesh = ArrayMesh()
		st.commit(array_mesh)
		
		
		self.multimesh = MultiMesh()
		# Set to 0 if we end up not using color format
		self.multimesh.set_color_format(MultiMesh.ColorFormat.COLOR_FLOAT)
		self.multimesh.transform_format = VisualServer.MULTIMESH_TRANSFORM_3D
		self.multimesh.mesh=array_mesh
		self.multimesh.instance_count = 2
		
		self.multimesh.set_instance_transform(0,Transform().translated(Vector3(-2-3*phi,-3-5*phi,1+2*phi)))
		self.multimesh.set_instance_transform(1,
						Transform().rotated(Vector3(0,-2-3*phi,1+2*phi).normalized(),
						4*pi/5).translated(Vector3(-2-3*phi,-3-5*phi,1+2*phi)))
		
		self.multimesh.mesh.surface_set_material(0,COLOR)
