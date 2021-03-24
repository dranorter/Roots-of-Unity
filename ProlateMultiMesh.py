from godot import exposed, export
from godot import *
#import debugging

WATER = ResourceLoader.load("res://assets/maujoe.basic_water_material/materials/basic_water_material.material")
COLOR = ResourceLoader.load("res://new_spatialmaterial.tres")

@exposed(tool=True)
class ProlateMultiMesh(MultiMeshInstance):
	
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
		st.add_vertex(Vector3(0,-2-3*phi,1+2*phi))
		st.add_vertex(Vector3(2+3*phi,-3-5*phi,1+2*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(2+3*phi,-3-5*phi,1+2*phi))
		st.add_vertex(Vector3(2+3*phi,-1-2*phi,0))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(2+3*phi,-1-2*phi,0))
		st.add_vertex(Vector3(3+5*phi,-1-2*phi,2+3*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(3+5*phi,-1-2*phi,2+3*phi))
		st.add_vertex(Vector3(1+2*phi,0,2+3*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(1+2*phi,0,2+3*phi))
		st.add_vertex(Vector3(1+2*phi,-2-3*phi,3+5*phi))
		
		st.add_vertex(Vector3(0,0,0))
		st.add_vertex(Vector3(1+2*phi,-2-3*phi,3+5*phi))
		st.add_vertex(Vector3(0,-2-3*phi,1+2*phi))
		
		st.add_vertex(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		st.add_vertex(Vector3(0,-2-3*phi,1+2*phi))
		st.add_vertex(Vector3(1+2*phi,-2-3*phi,3+5*phi))
		
		st.add_vertex(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		st.add_vertex(Vector3(1+2*phi,-2-3*phi,3+5*phi))
		st.add_vertex(Vector3(1+2*phi,0,2+3*phi))
		
		st.add_vertex(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		st.add_vertex(Vector3(1+2*phi,0,2+3*phi))
		st.add_vertex(Vector3(3+5*phi,-1-2*phi,2+3*phi))
		
		st.add_vertex(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		st.add_vertex(Vector3(3+5*phi,-1-2*phi,2+3*phi))
		st.add_vertex(Vector3(2+3*phi,-1-2*phi,0))
		
		st.add_vertex(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		st.add_vertex(Vector3(2+3*phi,-1-2*phi,0))
		st.add_vertex(Vector3(2+3*phi,-3-5*phi,1+2*phi))
		
		st.add_vertex(Vector3(3+5*phi,-3-5*phi,3+5*phi))
		st.add_vertex(Vector3(2+3*phi,-3-5*phi,1+2*phi))
		st.add_vertex(Vector3(0,-2-3*phi,1+2*phi))
		
		st.generate_normals()
		
		array_mesh = ArrayMesh()
		
		#arrays = builtins.Array()
		#arrays.resize(ArrayMesh.ARRAY_MAX)
		#arrays[ArrayMesh.ARRAY_VERTEX] = vertices
		#array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLE_FAN, arrays)
		st.commit(array_mesh)
		
		
		#array_mesh.regen_normalmaps()
		
		#vertices = PoolVector3Array()
		self.multimesh = MultiMesh()
		# Set to 0 if we end up not using color format
		self.multimesh.set_color_format(MultiMesh.ColorFormat.COLOR_FLOAT)
		self.multimesh.transform_format = VisualServer.MULTIMESH_TRANSFORM_3D
		self.multimesh.mesh=array_mesh
		self.multimesh.instance_count = 21
		
		flower1 = Transform().rotated(Vector3(0,-2-3*phi,1+2*phi).normalized(),2*pi/5)
		self.multimesh.set_instance_transform(1,flower1)
		self.multimesh.set_instance_transform(2,flower1 * flower1)
		self.multimesh.set_instance_transform(3,flower1 * flower1 * flower1)
		self.multimesh.set_instance_transform(4,flower1 * flower1 * flower1 * flower1)
		
		flower2 = Transform().rotated(Vector3(2+3*phi,-1-2*phi,0).normalized(),2*pi/5)
		self.multimesh.set_instance_transform(5,flower2 * flower2)
		self.multimesh.set_instance_transform(6,flower2 * flower2 * flower2)
		
		flower3 = Transform().rotated(Vector3(1+2*phi,0,2+3*phi).normalized(),2*pi/5)
		self.multimesh.set_instance_transform(7,flower3)
		self.multimesh.set_instance_transform(8,flower3 * flower3)
		self.multimesh.set_instance_transform(9,flower3 * flower3 * flower3)
		
		
		self.multimesh.set_instance_transform(10,flower1 * flower3 * flower3)
		self.multimesh.set_instance_transform(11,flower1 * flower1 * flower3)
		self.multimesh.set_instance_transform(12,flower1 * flower1 * flower3 * flower3)
		self.multimesh.set_instance_transform(13,flower1 * flower1 * flower3 * flower3 * flower3)
		self.multimesh.set_instance_transform(14,flower1 * flower1 * flower3 * flower3 * flower3 * flower2)
		
		self.multimesh.set_instance_transform(15,flower3 * flower2 * flower2)
		self.multimesh.set_instance_transform(16,flower3 * flower2 * flower2 * flower2)
		self.multimesh.set_instance_transform(17,flower3 * flower2 * flower2 * flower1)
		self.multimesh.set_instance_transform(18,flower3 * flower2 * flower2 * flower1 * flower1)
		self.multimesh.set_instance_transform(19,flower3 * flower2 * flower2 * flower1 * flower1 * flower1)
		
		# We assign colors and name transformations to keep 
		# track of the different directions.
		
		dir0 = Transform()
		self.multimesh.set_instance_color(0,Color(1,1,1))
		dir1 = flower1
		self.multimesh.set_instance_color(1,Color(0,0,0))
		dir2 = flower1 * flower1
		self.multimesh.set_instance_color(2,Color(1,0,0))
		dir3 = flower1 * flower1 * flower1
		self.multimesh.set_instance_color(3,Color(0,1,0))
		dir4 = flower1 * flower1 * flower1 * flower1
		self.multimesh.set_instance_color(4,Color(0,0,1))
		dir5 = flower2 * flower2
		self.multimesh.set_instance_color(5,Color(1,0,1))
		dir6 = flower2 * flower2 * flower2
		self.multimesh.set_instance_color(6,Color(0,1,1))
		dir7 = flower3
		self.multimesh.set_instance_color(7,Color(1,1,0))
		dir8 = flower3 * flower3
		self.multimesh.set_instance_color(8,Color(0.5,0.5,0.5))
		dir9 = flower3 * flower3 * flower3
		self.multimesh.set_instance_color(9,Color(1,0.5,0))
		dir10 = flower1 * flower3 * flower3
		self.multimesh.set_instance_color(10,Color(0.5,0,1))
		dir11 = flower1 * flower1 * flower3
		self.multimesh.set_instance_color(11,Color(0,1,0.5))
		dir12 = flower1 * flower1 * flower3 * flower3
		self.multimesh.set_instance_color(12,Color(0,0,0.5))
		dir13 = flower1 * flower1 * flower3 * flower3 * flower3
		self.multimesh.set_instance_color(13,Color(0,0.5,0))
		dir14 = flower1 * flower1 * flower3 * flower3 * flower3 * flower2
		self.multimesh.set_instance_color(14,Color(0.5,0,0))
		dir15 = flower3 * flower2 * flower2
		self.multimesh.set_instance_color(15,Color(0.5,0.25,0))
		dir16 = flower3 * flower2 * flower2 * flower2
		self.multimesh.set_instance_color(16,Color(0.7,0.45,0.2))
		dir17 = flower3 * flower2 * flower2 * flower1
		self.multimesh.set_instance_color(17,Color(1,0.5,0.5))
		dir18 = flower3 * flower2 * flower2 * flower1 * flower1
		self.multimesh.set_instance_color(18,Color(0.5,1,0.5))
		dir19 = flower3 * flower2 * flower2 * flower1 * flower1 * flower1
		self.multimesh.set_instance_color(19,Color(0.5,0.5,1))
		
		# Begin non-fixed chunk elements:
		
		self.multimesh.set_instance_transform(20,(flower3*flower3*flower3).translated(Vector3(2+3*phi,3+5*phi,-1-2*phi)))
		self.multimesh.set_instance_color(20,Color(1,0.5,0))
		
		self.multimesh.mesh.surface_set_material(0,COLOR)
