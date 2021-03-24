tool
extends CSGMesh


# Declare member variables here. Examples:
# var a = 2
# var b = "text"


# Called when the node enters the scene tree for the first time.
func _ready():
	print("Trying in gdscript")
	self.generate()

func generate():
	var vertices = PoolVector3Array()
	vertices.push_back(Vector3(0,1,0))
	vertices.push_back(Vector3(1,0,0))
	vertices.push_back(Vector3(0,0,1))
	var arr_mesh = ArrayMesh.new()
	var arrays = []
	arrays.resize(ArrayMesh.ARRAY_MAX)
	arrays[ArrayMesh.ARRAY_VERTEX] = vertices
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	mesh = arr_mesh
# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass
