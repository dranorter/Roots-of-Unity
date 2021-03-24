tool
extends CollisionShape


# Declare member variables here. Examples:
# var a = 2
# var b = "text"
#export(Mesh) var mesh

# Called when the node enters the scene tree for the first time.
func _ready():
	#var packed_scene = PackedScene.new()
	#packed_scene.pack(get_tree().get_current_scene())
	generate()

func generate():
	shape = ArrayMesh.new().create_convex_shape()
# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass
