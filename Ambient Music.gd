extends AudioStreamPlayer


func _ready():
	#playing = true
	pass


func _process(delta):
	if playing == false:
		if randf() < 0.0001*delta:
			playing = true
