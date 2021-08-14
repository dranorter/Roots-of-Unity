extends Spatial


# Declare member variables here. Examples:
# var a = 2
# var b = "text"


# Called when the node enters the scene tree for the first time.
func _ready():
	pass
	# TODO Get either of these working
	# Don't trap the mouse if loading in the background
	#if OS.is_window_focused():
	#	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	#
	# Respond to different keyboard layouts correctly
	#OS.keyboard_set_current_layout(1)

# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass


func _on_Player_player_moved(pos):
	$ChunkLoad/Chunk_Network.update_player_pos(pos, $ChunkLoad.transform)
