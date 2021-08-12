extends KinematicBody

signal player_moved

var _mouse_motion = Vector2()

var velocity


# Called when the node enters the scene tree for the first time.
func _ready():
	#$"..Chunkload".remove_child($"..ChunkLoad/Camera")
	$Camera.current = true
	_mouse_motion = Vector2(0,0)


func _process(delta):
	# Mouse movement.
	_mouse_motion.y = clamp(_mouse_motion.y, -1550, 1550)
	transform.basis = Basis(Vector3(0, _mouse_motion.x * -0.001, 0))
	$Camera.transform.basis = Basis(Vector3(_mouse_motion.y * -0.001, 0, 0))
	
	if Input.is_action_pressed("ui_cancel"):
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	if Input.get_mouse_mode() == Input.MOUSE_MODE_VISIBLE:
		if Input.is_mouse_button_pressed(BUTTON_LEFT):
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func _physics_process(delta):
	var velocity = Vector3(0,0,0)
	if Input.is_action_pressed("ui_right"):
		velocity += 10*transform.basis[0]
	if Input.is_action_pressed("ui_left"):
		velocity -= 10*transform.basis[0]
	if Input.is_action_pressed("ui_down"):
		velocity += 10*transform.basis[2]
	if Input.is_action_pressed("ui_up"):
		velocity -= 10*transform.basis[2]
	if Input.is_action_pressed("ui_page_up"):
		velocity += 10*transform.basis[1]
	if Input.is_action_pressed("ui_page_down"):
		velocity -= 10*transform.basis[1]
	move_and_slide(velocity)
	#translation += velocity*delta
#	emit_signal("player_moved", Position3D)

func _input(event):
	if event is InputEventMouseMotion:
		if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
			_mouse_motion += event.relative
