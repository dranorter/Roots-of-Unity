extends KinematicBody

signal player_moved

var _mouse_motion = Vector2()

var velocity = Vector3(0,0,0)
var pos_history = []

onready var camera = $"../CameraBody/Camera"
onready var camera_body = $"../CameraBody"

onready var gravity = ProjectSettings.get_setting("physics/3d/default_gravity")


# Called when the node enters the scene tree for the first time.
func _ready():
	camera.current = true
	_mouse_motion = Vector2(0,0)


func _process(delta):
	# Mouse movement.
	_mouse_motion.y = clamp(_mouse_motion.y, -1550, 1550)
	transform.basis = Basis(Vector3(0, _mouse_motion.x * -0.001, 0))
	camera_body.transform.basis = Basis(Vector3(0, _mouse_motion.x * -0.001, 0))
	camera.transform.basis = Basis(Vector3(_mouse_motion.y * -0.001, 0, 0))
	
	if Input.is_action_pressed("ui_cancel"):
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	if Input.get_mouse_mode() == Input.MOUSE_MODE_VISIBLE:
		if Input.is_mouse_button_pressed(BUTTON_LEFT):
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func _physics_process(delta):
	velocity = Vector3(0,clamp(velocity.y,-1,10),0)
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
	
	# Use two movement types to manage gravity better
	move_and_slide(Vector3(velocity.x, 0, velocity.z))
	move_and_collide(Vector3(0,velocity.y,0))
	
	# Move the camera
	# TODO When the player moves slowly, e.g. sliding down a slope, this
	# mechanic causes framey-feeling motion. Need to snap the camera
	# to a straight line extrapolated from recent camera positions.
	# TODO Gotta smooth movement of player model as well.
	if camera_body.transform.origin.distance_to(transform.origin) > 0.1:
		camera_body.transform.origin = transform.origin
		emit_signal("player_moved", transform.origin)
	
	# Apply gravity next tic
	var on_floor = $Feet.is_colliding()
	
	if on_floor:
		# TODO I want to include a 'smooth movement' option where if your feet
		# are within range of the ground, there's no gravity at all.
		velocity.y = -0.05
	if not on_floor:
		velocity -= gravity*delta*Vector3(0,1,0)
	
	if on_floor and Input.is_action_pressed("ui_jump"):
		velocity.y = 1
	

func _input(event):
	if event is InputEventMouseMotion:
		if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
			_mouse_motion += event.relative
