extends KinematicBody

signal player_moved

var _mouse_motion = Vector2()
var joy_input = Vector2(0,0)

var velocity = Vector3(0,0,0)
var pos_history = []

onready var camera = $"../CameraBody/Camera"
onready var camera_body = $"../CameraBody"
onready var camera_grab = $"../CameraBody/Camera/grab"
onready var interact_ray = $"../CameraBody/Camera/interact"

onready var gravity = ProjectSettings.get_setting("physics/3d/default_gravity")


# Called when the node enters the scene tree for the first time.
func _ready():
	camera.current = true
	_mouse_motion = Vector2(0,0)


func _process(delta):
	# Mouse movement.
	#if Input.is_action_pressed("ui_look_left") or Input.is_action_pressed("ui_look_right"):
	_mouse_motion.x += 50*pow(Input.get_joy_axis(0,0),2)*sign(Input.get_joy_axis(0,0))
	#if Input.is_action_pressed("ui_look_down") or Input.is_action_pressed("ui_look_up"):
	_mouse_motion.y += 50*pow(Input.get_joy_axis(0,1),2)*sign(Input.get_joy_axis(0,1))
	_mouse_motion.y = clamp(_mouse_motion.y, -1550, 1550)
	transform.basis = Basis(Vector3(0, _mouse_motion.x * -0.001, 0))
	camera_body.transform.basis = Basis(Vector3(0, _mouse_motion.x * -0.001, 0))
	camera.transform.basis = Basis(Vector3(_mouse_motion.y * -0.001, 0, 0))
	if Input.is_action_pressed("ui_cancel"):
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	if Input.get_mouse_mode() == Input.MOUSE_MODE_VISIBLE:
		if Input.is_mouse_button_pressed(BUTTON_LEFT):
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	
	var interactable = interact_ray.get_collider()
	print(interactable)

func _physics_process(delta):
	var input_moved = false
	var on_floor = $Feet.is_colliding()
	# Traction from floor or what we're looking at.
	var traction = on_floor or camera_grab.is_colliding()
	# When not in "grabbing range" of a surface, velocity is not reset.
	# Just clamp it (air friction).
	if not traction:
		velocity = Vector3(clamp(velocity.x,-4,4),clamp(velocity.y,0,0),clamp(velocity.z,-4,4))
	else:
		# When we are in "grabbing range", velocity is damped, and controls respond.
		#velocity = velocity*pow(0.1,15*delta)
		# Damping to 0 for now; gotta find something that feels ok.
		# TODO: When traction comes from something other than feet, vertical damping should be less.
		velocity = Vector3(0,0,0)
		if Input.is_action_pressed("ui_jump"):# Weird to have it in two places.
			velocity.y = 1
		
		if Input.is_action_pressed("ui_right"):
			velocity += 10*transform.basis[0]
			input_moved = true
		if Input.is_action_pressed("ui_left"):
			velocity -= 10*transform.basis[0]
			input_moved = true
		if Input.is_action_pressed("ui_down"):
			velocity += 10*transform.basis[2]
			input_moved = true
		if Input.is_action_pressed("ui_up"):
			# We move the direction the player is looking
			velocity -= 10*camera.global_transform.basis[2]
			input_moved = true
	if Input.is_action_pressed("ui_page_up"):
		velocity += 10*transform.basis[1]
		input_moved = true
	if Input.is_action_pressed("ui_page_down"):
		velocity -= 10*transform.basis[1]
		input_moved = true

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
	
	if on_floor:
		# TODO I want to include a 'smooth movement' option where if your feet
		# are within range of the ground, there's no gravity at all.
		velocity.y = -0.05
		# If you're standing still, your feet stabilize you
		if not input_moved:
			velocity.y = 0
	if not on_floor:
		velocity -= gravity*delta*Vector3(0,1,0)
	
	if on_floor and Input.is_action_pressed("ui_jump"):
		velocity.y = 1
	

func _input(event):
	if event is InputEventMouseMotion:
		if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
			_mouse_motion += event.relative
