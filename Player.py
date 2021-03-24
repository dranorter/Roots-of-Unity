# Explicit is better than implicit
from godot import exposed, export, Vector2, Node2D, ResourceLoader

#WEAPON_RES = ResourceLoader.load("res://weapon.tscn")
SPEED = Vector2(10, 10)

@exposed(tool=False)
class Player(Node2D):
		"""
		This is the file's main class which will be made available to Godot. This
		class must inherit from `godot.Node` or any of its children (e.g.
		`godot.KinematicBody`).

		Because Godot scripts only accept file paths, you can't have two `exposed` classes in the same file.
		"""
		# Exposed class can define some attributes as export(<type>) to achieve
		# similar goal than GDSscript's `export` keyword
		name = export(str)
		
		checkbox = export(bool)

		# Can export property as well
		@export(int)
		@property
		def age(self):
				return self._age

		@age.setter
		def age(self, value):
				self._age = value

		# All methods are exposed to Godot
		def talk(self, msg):
				print(f"I'm saying {msg}")

		def _ready(self):
				# Don't confuse `__init__` with Godot's `_ready`!
				#self.weapon = WEAPON_RES.instance()
				self._age = 42
				# Of course you can access property & methods defined in the parent
				name = self.get_name()
				print(f"{name} position x={self.position.x}, y={self.position.y}")

		def _process(self, delta):
				self.position += SPEED * delta

		...


class Helper:
		"""
		Other classes are considered helpers and cannot be called from outside
		Python. However they can be imported from another python module.
		"""
		...
