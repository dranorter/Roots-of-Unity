from godot import Engine
from godot import *

#import os
#print("Python PID: "+str(os.getpid()))

@exposed(tool=True)
class debugging(Node):
	broken_once = False

	def breakpoint():
		if Engine.editor_hint():
			print("Skipping breakpoint in editor mode")
		else:
			print("Awaiting debugger connection")
			import rpdb2
			rpdb2.start_embedded_debugger('notaflex',depth=1)
	
	def editor_breakpoint():
		if not Engine.editor_hint():
			pass
		else:
			print("Awaiting debugger connection")
			import rpdb2
			rpdb2.start_embedded_debugger('notaflex',depth=1)

	def break_once():
		global broken_once
		if not broken_once:
			broken_once = True
			import rpdb2
			rpdb2.start_embedded_debugger('notaflex',depth=1)
	
	def test_tool():
		print("tested")
