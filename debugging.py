from godot import *

#import os
#print("Python PID: "+str(os.getpid()))

@exposed(tool=True)
class debugging(Node):
	broken_once = False

	def breakpoint():
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
