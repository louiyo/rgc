import numpy
from neuron import h, gui


class RGC(object):
	"""Two section cell, a soma and an axon,
	simulating a retinal ganglion cell"""
	def __init__(self, x=0, y=0, z=0):
		self.x,self.y,self.z = x,y,z
		self.create_sections()
		self.build_topology()
		self.build_subsets()
		self.define_geometry()
		self.define_biophysics()


	def create_sections(self):
		self.soma = h.Section(name = 'soma', cell = self)
		self.hillock = h.Section(name = 'hillock', cell = self)
		self.myelin1 = h.Section(name = 'myelin1', cell = self)
		self.myelin2 = h.Section(name = 'myelin2', cell = self)
		self.ranvier = h.Section(name = 'ranvier', cell = self)


	def build_topology(self):
		"""Connect the sections of the cell to build a tree."""
		self.hillock.connect(self.soma(1))
		self.myelin1.connect(self.hillock(1))
		self.ranvier.connect(self.myelin1(1))
		self.myelin2.connect(self.ranvier(1))


	def shape_3D(self):
		"""
		Set the default shape of the cell in 3D coordinates.
		"""
		lensoma = self.soma.L
		lenhillock = self.hillock.L
		lenmyelin = self.myelin1.L
		lenranvier = self.ranvier.L

		h.pt3dclear(sec=self.soma)
		h.pt3dadd(0+self.x, 	lensoma+self.y, 				0+self.z, self.soma.diam, sec = self.soma)
		h.pt3dadd(0+self.x, 	self.y,		0+self.z, self.soma.diam, sec = self.soma)

		h.pt3dclear(sec=self.hillock)
		h.pt3dadd(0+self.x, 			lensoma+self.y, 	0+self.z, self.hillock.diam, sec = self.hillock)
		h.pt3dadd(0+self.x+lenhillock, 	self.y, 	0+self.z, self.hillock.diam, sec = self.hillock)

		h.pt3dclear(sec=self.myelin1)
		h.pt3dadd(0+self.x+lenhillock, 				self.y, 	0+self.z, self.myelin1.diam, sec = self.myelin1)
		h.pt3dadd(0+lenmyelin+self.x+lenhillock, 	self.y, 	0+self.z, self.myelin1.diam, sec = self.myelin1)

		"""h.pt3dclear(sec=self.ranvier)
		h.pt3dadd(0+lenmyelin+self.x+lenhillock, 					self.y, 	0+self.z, self.ranvier.diam, sec = self.ranvier)
		h.pt3dadd(0+lenmyelin+lenranvier+self.x+lenhillock, 		self.y, 	0+self.z, self.ranvier.diam, sec = self.ranvier)

		h.pt3dclear(sec=self.myelin2)
		h.pt3dadd(0+lenmyelin+self.ranvier.L+self.x+lenhillock, 					self.y, 	0+self.z, self.myelin2.diam, sec = self.myelin2)
		h.pt3dadd(0+lenmyelin+self.ranvier.L+self.myelin2.L+self.x+lenhillock, 	self.y, 	0+self.z, self.myelin2.diam, sec = self.myelin2)"""



	def define_geometry(self):
		"""Set the 3D geometry of the cell."""
		self.soma.L = self.soma.diam = 12.6157 # microns
		self.soma.nseg 		= 11

		self.hillock.L 		= 14				   		
		self.hillock.diam 	= 5	                   
		self.hillock.nseg 	= 3

		self.myelin1.diam = self.myelin2.diam = 6
		self.myelin1.L 	  = self.myelin2.L = 75
		self.myelin1.nseg = self.myelin2.nseg = 11

		self.ranvier.diam 	= 3
		self.ranvier.L 		= 25
		
		self.shape_3D()


	def define_biophysics(self):

		self.stim = h.VClamp(self.soma(0.5))
		
		# The VClamp is a mecanism placed at a particular segment (here in soma(0.5))
		# that can be stimulated for a brief period of time, once in a simulation.
		# check method retinal_GC.stimulate()
		
		# Axial resistance in Ohm * cm
		self.soma.Ra, self.hillock.Ra, self.myelin1.Ra, self.myelin2.Ra, self.ranvier.Ra  = 110, 0.01, 70, 70, 0.01 
		# Membrane capacitance in micro Farads / cm^2
		self.soma.cm, self.hillock.cm, self.myelin1.cm, self.myelin2.cm, self.ranvier.cm  = 1, 1, 0.6, 0.6, 2
		# Insert active Hodgkin-Huxley current in the soma
		self.soma.insert('hh')
		for seg in self.soma:
		    seg.hh.gnabar = 0.070 	# Sodium conductance in S/cm2
		    seg.hh.gkbar = 0.018  	# Potassium conductance in S/cm2
		    seg.hh.gl = 0.00012   	# Leak conductance in S/cm2
		    seg.hh.el = -65    # Reversal potential in mV
		self.hillock.insert('hh')
		for seg in self.hillock:
		    seg.hh.gnabar = 0.150 	# Sodium conductance in S/cm2
		    seg.hh.gkbar = 0.018  		# Potassium conductance in S/cm2
		    seg.hh.gl = 0.00012   		# Leak conductance in S/cm2
		    seg.hh.el = -65	     	# Reversal potential in mV    ???
		self.myelin1.insert('hh')
		for seg in self.myelin1:
		    seg.hh.gnabar = 0.070 	# Sodium conductance in S/cm2
		    seg.hh.gkbar = 0.018  	# Potassium conductance in S/cm2
		    seg.hh.gl = 0.2   	# Leak conductance in S/cm2 
		    seg.hh.el = -65     # Reversal potential in mV    ???
		self.myelin2.insert('hh')
		for seg in self.myelin2:
		    seg.hh.gnabar = 0.070 	# Sodium conductance in S/cm2
		    seg.hh.gkbar = 0.018  	# Potassium conductance in S/cm2
		    seg.hh.gl = 0.2   	# Leak conductance in S/cm2 
		    seg.hh.el = -65     # Reversal potential in mV    ???
		self.ranvier.insert('hh')
		for seg in self.ranvier:
		    seg.hh.gnabar = 0.400 	# Sodium conductance in S/cm2
		    seg.hh.gkbar =   0.099	# Potassium conductance in S/cm2
		    seg.hh.gl = 0.12   	# Leak conductance in S/cm2
		    seg.hh.el = -65     # Reversal potential in mV
		# Insert passive current in the soma"""
		self.soma.insert('pas')
		for seg in self.soma:
		    seg.pas.g = 0.001  # Passive conductance in S/cm2
		    seg.pas.e = -65    # Leak reversal potential mV
		    #  			^^^^NOT SURE if useful...


		# A COMPLETER AVEC LES DONNEES
	def build_subsets(self):
		"""Build subset lists. For now we define 'all'."""
		self.all = h.SectionList()
		self.all.wholetree(sec=self.soma)


	def move(self,x,y,z):
		for sec in self.all:
			for i in range(sec.n3d()):
				x_new = sec.x3d(i) + x
				y_new = sec.y3d(i) + y
				z_new = sec.z3d(i) + z
				h.pt3dchange(i, x_new, y_new, z_new, sec.diam3d(i), sec=sec)

	def stimulate(self, voltage, duration = 1):


		"""self.stim.delay = 5
		self.stim.amp = voltage
		self.stim.dur = duration"""

		self.stim.amp[0] = 0
		self.stim.dur[0] = 5
		self.stim.amp[1] = voltage
		self.stim.dur[1] = duration
		self.stim.amp[2] = self.stim.dur[2] = 0


		"""self.stim.amp1 = 0
		self.stim.dur1 = 5
		self.stim.amp2 = voltage
		self.stim.dur2 = duration
		self.stim.amp3 =  self.stim.dur3 = 0"""
