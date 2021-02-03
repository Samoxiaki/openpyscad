import math


class OpsMath:
	"""
	This module provides access to the mathematical functions defined by the C standard. Modified/Extended to operate with openSCAD functionalities.
	"""
	
	__cos_degrees = {
		0:		1.0,
		30: 	math.sqrt(3)/2,
		45:		math.sqrt(2)/2,
		60:		0.5,
		90:		0.0,
		120:	-0.5,
		135:	-math.sqrt(2)/2,
		150:	-math.sqrt(3)/2,
		180:	-1.0,
		210:	-math.sqrt(3)/2,
		225:	-math.sqrt(2)/2,
		240:	-0.5,
		270:	0.0,
		300:	0.5,
		315:	math.sqrt(2)/2,
		330:	math.sqrt(3)/2,
		360:	1.0
	}
	__sin_degrees = {
		0:		0.0,
		30: 	0.5,
		45:		math.sqrt(2)/2,
		60:		math.sqrt(3)/2,
		90:		1.0,
		120:	math.sqrt(3)/2,
		135:	math.sqrt(2)/2,
		150:	0.5,
		180:	0.0,
		210:	-0.5,
		225:	-math.sqrt(2)/2,
		240:	-math.sqrt(3)/2,
		270:	-1.0,
		300:	-math.sqrt(3)/2,
		315:	-math.sqrt(2)/2,
		330:	-0.5,
		360:	0.0
	}
	
	__tan_degrees = {
		0:		0.0,
		30: 	1/math.sqrt(3),
		45:		1.0,
		60:		math.sqrt(3),
		90:		math.inf,
		120:	-math.sqrt(3),
		135:	-1,
		150:	-1/math.sqrt(3),
		180:	0.0,
		210:	1/math.sqrt(3),
		225:	1,
		240:	math.sqrt(3),
		270:	math.inf,
		300:	-math.sqrt(3),
		315:	-1.0,
		330:	-1/math.sqrt(3),
		360:	0.0
	}
	
	@staticmethod
	def __reverse_chk(value, degree_list):
		"""
		Check values from degree table in reverse for asin(), acos() and atan()  
		"""
		for k, v in degree_list.items():
			try:
				if(math.isclose(v, value)):
					return k
			except:
				return None
		return None 
	
	@staticmethod
	def acos(x):
		"""
		Return the arc cosine (measured in degrees) of x.
		"""
		x_chk = OpsMath.__reverse_chk(x, OpsMath.__cos_degrees)
		return x_chk if (x_chk is not None) else math.degrees(math.acos(x))
		
	@staticmethod
	def asin(x):
		"""
		Return the arc sine (measured in degrees) of x.
		"""
		x_chk = OpsMath.__reverse_chk(x, OpsMath.__sin_degrees)
		return x_chk if (x_chk is not None) else math.degrees(math.asin(x))
		
	@staticmethod
	def atan(x):
		"""
		Return the arc tangent (measured in degrees) of x.
		"""
		x_chk = OpsMath.__reverse_chk(x, OpsMath.__tan_degrees)
		return x_chk if (x_chk is not None) else math.degrees(math.atan(x))
		
	@staticmethod	
	def atan2(y, x):
		"""
		Return the arc tangent (measured in degrees) of y/x.
			
		Unlike atan(y/x), the signs of both x and y are considered.	
		"""
		res = math.degrees(math.atan2(y, x))
		for k in OpsMath.__tan_degrees.keys():
			try:
				if(math.isclose(k, res)):
					return k
				elif(math.isclose(-k, res)):
					return -k
			except:
				return res
				
		return res
	
	@staticmethod
	def cos(x):
		"""
		Return the cosine of x (measured in degrees).
		"""
		x_chk = abs(x % 360)
		return OpsMath.__cos_degrees.get(x_chk, math.cos(math.radians(x)))
	
	@staticmethod
	def sin(x):
		"""
		Return the sine of x (measured in degrees).
		"""
		x_chk = abs(x % 360)
		return OpsMath.__sin_degrees.get(x_chk, math.sin(math.radians(x)))
	
	@staticmethod
	def tan(x):
		"""
		Return the tangent of x (measured in degrees).
		"""
		x_chk = abs(x % 360)
		return OpsMath.__tan_degrees.get(x_chk,  math.tan(math.radians(x)))

	@staticmethod
	def inv_hypot_y(hypot_l, angle):
		"""
		Return Y projection given a hypotenuse and an angle.
		"""
		return hypot_l * OpsMath.sin(angle)
		
	@staticmethod
	def inv_hypot_x(hypot_l, angle):
		"""
		Return X projection given a hypotenuse and an angle.
		"""
		return hypot_l * OpsMath.cos(angle)
	
	@staticmethod
	def inv_hypot(hypot_l, angle):
		"""
		Return X, Y projection given a hypotenuse and an angle.
		"""
		return (hypot_l * OpsMath.cos(angle), hypot_l * OpsMath.sin(angle))
	
	@staticmethod
	def bisector_angle(a1, a2):
		"""
		Return angle between two given angles.
		"""
		return (abs(a1 % 360) + abs(a2 % 360))/2
	
	@staticmethod
	def intersect_2lines(p1, a1, p2, a2):
		"""
		Return intersection points (x, y) between 2 lines.
		If lines are parallel, returns (None, None).
		Line is defined by a point (x, y) and an angle (degrees).
		"""
		if(not (isinstance(p1, (list, tuple)) and len(p1)>=2)):
			raise TypeError("p1 must be an [x, y] point.")
		elif(not (isinstance(p2, (list, tuple)) and len(p2)>=2)):
			raise TypeError("p2 must be an [x, y] point.")

		p1x = p1[0]
		p1y = p1[1]
		p2x = p2[0]
		p2y = p2[1]
		m1  = OpsMath.tan(a1)
		m2  = OpsMath.tan(a2)
		
		# eq1: y = (x - p1x) * m1 + p1y | y = m1*x + (p1y - m1*p1x)
		# eq2: y = (x - p2x) * m2 + p2y | y = m2*x + (p2y - m2*p2x) 
		
		# Parallel lines
		if(m1 == m2):
			return None
			
		# Find intersection point
		# If one of the lines has tangent==inf, this line is vertical.
		# Else, general rule
		# x = (d-c)/(a-b)
		# y = (ad-bc)/(a-b)
		# ix = ((p2y - m2*p2x) - (p1y - m1*p1x))/(m1-m2)
		# iy = (m1*(p2y - m2*p2x) - m2*(p1y - m1*p1x))/(m1-m2)
		if(m1 == OpsMath.inf):
			ix = p1x
			iy = (ix - p2x) * m2 + p2y
		elif(m2 == OpsMath.inf):
			ix = p2x
			iy = (ix - p1x) * m1 + p1y
		else:
			ix = ((p2y - m2*p2x) - (p1y - m1*p1x))/(m1-m2)
			iy = (m1*(p2y - m2*p2x) - m2*(p1y - m1*p1x))/(m1-m2)
			
		return (ix, iy)
	
	@staticmethod
	def pivot_2p(p1, a1, p2, a2, r):
		"""
		Return pivot point [x,y], arc angle, angle offset, tangent point on line 1 [x,y], and tangent point on line 2 [x,y] between two given points, their angles and the radius to the pivot point.
		If lines are parallel, returns None, None, None, None, None
		[2 line intersection tangent]
		"""
		
		ix, iy = OpsMath.intersect_2lines(p1, a1, p2, a2)
		if(ix is None or iy is None):
			# Lines are parallel
			return (None, None, None, None, None)
		
		p1x = p1[0]
		p1y = p1[1]
		p2x = p2[0]
		p2y = p2[1]

		
		# Create bisector line
		# eqbis: y = (x - ix)*m_bis + iy 
		angle_bis = OpsMath.bisector_angle(a1, a2)
		angle_bis_rel_a1 = abs(angle_bis - a1)  
		angle_bis_rel_a2 = abs(angle_bis - a2)  
		hypot_len = r / OpsMath.sin(angle_bis_rel_a1)
		
		# Find pivot point
		d_from_bis_x = hypot_len * OpsMath.cos(angle_bis)
		d_from_bis_y = hypot_len * OpsMath.sin(angle_bis)
		pivot_x = ix + d_from_bis_x
		pivot_y = iy + d_from_bis_y
		
		# TODO: Utilizar las rectas con angulos normales.
		# Obtener el angulo del arco a partir de ahi
		# Find tangent point on line 1
		d_from_line1 = hypot_len * OpsMath.cos(angle_bis_rel_a1)
		d_from_line1x = d_from_line1 * OpsMath.cos(a1)
		d_from_line1y = d_from_line1 * OpsMath.sin(a1)
		tangent_x1 = ix + d_from_line1x
		tangent_y1 = iy + d_from_line1y
		
		# Find tangent point on line 2
		d_from_line2 = hypot_len * OpsMath.cos(angle_bis_rel_a2)
		d_from_line2x = d_from_line2 * OpsMath.cos(a2)
		d_from_line2y = d_from_line2 * OpsMath.sin(a2)
		tangent_x2 = ix + d_from_line2x
		tangent_y2 = iy + d_from_line2y
		
		# Find arc angle
		hypot_btw_tangents = OpsMath.hypot((tangent_x2-tangent_x1), (tangent_y2-tangent_y1))
		arc_angle = (90 - OpsMath.acos((hypot_btw_tangents/2)/r))*2
		arc_angle_offset = OpsMath.atan2((tangent_x1 - pivot_x), (tangent_y1 - pivot_y))
		
		return ([pivot_x, pivot_y], arc_angle, arc_angle_offset, [tangent_x1,tangent_y1], [tangent_x2,tangent_y2])
		
		
		
		
	
	acosh = math.acosh

	asinh = math.asinh

	atanh = math.atanh

	ceil = math.ceil

	copysign = math.copysign

	cosh = math.cosh

	degrees = math.degrees

	erf = math.erf

	exp = math.exp

	expm1 = math.expm1

	fabs = math.fabs

	factorial = math.factorial

	floor = math.floor

	fmod = math.fmod

	frexp = math.frexp

	fsum = math.fsum

	gamma = math. gamma

	gcd = math.gcd

	hypot = math.hypot

	isclose = math.isclose

	isfinite = math.isfinite
	 
	isinf = math.isinf

	isnan = math.isnan

	ldexp = math.ldexp

	lgamma = math.lgamma

	log = math.log

	log10 = math.log10

	log1p = math.log1p

	log2 = math.log2
	 
	modf = math.modf
	 
	pow = math.pow
	 
	radians = math.radians
	 
	remainder = math.remainder

	sinh = math.sinh
	 
	sqrt = math.sqrt

	tanh = math.tanh
	 
	trunc = math.trunc

	# Constants

	e = math.e
	inf = math.inf
	nan = math.nan
	pi = math.pi
	tau = math.tau

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
