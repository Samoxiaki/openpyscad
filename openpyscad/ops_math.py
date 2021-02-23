import math
from decimal import Decimal, localcontext, ROUND_HALF_UP
import copy
from abc import ABCMeta, abstractmethod
from .util import Validator
from functools import wraps


__all__ = ["OpsMath", "MathPoint", "MathVector", "MathLine"]

# 卐卐卐卐卐卐卐卐卐 begin (Methods to operate with Decimal values)

__PRECISSION = 48
__ROUND_PRECISSION = 6
__REL_TOLERANCE = 1e-12

def decimal_cast(value):
	return Decimal(str(value))

def precise_method(func):
	@wraps(func)
	def rfunc(*args, **kwargs):
		global __PRECISSION
		with localcontext() as ctx:
			ctx.prec = __PRECISSION
			ctx.rounding = ROUND_HALF_UP
			return func(*args, **kwargs)
	return rfunc
		
# Operate as Decimal, return a float
@precise_method
def decimal_add(value_a, value_b):
	return float(decimal_cast(value_a) + decimal_cast(value_b))

@precise_method
def decimal_sub(value_a, value_b):
	return float(decimal_cast(value_a) - decimal_cast(value_b))
	
@precise_method
def decimal_mul(value_a, value_b):
	return float(decimal_cast(value_a) * decimal_cast(value_b))
	
@precise_method
def decimal_div(value_a, value_b):
	return float(decimal_cast(value_a) / decimal_cast(value_b))

@precise_method
def decimal_pow(value_a, value_b):
	return float(decimal_cast(value_a) ** decimal_cast(value_b))
	
@precise_method
def decimal_aproximate(value):
	global __REL_TOLERANCE, __ROUND_PRECISSION
	rounded_value = round(value, __ROUND_PRECISSION)
	if(math.isclose(rounded_value, value, rel_tol=__REL_TOLERANCE)):
		return Decimal(rounded_value)
	else:
		return decimal_cast(value)
	
# 卐卐卐卐卐卐卐卐卐 end

class OpsMath:
	"""
	This module provides access to the mathematical functions defined by the C standard. Modified/Extended to operate with openSCAD functionalities.
	"""
	
	__cos_degrees = {
		0:		Decimal(1),
		30: 	Decimal(3).sqrt()/Decimal(2),
		45:		Decimal(2).sqrt()/Decimal(2),
		60:		Decimal(0.5),
		90:		Decimal(0),
		120:	Decimal(-0.5),
		135:	-Decimal(2).sqrt()/Decimal(2),
		150:	-Decimal(3).sqrt()/Decimal(2),
		180:	Decimal(-1),
		210:	-Decimal(3).sqrt()/Decimal(2),
		225:	-Decimal(2).sqrt()/Decimal(2),
		240:	Decimal(-0.5),
		270:	Decimal(0),
		300:	Decimal(0.5),
		315:	Decimal(2).sqrt()/Decimal(2),
		330:	Decimal(3).sqrt()/Decimal(2),
		360:	Decimal(1)
	}
	__sin_degrees = {
		0:		Decimal(0),
		30: 	Decimal(0.5),
		45:		Decimal(2).sqrt()/Decimal(2),
		60:		Decimal(3).sqrt()/Decimal(2),
		90:		Decimal(1),
		120:	Decimal(3).sqrt()/Decimal(2),
		135:	Decimal(2).sqrt()/Decimal(2),
		150:	Decimal(0.5),
		180:	Decimal(0),
		210:	Decimal(-0.5),
		225:	-Decimal(2).sqrt()/Decimal(2),
		240:	-Decimal(3).sqrt()/Decimal(2),
		270:	Decimal(-1),
		300:	-Decimal(3).sqrt()/Decimal(2),
		315:	-Decimal(2).sqrt()/Decimal(2),
		330:	Decimal(-0.5),
		360:	Decimal(0)
	}
	
	__tan_degrees = {
		0:		Decimal(0),
		30: 	Decimal(1)/Decimal(3).sqrt(),
		45:		Decimal(1),
		60:		Decimal(3).sqrt(),
		90:		Decimal(math.inf),
		120:	-Decimal(3).sqrt(),
		135:	Decimal(-1),
		150:	Decimal(-1)/Decimal(3).sqrt(),
		180:	Decimal(0),
		210:	Decimal(1)/Decimal(3).sqrt(),
		225:	Decimal(1),
		240:	Decimal(3).sqrt(),
		270:	Decimal(math.inf),
		300:	-Decimal(3).sqrt(),
		315:	Decimal(-1),
		330:	Decimal(-1)/Decimal(3).sqrt(),
		360:	Decimal(0)
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
	@precise_method
	def acos(x, as_decimal=False):
		"""
		Return the arc cosine (measured in degrees) of x.
		"""
		x_chk = OpsMath.__reverse_chk(x, OpsMath.__cos_degrees)
		if(as_decimal):
			return x_chk if (x_chk is not None) else OpsMath.degrees(decimal_cast(math.acos(x)), as_decimal=True)
		else:
			return float(x_chk) if (x_chk is not None) else OpsMath.degrees(decimal_cast(math.acos(x)))
		
	@staticmethod
	@precise_method
	def asin(x, as_decimal=False):
		"""
		Return the arc sine (measured in degrees) of x.
		"""
		x_chk = OpsMath.__reverse_chk(x, OpsMath.__sin_degrees)
		if(as_decimal):
			return x_chk if (x_chk is not None) else OpsMath.degrees(decimal_cast(math.asin(x)), as_decimal=True)
		else:
			return float(x_chk) if (x_chk is not None) else OpsMath.degrees(decimal_cast(math.asin(x)))
		
	@staticmethod
	@precise_method
	def atan(x, as_decimal=False):
		"""
		Return the arc tangent (measured in degrees) of x.
		"""
		x_chk = OpsMath.__reverse_chk(x, OpsMath.__tan_degrees)
		if(as_decimal):
			return x_chk if (x_chk is not None) else OpsMath.degrees(decimal_cast(math.atan(x)), as_decimal=True)
		else:
			return float(x_chk) if (x_chk is not None) else OpsMath.degrees(decimal_cast(math.atan(x)))
		
	@staticmethod	
	@precise_method
	def atan2(y, x, as_decimal=False):
		"""
		Return the arc tangent (measured in degrees) of y/x.
			
		Unlike atan(y/x), the signs of both x and y are considered.	
		"""
		res = OpsMath.degrees(decimal_cast(math.atan2(x, y)), as_decimal=True)
		for k in OpsMath.__tan_degrees.keys():
			try:
				if(math.isclose(k, res)):
					return k if(as_decimal) else float(k)
				elif(math.isclose(-k, res)):
					return -k if(as_decimal) else float(-k)
			except:
				return res if(as_decimal) else float(res)
				
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def cos(x, as_decimal=False):
		"""
		Return the cosine of x (measured in degrees).
		"""
		x_chk = abs(x % 360)
		res = OpsMath.__cos_degrees.get(x_chk, decimal_cast(math.cos(OpsMath.radians(x, as_decimal=True))))
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def sin(x, as_decimal=False):
		"""
		Return the sine of x (measured in degrees).
		"""
		x_chk = abs(x % 360)
		res = OpsMath.__sin_degrees.get(x_chk, decimal_cast(math.sin(OpsMath.radians(x, as_decimal=True))))
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def tan(x, as_decimal=False):
		"""
		Return the tangent of x (measured in degrees).
		"""
		x_chk = abs(x % 360)
		res = OpsMath.__tan_degrees.get(x_chk, decimal_cast(math.tan(OpsMath.radians(x, as_decimal=True))))
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def hypot(x, y, as_decimal=False):
		"""
		Return the Euclidean distance, sqrt(x*x + y*y).
		"""
		dx = decimal_cast(x)
		dy = decimal_cast(y)
		res = Decimal(dx**2 + dy**2).sqrt()
		return res if(as_decimal) else float(res)

	@staticmethod
	@precise_method
	def inv_hypot_y(hypot_l, angle, as_decimal=False):
		"""
		Return Y projection given a hypotenuse and an angle.
		"""
		res = decimal_cast(hypot_l) * OpsMath.sin(angle, as_decimal)
		return res if(as_decimal) else float(res)
		
	@staticmethod
	@precise_method
	def inv_hypot_x(hypot_l, angle, as_decimal=False):
		"""
		Return X projection given a hypotenuse and an angle.
		"""
		res = decimal_cast(hypot_l) * OpsMath.cos(angle, as_decimal)
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def inv_hypot(hypot_l, angle, as_decimal=False):
		"""
		Return X, Y projection given a hypotenuse and an angle.
		"""
		return (OpsMath.inv_hypot_x(hypot_l, angle, as_decimal), OpsMath.inv_hypot_y(hypot_l, angle, as_decimal))
	
	@staticmethod
	@precise_method
	def sqrt(x, as_decimal=False):
		"""
		Return the square root of x.
		"""
		res = decimal_cast(x).sqrt()
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def pow(x, y, as_decimal=False):
		"""
		Return x**y (x to the power of y).
		"""
		res = decimal_cast(x) ** decimal_cast(y)
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def bisector_angle(a1, a2, as_decimal=False):
		"""
		Return angle between two given angles.
		"""
		res = (decimal_cast(abs(a1 % 360)) + decimal_cast(abs(a2 % 360)))/Decimal(2)
		return res if(as_decimal) else float(res)
		
	@staticmethod
	@precise_method
	def degrees(radians, as_decimal=False):
		"""
		Convert angle x from radians to degrees.
		"""
		res = (decimal_cast(radians)/OpsMath.tau) * Decimal(360)
		return res if(as_decimal) else float(res)
	
	@staticmethod
	@precise_method
	def radians(degrees, as_decimal=False):
		"""
		Convert angle x from degrees to radians.
		"""
		res = (decimal_cast(degrees)/Decimal(360)) * OpsMath.tau
		return res if(as_decimal) else float(res)
	
	@staticmethod
	def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
		"""
		Determine whether two floating point numbers are close in value.
		
		  rel_tol
			maximum difference for being considered "close", relative to the
			magnitude of the input values
		  abs_tol
			maximum difference for being considered "close", regardless of the
			magnitude of the input values
		
		Return True if a is close in value to b, and False otherwise.
		
		For the values to be considered close, the difference between them
		must be smaller than at least one of the tolerances.
		
		-inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
		is, NaN is not close to anything, even itself.  inf and -inf are
		only close to themselves.
		"""
		return math.isclose(float(a), float(b))
			
	
	# @staticmethod
	# def intersect_2lines(p1, a1, p2, a2):
		# """
		# Return intersection points (x, y) between 2 lines.
		# If lines are parallel, returns (None, None).
		# Line is defined by a point (x, y) and an angle (degrees).
		# """
		# if(not (isinstance(p1, (list, tuple)) and len(p1)>=2)):
			# raise TypeError("p1 must be an [x, y] point.")
		# elif(not (isinstance(p2, (list, tuple)) and len(p2)>=2)):
			# raise TypeError("p2 must be an [x, y] point.")

		# p1x = p1[0]
		# p1y = p1[1]
		# p2x = p2[0]
		# p2y = p2[1]
		# m1  = OpsMath.tan(a1)
		# m2  = OpsMath.tan(a2)
		
		# # eq1: y = (x - p1x) * m1 + p1y | y = m1*x + (p1y - m1*p1x)
		# # eq2: y = (x - p2x) * m2 + p2y | y = m2*x + (p2y - m2*p2x) 
		
		# # Parallel lines
		# if(m1 == m2):
			# return None
			
		# # Find intersection point
		# # If one of the lines has tangent==inf, this line is vertical.
		# # Else, general rule
		# # x = (d-c)/(a-b)
		# # y = (ad-bc)/(a-b)
		# # ix = ((p2y - m2*p2x) - (p1y - m1*p1x))/(m1-m2)
		# # iy = (m1*(p2y - m2*p2x) - m2*(p1y - m1*p1x))/(m1-m2)
		# if(m1 == OpsMath.inf):
			# ix = p1x
			# iy = (ix - p2x) * m2 + p2y
		# elif(m2 == OpsMath.inf):
			# ix = p2x
			# iy = (ix - p1x) * m1 + p1y
		# else:
			# ix = ((p2y - m2*p2x) - (p1y - m1*p1x))/(m1-m2)
			# iy = (m1*(p2y - m2*p2x) - m2*(p1y - m1*p1x))/(m1-m2)
			
		# return (ix, iy)
	
	# @staticmethod
	# def pivot_2p(p1, a1, p2, a2, r):
		# """
		# Return pivot point [x,y], arc angle, angle offset, tangent point on line 1 [x,y], and tangent point on line 2 [x,y] between two given points, their angles and the radius to the pivot point.
		# If lines are parallel, returns None, None, None, None, None
		# [2 line intersection tangent]
		# """
		
		# ix, iy = OpsMath.intersect_2lines(p1, a1, p2, a2)
		# if(ix is None or iy is None):
			# # Lines are parallel
			# return (None, None, None, None, None)
		
		# p1x = p1[0]
		# p1y = p1[1]
		# p2x = p2[0]
		# p2y = p2[1]

		
		# # Create bisector line
		# # eqbis: y = (x - ix)*m_bis + iy 
		# angle_bis = OpsMath.bisector_angle(a1, a2)
		# angle_bis_rel_a1 = abs(angle_bis - a1)  
		# angle_bis_rel_a2 = abs(angle_bis - a2)  
		# hypot_len = r / OpsMath.sin(angle_bis_rel_a1)
		
		# # Find pivot point
		# d_from_bis_x = hypot_len * OpsMath.cos(angle_bis)
		# d_from_bis_y = hypot_len * OpsMath.sin(angle_bis)
		# pivot_x = ix + d_from_bis_x
		# pivot_y = iy + d_from_bis_y
		
		# # TODO: Utilizar las rectas con angulos normales.
		# # Obtener el angulo del arco a partir de ahi
		# # Find tangent point on line 1
		# d_from_line1 = hypot_len * OpsMath.cos(angle_bis_rel_a1)
		# d_from_line1x = d_from_line1 * OpsMath.cos(a1)
		# d_from_line1y = d_from_line1 * OpsMath.sin(a1)
		# tangent_x1 = ix + d_from_line1x
		# tangent_y1 = iy + d_from_line1y
		
		# # Find tangent point on line 2
		# d_from_line2 = hypot_len * OpsMath.cos(angle_bis_rel_a2)
		# d_from_line2x = d_from_line2 * OpsMath.cos(a2)
		# d_from_line2y = d_from_line2 * OpsMath.sin(a2)
		# tangent_x2 = ix + d_from_line2x
		# tangent_y2 = iy + d_from_line2y
		
		# # Find arc angle
		# hypot_btw_tangents = OpsMath.hypot((tangent_x2-tangent_x1), (tangent_y2-tangent_y1))
		# arc_angle = (90 - OpsMath.acos((hypot_btw_tangents/2)/r))*2
		# arc_angle_offset = OpsMath.atan2((tangent_x1 - pivot_x), (tangent_y1 - pivot_y))
		
		# return ([pivot_x, pivot_y], arc_angle, arc_angle_offset, [tangent_x1,tangent_y1], [tangent_x2,tangent_y2])
		
	acosh = math.acosh

	asinh = math.asinh

	atanh = math.atanh

	ceil = math.ceil

	copysign = math.copysign

	cosh = math.cosh

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
	 
	remainder = math.remainder

	sinh = math.sinh
	
	tanh = math.tanh
	 
	trunc = math.trunc

	# Constants

	e = Decimal("2.718281828459045235360287471352662497757247093699")
	inf = Decimal(math.inf)
	nan = Decimal(math.nan)
	pi = Decimal("3.141592653589793238462643383279502884197169399375")
	tau = Decimal("3.141592653589793238462643383279502884197169399375")*Decimal(2)

 
class __MathObject(metaclass=ABCMeta):
	
	def clone(self):
		return copy.deepcopy(self)
	
	def check_class(self, other):
		if(not isinstance(other, type(self))):
			raise TypeError("Class", type(other),"is not an instance of", type(self))
			
class __MathPolynomial(__MathObject):
	
	@abstractmethod
	def get_points(self, *args, **kwargs):
		pass
	


class MathPoint(__MathObject):
	
	def __init__(self, point=[0,0,0]):
		p = Validator.validate_point(point)
		self.x = p[0]
		self.y = p[1]
		self.z = p[2]
	
	def xy_projection(self):
		"""
		Return projection on XY plane.
		"""
		return MathPoint([self.x, self.y, 0])
	
	def xz_projection(self):
		"""
		Return projection on XZ plane.
		"""
		return MathPoint([self.x, 0, self.z])
	
	def yz_projection(self):
		"""
		Return projection on YZ plane.
		"""
		return MathPoint([0, self.y, self.z])
	
	def __len__(self):
		return 3
		
	def __getitem__(self, key):
		if(isinstance(key, int)):
			if(key == 0):
				return self.x
			elif(key == 1):
				return self.y
			elif(key == 2):
				return self.z
			else:
				raise IndexError("Index " + str(key) + " out of range.")
		else:
			raise TypeError("Unexpected key type: " + str(type(key)))
	
	def __add__(self, other):
		p = Validator.validate_point(other)
		x = decimal_add(self.x, p[0])
		y = decimal_add(self.y, p[1])
		z = decimal_add(self.z, p[2])
		return MathPoint([x,y,z])
	
	def __radd__(self, other):
		return self.__iadd__(other)		
	
	def __iadd__(self, other):
		p = Validator.validate_point(other)
		self.x = decimal_add(self.x, p[0])
		self.y = decimal_add(self.y, p[1])
		self.z = decimal_add(self.z, p[2])
		
	def __sub__(self, other):
		p = Validator.validate_point(other)
		x = decimal_sub(self.x, p[0])
		y = decimal_sub(self.y, p[1])
		z = decimal_sub(self.z, p[2])
		return MathPoint([x,y,z])
	
	def __rsub__(self, other):
		return self.__sub__(other)
	
	def __isub__(self, other):
		p = Validator.validate_point(other)
		self.x = decimal_sub(self.x, p[0])
		self.y = decimal_sub(self.y, p[1])
		self.z = decimal_sub(self.z, p[2])
	
	def __repr__(self):
		return "MathPoint(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
	
	def __str__(self):
		return str([self.x, self.y, self.z])
	
	def __eq__(self, other):
		if(isinstance(other, type(self))):
			return OpsMath.isclose(self.x, other.x) and OpsMath.isclose(self.y, other.y) and OpsMath.isclose(self.z, other.z)
		return False
	
class MathVector(__MathObject):
	
	@staticmethod
	def from_points(point_a, point_b):
		"""
		Return the vector created by 2 points.
		"""
		pa = MathPoint(point_a)
		pb = MathPoint(point_b)
		return MathVector(pb-pa)
		
	@staticmethod
	def from_x_rotation(length, y_rotation=0, z_rotation=0):
		"""
		Return the vector created by rotating (anti-clockwise) a vector [length,0,0]
		"""
		return MathVector([length,0,0]).rotate([0,y_rotation, z_rotation])
		
	@staticmethod
	def from_y_rotation(length, x_rotation=0, z_rotation=0):
		"""
		Return the vector created by rotating (anti-clockwise) a vector [0,length,0]
		"""
		return MathVector([0, length,0]).rotate([x_rotation, 0, z_rotation])
		
	@staticmethod
	def from_z_rotation(length, x_rotation=0, y_rotation=0):
		"""
		Return the vector created by rotating (anti-clockwise) a vector [0,0,length]
		"""
		return MathVector([0,0,length]).rotate([x_rotation,y_rotation, 0])
		
	
	def __init__(self, point = [0,0,0] ):
		if(isinstance(point, MathPoint)):
			self.x = point.x
			self.y = point.y
			self.z = point.z
		else:
			p = Validator.validate_point(point)
			self.x = p[0]
			self.y = p[1]
			self.z = p[2]
	
	@precise_method
	def module(self, as_decimal=False):
		"""
		Return vector module.
		"""
		x = Decimal(self.x) ** 2
		y = Decimal(self.y) ** 2
		z = Decimal(self.z) ** 2
		return (x+y+z).sqrt() if (as_decimal) else float((x+y+z).sqrt())
	
	@precise_method
	def set_module(self, module):
		"""
		Set module of this vector. Return itself.
		"""
		self_l = self.module(as_decimal = True)
		self.x = float((Decimal(self.x)/self_l) * Decimal(module))
		self.y = float((Decimal(self.y)/self_l) * Decimal(module))
		self.z = float((Decimal(self.z)/self_l) * Decimal(module))
		return self
	
	@precise_method
	def unit_vector(self):
		"""
		Return vector with module of 1. If vector's module is 0, return None.
		"""
		v_module = self.module(as_decimal=True)
		if(v_module != 0): 
			x = float(Decimal(self.x)/v_module)
			y = float(Decimal(self.y)/v_module)
			z = float(Decimal(self.z)/v_module)
			return MathVector([x, y, z])
		else:
			return None
	
	def xy_projection(self):
		"""
		Return projection on XY plane.
		"""
		return MathVector([self.x, self.y, 0])
	
	def xz_projection(self):
		"""
		Return projection on XZ plane.
		"""
		return MathVector([self.x, 0, self.z])
	
	def yz_projection(self):
		"""
		Return projection on YZ plane.
		"""
		return MathVector([0, self.y, self.z])
	
	@precise_method
	def dot(self, other, as_decimal=False):
		"""
		Return dot product (Scalar) of this vector and another.
		"""
		p = Validator.validate_point(other)
		v = decimal_cast(self.x) * decimal_cast(p[0])
		v += decimal_cast(self.y) * decimal_cast(p[1])
		v += decimal_cast(self.z) * decimal_cast(p[2])
		v = decimal_aproximate(v)
		
		return v if(as_decimal) else float(v)
	
	@precise_method
	def vectorial(self, other):
		"""
		Return vectorial product of this vector and another.
		"""
		p = Validator.validate_point(other)
		
		i = decimal_aproximate(decimal_cast(self.y)*decimal_cast(p[2]) - decimal_cast(self.z)*decimal_cast(p[1]))
		j = decimal_aproximate(decimal_cast(self.x)*decimal_cast(p[2]) - decimal_cast(self.z)*decimal_cast(p[0]))
		k = decimal_aproximate(decimal_cast(self.x)*decimal_cast(p[1]) - decimal_cast(self.y)*decimal_cast(p[0]))
		
		return MathVector([i,j,k])
	
	def xy_quadrant(self):
		"""
		Return quadrant position of this vector on plane XY (1-4)
		"""
		if(self.x >= 0 and self.y >=0):
			return 1
		elif(self.x <= 0 and self.y >=0):
			return 2
		elif(self.x <= 0 and self.y <=0):
			return 3
		elif(self.x >= 0 and self.y <=0):
			return 4
		else:
			return 0
			
	def xz_quadrant(self):
		"""
		Return quadrant position of this vector on plane XZ (1-4)
		"""
		if(self.x >= 0 and self.z >=0):
			return 1
		elif(self.x <= 0 and self.z >=0):
			return 2
		elif(self.x <= 0 and self.z <=0):
			return 3
		elif(self.x >= 0 and self.z <=0):
			return 4
		else:
			return 0
			
	def yz_quadrant(self):
		"""
		Return quadrant position of this vector on plane YZ (1-4)
		"""
		if(self.y >= 0 and self.z >=0):
			return 1
		elif(self.y <= 0 and self.z >=0):
			return 2
		elif(self.y <= 0 and self.z <=0):
			return 3
		elif(self.y >= 0 and self.z <=0):
			return 4
		else:
			return 0
	
	@precise_method
	def angle(self, other=[1,0,0], as_decimal=False):
		"""
		Return the angle between this vector and another.
		Other vector can be blank, this will return the angle between this vector and [1,0,0]
		If one of the vectors is NULL, return None.
		"""
		dot_p = self.dot(other, as_decimal = True)
		self_l = self.module(as_decimal = True)
		other_l = MathVector(other).module(as_decimal = True)
		if(self_l != 0 and other_l != 0):
			return OpsMath.acos(dot_p/(self_l*other_l), as_decimal=as_decimal)
		else:
			return None
	
	@precise_method
	def xy_angle(self, other=[1,0,0], as_decimal=False):
		"""
		Return the angle between this vector and another on XY plane.
		Other vector can be blank, this will return the angle between this vector and [1,0,0]
		"""
		v1 = self.xy_projection()
		v2 = MathVector(other).xy_projection()
		a1 = OpsMath.atan2(v1.y, v1.x, as_decimal=True)
		a2 = OpsMath.atan2(v2.y, v2.x, as_decimal=True)
		res = (a1-a2)%360
		return res if(as_decimal) else float(res)
	
	@precise_method
	def xz_angle(self, other=[1,0,0], as_decimal=False):
		"""
		Return the angle between this vector and another on XZ plane.
		Other vector can be blank, this will return the angle between this vector and [1,0,0]
		"""
		v1 = self.xz_projection()
		v2 = MathVector(other).xz_projection()
		a1 = OpsMath.atan2(v1.z, v1.x, as_decimal=True)
		a2 = OpsMath.atan2(v2.z, v2.x, as_decimal=True)
		res = (a1-a2)%360
		return res if(as_decimal) else float(res)
	
	@precise_method
	def yz_angle(self, other=[0,1,0], as_decimal=False):
		"""
		Return the angle between this vector and another on YZ plane.
		Other vector can be blank, this will return the angle between this vector and [0,1,0]
		"""
		v1 = self.yz_projection()
		v2 = MathVector(other).yz_projection()
		a1 = OpsMath.atan2(v1.z, v1.y, as_decimal=True)
		a2 = OpsMath.atan2(v2.z, v2.y, as_decimal=True)
		res = (a1-a2)%360
		return res if(as_decimal) else float(res)
	
	def xy_normal(self, anticlock_wise=True):
		"""
		Return normal vector on XY plane.
		Set anticlock_wise to False to invert direction.
		"""
		quadrant = self.xy_quadrant()
		new_quadrant = (quadrant + 1)%4 +1 if(anticlock_wise) else  (quadrant - 1)%4 +1
		if(new_quadrant == 1):
			x = abs(self.x)
			y = abs(self.y)
		elif(new_quadrant == 2):
			x = -abs(self.x)
			y = abs(self.y)
		elif(new_quadrant == 3):
			x = -abs(self.x)
			y = -abs(self.y)
		elif(new_quadrant == 4):
			x = abs(self.x)
			y = -abs(self.y)
		
		return MathVector([x, y, self.z])
		
	def xz_normal(self, anticlock_wise=True):
		"""
		Return normal vector on XZ plane.
		Set anticlock_wise to False to invert direction.
		"""
		quadrant = self.xz_quadrant()
		new_quadrant = (quadrant + 1)%4 +1 if(anticlock_wise) else  (quadrant - 1)%4 +1
		if(new_quadrant == 1):
			x = abs(self.x)
			z = abs(self.z)
		elif(new_quadrant == 2):
			x = -abs(self.x)
			z = abs(self.z)
		elif(new_quadrant == 3):
			x = -abs(self.x)
			z = -abs(self.z)
		elif(new_quadrant == 4):
			x = abs(self.x)
			z = -abs(self.z)
		
		return MathVector([x, self.y, z])
	
	def yz_normal(self, anticlock_wise=True):
		"""
		Return normal vector on YZ plane.
		Set anticlock_wise to False to invert direction.
		"""
		quadrant = self.yz_quadrant()
		new_quadrant = (quadrant + 1)%4 +1 if(anticlock_wise) else  (quadrant - 1)%4 +1
		if(new_quadrant == 1):
			y = abs(self.y)
			z = abs(self.z)
		elif(new_quadrant == 2):
			y = -abs(self.y)
			z = abs(self.z)
		elif(new_quadrant == 3):
			y = -abs(self.y)
			z = -abs(self.z)
		elif(new_quadrant == 4):
			y = abs(self.y)
			z = -abs(self.z)
		
		return MathVector([self.x, y, z])
	
	def are_orthogonal(self, vector):
		"""
		Return True if this vector and another are orthogonal.
		"""
		self.check_class(vector)
		self_u = self.unit_vector()
		vector_u = vector.unit_vector()
		if(None not in [self_u, vector_u]):
			return self_u.dot(vector_u, as_decimal = True) == 0
		else:
			return False
	
	def rotate(self, rotation=[0,0,0]):
		r = Validator.validate_rotation(rotation)
		return self.x_rotate(r[0]).y_rotate(r[1]).z_rotate(r[2])
	
	@precise_method
	def x_rotate(self, angle=0):
		rot = Validator.validate_angle(angle)
		if(rot!=0):
			y = (decimal_cast(self.y) * OpsMath.cos(rot, as_decimal=True)) - (decimal_cast(self.z) * OpsMath.sin(rot, as_decimal=True))
			z = (decimal_cast(self.y) * OpsMath.sin(rot, as_decimal=True)) + (decimal_cast(self.z) * OpsMath.cos(rot, as_decimal=True))
			
			self.y = float(y)
			self.z = float(z)
			
		return self
	
	@precise_method
	def y_rotate(self, angle=0):
		rot = Validator.validate_angle(angle)
		if(rot!=0):
			x = (decimal_cast(self.x) * OpsMath.cos(rot, as_decimal=True)) + (decimal_cast(self.z) * OpsMath.sin(rot, as_decimal=True))
			z = (Decimal(-1) * decimal_cast(self.x) * OpsMath.sin(rot, as_decimal=True)) + (decimal_cast(self.z) * OpsMath.cos(rot, as_decimal=True))
			
			self.y = float(y)
			self.z = float(z)
			
		return self
	
	@precise_method
	def z_rotate(self, angle=0):
		rot = Validator.validate_angle(angle)
		if(rot!=0):
			x = (decimal_cast(self.x) * OpsMath.cos(rot, as_decimal=True)) - (decimal_cast(self.y) * OpsMath.sin(rot, as_decimal=True))
			y = (decimal_cast(self.x) * OpsMath.sin(rot, as_decimal=True)) + (decimal_cast(self.y) * OpsMath.cos(rot, as_decimal=True))
			
			self.y = float(y)
			self.z = float(z)
			
		return self
		
	def is_null(self):
		"""
		Return True if NULL vector.
		"""
		return self.module(as_decimal=True) == 0
	
	def __len__(self):
		return 3
	
	def __getitem__(self, key):
		if(isinstance(key, int)):
			if(key == 0):
				return self.x
			elif(key == 1):
				return self.y
			elif(key == 2):
				return self.z
			else:
				raise IndexError("Index " + str(key) + " out of range.")
		else:
			raise TypeError("Unexpected key type: " + str(type(key)))
	
	def __add__(self, other):
		p = Validator.validate_point(other)
		x = decimal_add(self.x, p[0])
		y = decimal_add(self.y, p[1])
		z = decimal_add(self.z, p[2])
		return MathVector([x,y,z])
	
	def __radd__(self, other):
		return self.__iadd__(other)	
	
	def __iadd__(self, other):
		p = Validator.validate_point(other)
		self.x = decimal_add(self.x, p[0])
		self.y = decimal_add(self.y, p[1])
		self.z = decimal_add(self.z, p[2])
	
	def __sub__(self, other):
		p = Validator.validate_point(other)
		x = decimal_sub(self.x, p[0])
		y = decimal_sub(self.y, p[1])
		z = decimal_sub(self.z, p[2])
		return MathVector([x,y,z])
	
	def __isub__(self, other):
		p = Validator.validate_point(other)
		self.x = decimal_sub(self.x, p[0])
		self.y = decimal_sub(self.y, p[1])
		self.z = decimal_sub(self.z, p[2])
	
	def __mul__(self, other):
		if(isinstance(other, (int, float, Decimal))):
			x = decimal_mul(self.x, other)
			y = decimal_mul(self.y, other)
			z = decimal_mul(self.z, other)
			return MathVector([x, y, z])
		else:
			return self.dot(other)
	
	def __rmul__(self, other):
		return self.__mul__(other)
	
	def __imul__(self, other):
		if(isinstance(other, (int, float, Decimal))):
			self.x = decimal_mul(self.x, other)
			self.y = decimal_mul(self.y, other) 
			self.z = decimal_mul(self.z, other)
		else:
			raise TypeError("Cannot assign the result of dot product to a MathVector instance.")
			
	def __truediv__(self, other):
		x = decimal_div(self.x, other)
		y = decimal_div(self.y, other)
		z = decimal_div(self.z, other)
		return MathVector([x, y, z])
		
	def __itruediv__(self, other):
		self.x = decimal_div(self.x, other)
		self.y = decimal_div(self.y, other) 
		self.z = decimal_div(self.z, other) 
	
	def __neg__(self):
		return MathVector([-self.x, -self.y, -self.z])
	
	def __repr__(self):
		return "MathVector(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
	
	def __str__(self):
		return str([self.x, self.y, self.z])
		
	def __eq__(self, other):
		if(isinstance(other, type(self))):
			return OpsMath.isclose(self.x, other.x) and OpsMath.isclose(self.y, other.y) and OpsMath.isclose(self.z, other.z)
		return False
		
		
class MathLine(__MathPolynomial):
	# From equation
	
	@staticmethod
	def from_x_rotation(origin=[0,0,0], y_rotation=0, z_rotation=0):
		"""
		Return the line created by rotating (anti-clockwise) a unit vector [1,0,0] centered at origin
		"""
		vector = MathVector([1,0,0]).rotate([0,y_rotation, z_rotation])
		return MathLine(origin = origin, vector=vector)
		
	@staticmethod
	def from_y_rotation(origin=[0,0,0], x_rotation=0, z_rotation=0):
		"""
		Return the line created by rotating (anti-clockwise) a unit vector [0,1,0] centered at origin
		"""
		vector = MathVector([0,1,0]).rotate([x_rotation, 0, z_rotation])
		return MathLine(origin = origin, vector=vector)
		
	@staticmethod
	def from_z_rotation(origin=[0,0,0], x_rotation=0, y_rotation=0):
		"""
		Return the line created by rotating (anti-clockwise) a unit vector [0,0,1] centered at origin
		"""
		vector = MathVector([0,0,1]).rotate([x_rotation,y_rotation, 0])
		return MathLine(origin = origin, vector=vector)
	
	def __init__(self, origin=[0,0,0], vector=[1,0,0]):
		self._origin = MathPoint(origin)
		self._vector = Validator.validate_unit_vector(MathVector(vector))
		if(self._vector is None):
			self._vector = MathVector([0,0,0])
	
	@property
	def vector(self):
		return self._vector
		
	@vector.setter
	def vector(self, value):
		self._vector = Validator.validate_unit_vector(MathVector(value).unit_vector())
		if(self._vector is None):
			self._vector = MathVector([0,0,0])
			
	@property
	def origin(self):
		return self._origin
		
	@origin.setter
	def origin(self, value):
		self._origin = MathPoint(value)
	
	@precise_method
	def at_x(self, x_coord):
		"""
		Return point at given X coordinate.
		If no point is found, return None
		"""
		
		if(self._origin.x == x_coord):
			return self._origin.clone()
	
		if(self._vector.x == 0):
			return None
		
		coord_diff = decimal_cast(x_coord - self._origin.x) / decimal_cast(self._vector.x)
		return self._origin + (coord_diff * self._vector)
	
	@precise_method
	def at_y(self, y_coord):
		"""
		Return point at given Y coordinate.
		If no point is found, return None
		"""
		
		if(self._origin.y == y_coord):
			return self._origin.clone()
		
		if(self._vector.y == 0):
			return None
			
		coord_diff = decimal_cast(y_coord - self._origin.y) / decimal_cast(self._vector.y)
		return self._origin + (coord_diff * self._vector)
	
	@precise_method	
	def at_z(self, z_coord):
		"""
		Return point at given Z coordinate.
		If no point is found, return None
		"""
		
		if(self._origin.z == z_coord):
			return self._origin.clone()
		
		if(self._vector.z == 0):
			return None
			
		coord_diff = decimal_cast(z_coord - self._origin.z) / decimal_cast(self._vector.z)
		return self._origin + (coord_diff * self._vector)
	
	def has_point(self, point):
		"""
		Return True if point is on this line.
		"""
		p = point if(isinstance(point, MathPoint)) else MathPoint(Validator.validate_point(point))
		x = self.at_x(p.x)
		y = self.at_y(p.y)
		z = self.at_z(p.z)
		
		if(x is not None and x == p):
			return True
		elif(y is not None and y == p):
			return True
		elif(z is not None and z == p):
			return True
		else:
			return False
	
	def xy_projection(self):
		"""
		Return projection on XY plane.
		"""
		return MathLine(origin=[self._origin.x, self.origin.y, 0], vector = self._vector.xy_projection())
	
	def xz_projection(self):
		"""
		Return projection on XZ plane.
		"""
		return MathLine(origin=[self._origin.x, 0, self._origin.z], vector = self._vector.xz_projection())
	
	def yz_projection(self):
		"""
		Return projection on YZ plane.
		"""
		return MathLine(origin=[0, self._origin.y, self._origin.z], vector = self._vector.yz_projection())
	
	def xy_slope(self, as_decimal=False):
		"""
		Return the slope of this line in XY plane.
		"""
		return OpsMath.tan(self._vector.xy_angle(as_decimal=True), as_decimal=as_decimal)
		
	def xz_slope(self, as_decimal=False):
		"""
		Return the slope of this line in XZ plane.
		"""
		return OpsMath.tan(self._vector.xz_angle(as_decimal=True), as_decimal=as_decimal)
		
	def yz_slope(self, as_decimal=False):
		"""
		Return the slope of this line in YZ plane.
		"""
		return OpsMath.tan(self._vector.yz_angle(as_decimal=True), as_decimal=as_decimal)
	
	def are_parallel(self, line):
		"""
		Return True if this line and another are parallel.
		"""
		self.check_class(line)
		return self._vector == line.vector or self._vector == -line.vector
		
	def are_coincident(self, line):
		"""
		Return True if this line and another are coincident.
		"""
		#self.check_class(line) # Checked in are_parallel
		if(self.are_parallel(line)):
			return self._origin == line.origin
		return False
	
	def are_secant(self, line):
		"""
		Return True if this line and another are secant.
		"""
		self.check_class(line)
		return self.intersection_point(line) is not None
	
	def are_skewed(self, line):
		"""
		Return True if this line and another skew each other.
		"""
		return not self.are_parallel(line) and not self.are_secant(line)
	
	def are_oblique(self, line):
		"""
		Return True if this line and another are oblique.
		"""
		#self.check_class(line) # Checked in are_parallel
		if(self.are_parallel(line)):
			return False
	
	def are_orthogonal(self, line):
		"""
		Return True if this line and another are orthogonal.
		"""
		#self.check_class(line) # Checked in are_oblique
		if(self.are_oblique(line)):
			return self._vector.are_orthogonal(line.vector)
			
		return False
		
	def are_perpendicular(self, line):
		"""
		Return True if this line and another are perpendicular.
		"""
		#self.check_class(line) # Checked in are_orthogonal
		return self.are_orthogonal(line) and self.are_secant(line)
	
	@precise_method
	def skew_point(self, other):
		"""
		Return skew point (shortest distance) between this line and another line or point.
		Skew point is always contained on this line.
		
		Point -> Skew Point
		Coincident -> Origin Point
		Parallel -> Origin Point
		Secant -> Intersection Point
		Skewed -> Skew Point
		"""
		if(isinstance(other, MathPoint)):
			p_vector = MathVector.from_points(self._origin, other)
			angle = decimal_aproximate(p_vector.angle(self._vector, as_decimal=True))
			hypot = decimal_aproximate(p_vector.module(as_decimal=True))
			factor = decimal_aproximate(hypot * OpsMath.cos(angle, as_decimal=True))
			return self._origin + (factor * self._vector)
			
		elif(isinstance(other, MathLine)):
			# https://en.wikipedia.org/wiki/Skew_lines
			rs_vector = MathVector.from_points(self._origin, other.origin)
			n2 = other.vector.vectorial(self._vector.vectorial(other.vector))
			num_r = rs_vector.dot(n2, as_decimal=True)
			den_r = self._vector.dot(n2, as_decimal=True)
			return self._origin + (num_r/den_r)*self._vector
		
		else:
			raise TypeError(str(type(other)) + " is not a MathPoint or a MathLine object") 
	
	@precise_method
	def intersection_point(self, line):
		"""
		Return intersection point between this line and another if exists, else return None.
		"""
		self.check_class(line)
		breakpoint()
		# If other line also contains skew_point, it has an intersection
		skew_point = self.skew_point(line)
		return skew_point if(line.has_point(skew_point)) else None
		
		
		# #Check if origin is contained in lines
		# if(self.has_point(line.origin)):
			# return MathPoint(line.origin)
		# elif(line.has_point(self.origin)):
			# return MathPoint(self.origin)
		# #Check if parallel (No intersection, if equal lines, origin was returned)
		# elif(self.are_parallel(line)):
			# return None
		
		
		# #Simplify intersecting problem R3 -> R2, 3 times, one for every projection
		
		# ### Projection on XY, find intersection
		# xy_proj_self = self.xy_projection()
		# a = xy_proj_self.xy_slope()
		# c = xy_proj_self.at_x(0)
		# c = c.y if(c is not None) else None
		# xy_proj_line = line.xy_projection()
		# b = xy_proj_line.xy_slope()
		# d = xy_proj_line.at_x(0)
		# d = d.y if(d is not None) else None
		
		
		# if(None not in [c,d]):
			# try:
				# x_value = (d-c)/(a-b)
				# p = line.at_x(x_value)
				
				# if(p is not None and self.has_point(p)):
					# return p
			# except:
				# #Ignore division by 0 exceptions, etc...
				# pass
		
		# # Slope of XY projection of self is inf (vertical line)
		# if(OpsMath.isinf(a)):
			# x_coord_proj_self = xy_proj_self.origin.x
			# p = line.at_x(x_coord_proj_self)
			# if(p is not None and self.has_point(p)):
					# return p
		# # Slope of XY projection of line is inf (vertical line)
		# if(OpsMath.isinf(b)):
			# x_coord_proj_line = xy_proj_line.origin.x
			# p = self.at_x(x_coord_proj_line)
			# if(p is not None and line.has_point(p)):
					# return p
			
			
		# ### Projection on XZ, find intersection
		# xz_proj_self = self.xz_projection()
		# a = xz_proj_self.xz_slope()
		# c = xz_proj_self.at_x(0)
		# c = c.z if(c is not None) else None
		
		# xz_proj_line = line.xz_projection()
		# b = xz_proj_line.xz_slope()
		# d = xz_proj_line.at_x(0)
		# d = d.z if(d is not None) else None
		
		# if(None not in [c,d]):
			# try:
				# x_value = (d-c)/(a-b)
				# p = line.at_x(x_value)
				
				# if(p is not None and self.has_point(p)):
					# return p
			# except:
				# #Ignore division by 0 exceptions, etc...
				# pass
			
		# # Slope of XZ projection of self is inf (vertical line)
		# if(OpsMath.isinf(a)):
			# x_coord_proj_self = xz_proj_self.origin.x
			# p = line.at_x(x_coord_proj_self)
			# if(p is not None and self.has_point(p)):
					# return p
		# # Slope of XZ projection of line is inf (vertical line)
		# if(OpsMath.isinf(b)):
			# x_coord_proj_line = xz_proj_line.origin.x
			# p = self.at_x(x_coord_proj_line)
			# if(p is not None and line.has_point(p)):
					# return p
			
		# # Projection on YZ, find intersection
		# yz_proj_self = self.yz_projection()
		# a = yz_proj_self.yz_slope()
		# c = yz_proj_self.at_y(0)
		# c = c.z if(c is not None) else None
		
		# yz_proj_line = line.yz_projection()
		# b = yz_proj_line.yz_slope()
		# d = xz_proj_line.at_y(0)
		# d = d.z if(d is not None) else None
		
		# if(None not in [c,d]):
			# try:
				# y_value = (d-c)/(a-b)
				# p = line.at_y(y_value)
				
				# if(p is not None and self.has_point(p)):
					# return p
			# except:
				# #Ignore division by 0 exceptions, etc...
				# pass
				
		# # Slope of YZ projection of self is inf (vertical line)
		# if(OpsMath.isinf(a)):
			# y_coord_proj_self = yz_proj_self.origin.y
			# p = line.at_y(y_coord_proj_self)
			# if(p is not None and self.has_point(p)):
					# return p
		# # Slope of YZ projection of line is inf (vertical line)
		# if(OpsMath.isinf(b)):
			# y_coord_proj_line = yz_proj_line.origin.y
			# p = self.at_y(y_coord_proj_line)
			# if(p is not None and line.has_point(p)):
					# return p	
		
		# # No intersection found, return None
		return None
	
	@precise_method
	def distance(self, other):
		"""
		Return distance between this line and another line or point.
		"""
		
		if(isinstance(other, MathPoint)):
			p_vector = MathVector.from_points(self._origin, other)
			n = p_vector.vectorial(self._vector)
			mod_n = n.module()
			self_vmod = self._vector.module() # Should be 1
			
			return mod_n/self_vmod if(self_vmod!=0) else p_vector.module()
			
			
		elif(isinstance(other, MathLine)):
			if(self.are_coincident(other) or self.are_secant(other)):
				return 0.0
			
			elif(self.are_parallel(other)):
				return self.distance(other.origin)	
			
			else:
				n = self._vector.vectorial(other.vector)
				p_vector = MathVector.from_points(self._origin, other.origin)
				a_dot = n.dot(p_vector)
				b_dot = n.dot(n)
				if(b_dot !=0):
					return ((a_dot/b_dot) * n).module()
				else:
					return other.distance(self._origin)
		else:
			raise TypeError("Class", type(other),"is not an instance of MathLine or MathPoint") 
	
	@precise_method
	def get_points(self, from_x=None, to_x=None, from_y=None, to_y=None, from_z=None, to_z=None, num=2):
		"""
		Return a list of points defined by constraints:
		Origin -> from_x, from_y, from_z
		End -> to_x, to_y, to_z
		Nª Points -> num (min 2)
		If is not posible to return any points, return empty list.
		"""
		
		slices = int(num)-1 if(num and isinstance(num, (int, float, Decimal)) and num>=2) else 1
		
		origin = None
		if(from_x is not None):
			origin = self.at_x(from_x)
			
		elif(from_y is not None):
			origin = self.at_y(from_y)
		
		elif(from_z is not None):
			origin = self.at_z(from_z)
		
		if(origin is None):
			return []
		
		end = None
		if(to_x is not None):
			end = self.at_x(to_x)
			
		elif(to_y is not None):
			end = self.at_y(to_y)
		
		elif(to_z is not None):
			end = self.at_z(to_z)
		
		if(end is None):
			return []
			
		# Get sweeping coordinate
		sweep_x = False
		sweep_y = False
		sweep_z = False
		initial_value = None
		end_value = None
		
		if(origin.x != end.x):
			sweep_x = True
			initial_value = origin.x
			end_value = end.x
			
		elif(origin.y != end.y):
			sweep_y = True
			initial_value = origin.x
			end_value = end.x
			
		elif(origin.z != end.z):
			sweep_z = True
			initial_value = origin.x
			end_value = end.x
			
		else:
			return []
		
		
		#Sweep coordinate values
		step = (end_value - initial_value)/slices
		points = []
		
		for i in range(0, slices+1):
			next_coord = (i*step) + initial_value
			p = self.at_x(next_coord) if(sweep_x) else (self.at_y(next_coord) if(sweep_y) else (self.at_z if(sweep_z) else None))
			if(p is not None):
				points.append(p)
			else:
				return []
		
		return points
		
	
	def __str__(self):
		return "MathLine(" + str(self._origin) + " -> " + str(self._vector) + ")"
 
 
 
 
 
 
 
 
 
 
 
 
 
