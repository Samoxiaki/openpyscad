from .base import BaseObject
from .boolean import Union
from .shapes_3d import *
from .transformations import *
from decimal import Decimal
import os

__all__ = ['Project', 'Validator', 'Workspace', 'PolyhedronBuilder']

class Project:
	
	def __init__(self, name="project", path="", params={}, workspaces={}, file_extension="scad", ws_separator="_"):
		self.name = name
		self.path = path 
		self.params = params
		self.workspaces = workspaces
		self.file_extension = file_extension.lstrip(".")
		self.ws_separator = ws_separator
	
	def __setattr__(self, name, value):
		if(name == "file_extension"):
			self.__dict__[name] = str(value).lstrip(".")
		else:			
			self.__dict__[name] = value
	
	def add_workspace(self, name, workspace):
		if(isinstance(workspace, Workspace)):
			self.workspaces[name] = workspace
		else:
			raise TypeError(str(workspace) + " is not a Workspace object")
	
	def remove_workspace(self, name):
		if(name in self.workspaces.keys()):
			return self.workspaces.pop(name)
		else:
			return None
	
	def get_workspace(self, name):
		if(name in self.workspaces.keys()):
			return self.workspaces[name]
		else:
			raise TypeError("'" + str(name) + "' Workspace not found")

	def add_param(self, name, param):
		self.params[name] = param
		
	def remove_param(self, name):
		if(name in self.params.keys()):
			return self.params.pop(name)
		else:
			return None
	
	def get_param(self, name):
		if(name in self.params.keys()):
			return self.params[name]
		else:
			raise TypeError("'" + str(name) + "' param not found")	
	
	def dumps(self, indent_level=0, fp=None, divide=False):
		if(divide):
			dump_dict = {}
			for k, v in self.workspaces.items():
				dump_dict[k] = v.dumps(indent_level, fp)
				
			return dump_dict
		else:
			project_union = Union()
			for k, v in self.workspaces.items():
				project_union.append(v)
				
			return project_union.dumps(indent_level, fp)
	
	def __get_file_name(self, ws_name=None):
		write_path = self.path if(self.path) else os.path.realpath(os.path.curdir)
		if(ws_name is not None):
			os.path.join(write_path, ".".join([self.ws_separator.join([self.name, ws_name]), self.file_extension]))
		else:
			return os.path.join(write_path, ".".join([self.name, self.file_extension]))
	
	def write(self, with_print=False, divide=False):
		# Create paths if not exist.
		write_path = self.path if(self.path) else os.path.realpath(os.path.curdir)
		os.makedirs(write_path, exist_ok=True)
			
		
		# Save full project
		project_file_name = self.__get_file_name()
		project_union = Union()
		
		for k, v in self.workspaces.items():
			project_union.append(v)
			
		if(with_print):
			print("PROJECT  : " + self.name)
			print("FILE_NAME: " + project_file_name)
			print("----------")
		project_union.write(project_file_name, with_print=with_print)
		
		# Save workspaces on different files
		if(divide):
			for k, v in self.workspaces.items():
				division_file_name = self.__get_file_name(k)
				if(with_print):
					print()
					print("WORKSPACE: " + k)
					print("FILE_NAME: " + division_file_name)
					print("----------")
				v.write(division_file_name, with_print=with_print)
			
		

class Validator:
	
	__all_colors = Colors.all_colors()
	
	@staticmethod
	def validate_point(point, dim=3):
		"""
		Return parsed point if valid, else raise an exception.
		"""
		from .ops_math import MathPoint
		from .ops_math import MathVector
		
		if(point is not None and isinstance(point, (MathPoint, MathVector))):
			return point
			
		elif(point is not None and isinstance(point, (list, tuple))):
			ret_point = [0.0] * dim
			for i in range(0, min(dim, len(point))):
				try:
					ret_point[i] = float(point[i])
				except:
					raise TypeError("Cannot parse value to a number: " + point[i])
			return ret_point
			
		else:
			raise TypeError("Invalid point: " + point)
			
	@staticmethod
	def validate_unit_vector(vector, dim=3):
		"""
		Return unit vector of given vector if valid, else raise an exception.
		"""
		from .ops_math import MathVector
		
		if(vector is not None and isinstance(vector, MathVector)):
			return vector.unit_vector()

		elif(vector is not None and isinstance(vector, (list, tuple))):
			ret_vector = [0.0] * dim
			acummulator = 0.0
			for i in range(0, min(dim, len(vector))):
				try:
					ret_vector[i] = float(vector[i])
					acummulator += 	ret_vector[i]**2
				except:
					raise TypeError("Cannot parse value to a number: " + vector[i])
			
			acummulator = OpsMath.sqrt(acummulator)
			if(acummulator == 0):
				raise TypeError("Invalid vector [NULL]: " + vector)
			
			for i in range(0, len(ret_vector)):
				ret_vector[i] /= acummulator
				
			return ret_vector
		else:
			raise TypeError("Invalid vector: " + vector)
	
	@staticmethod
	def validate_rotation(rotation, dim=3):
		"""
		Return parsed rotation if valid, else raise an exception.
		"""
		if(rotation is not None and isinstance(rotation, (list, tuple))):
			ret_rotation = [0.0] * dim
			for i in range(0, min(dim, len(rotation))):
				try:
					ret_rotation[i] = float(rotation[i]) % 360
				except:
					raise TypeError("Cannot parse value to a number: " + rotation[i])
			return ret_rotation
		else:
			raise TypeError("Invalid rotation: " + rotation)
			
	@staticmethod
	def validate_angle(angle, from_radians=False):
		"""
		Return parsed angle if valid, else raise an exception.
		Set from_radians=True to convert from radians to degrees.
		"""
		from .ops_math import OpsMath
		
		if(angle is not None and isinstance(angle, (int, float, Decimal))):
			ret_angle = 0.0
			try:
				ret_angle= float(angle if(not from_radians) else OpsMath.degrees(angle)) % 360
			except:
				raise TypeError("Cannot parse value to a number: " + angle)
			return ret_angle
		else:
			raise TypeError("Invalid angle: " + angle)
	
	@staticmethod
	def validate_color(color):
		"""
		Return parsed color if valid, else raise an exception.
		"""
		
		# Check if None (doesn't raise exception)
		if(color is None):
			return None
		
		# Check by name
		lowercase_color = color.lower()
		for k, v in Validator.__all_colors.items():
			if(lowercase_color in [v.lower(), k.lower()]):
				return v
		
		# Check if RGBA
		offset_1 = color.startswith("#")
		min_l = 7 if(offset_1) else 6
		max_l = 9 if(offset_1) else 8
			
		if(len(color) in [min_l, max_l]):
			red = color[1:3] if(offset_1) else color[0:2]
			green = color[3:5] if(offset_1) else color[2:4]
			blue = color[5:7] if(offset_1) else color[4:6]
			alpha = "FF"
			if(len(color) == max_l):
				alpha = color[7:9] if(offset_1) else color[6:8]
			
			try:
				int(red, base=16)
				int(blue, base=16)
				int(green, base=16)
				int(alpha, base=16)
				return "#" + red + green + blue + alpha
			except:
				raise TypeError("Invalid color: " + color)
		else:
			raise TypeError("Invalid color: " + color)
		

class Workspace(BaseObject):
	
	def __init__(self, position=[0,0,0], rotation=[0,0,0], color=None, visible=True):
		"""
		Create a new Workspace object. 
		Param processing order:
		1. Visibility
		2. Color
		3. Rotation
		4. Position
		"""
		super().__init__()
		self._position = Validator.validate_point(position)
		self._rotation = Validator.validate_rotation(rotation)
		self._color = Validator.validate_color(color)
		self._visible = True if(visible) else False
			
	
	def dumps(self, indent_level=0, fp=None):
		dumped_obj = Union()
		for child in self.children:
			dumped_obj.append(child)
		
		if(not self._visible):
			dumped_obj = dumped_obj.disable()
			
		if(self._color):
			dumped_obj = dumped_obj.color(self._color)
			
		dumped_obj = dumped_obj.rotate(self._rotation).translate(self._position)
		
		return dumped_obj.dumps(indent_level=indent_level, fp=fp)

	def translate(self, translation):
		parsed_translation = Validator.validate_point(translation)
		for i in range(0,len(parsed_translation)):
			self._position[i]+= parsed_translation[i]
			
		return self.clone()
	
	def move(self, new_position):
		self._position = Validator.validate_point(new_position)
		return self.clone()
		
	
	def rotate(self, rotation):
		parsed_rotation = Validator.validate_rotation(rotation)
		for i in range(0,len(parsed_rotation)):
			self._rotation[i]+= parsed_rotation[i]
			
		return self.clone()
	
	def set_rotation(self, new_rotation):
		self._rotation = Validator.validate_rotation(new_rotation)
		return self.clone()
		
	def color(self, color):
		self._color = Validator.validate_color(color)
		return self.clone()
		
	def disable(self):
		self._visible = False
		
	def enable(self):
		self_visible = True
	
class PolyhedronBuilder:
	
	def __init__(self):
		self.points = []
		self.faces = []
		
	def __get_point_index(self, point):
		
		for i in range(0, len(self.points)):
			if(point == self.points[i]):
				return i
		index = len(self.points)
		self.points.append(point)
		return index
	
	def add_face(self, points):
		face = []
		for p in points:
			validated_point = Validator.validate_point(p)
			if(isinstance(validated_point, (list, tuple))):
				face.append(self.__get_point_index(validated_point))
			else:
				face.append(self.__get_point_index(list(validated_point)))
		
		self.faces.append(face)
		return face
	
	def build(self):
		return Polyhedron(points = self.points, faces = self.faces)
			
