# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .base import BaseObject
from .boolean import Union
from .shapes_3d import *
from .transformations import *

__all__ = ['Workspace', 'PolyhedronBuilder']


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
		self._position = Workspace.__validation_check(position, "position")
		self._rotation = Workspace.__validation_check(rotation, "rotation")
		self._color = Workspace.__validation_check(color, "color")
		self._visible = Workspace.__validation_check(visible, "boolean")
		
	
	@staticmethod
	def __validation_check(obj, obj_type="position"):
		"""
		Return parsed values from object if valid.
		If not valid, raise an exception.
		"""
		if(obj_type == "position"):
			if(obj and isinstance(obj, (list, tuple))):
				position = [0,0,0]
				for i in range(0, min(3, len(obj))):
					try:
						position[i] = float(obj[i])
					except:
						raise TypeError("Cannot parse value to a number: " + obj[i])
				return position
			else:
				raise TypeError("Invalid position: " + obj)
		
		elif(obj_type == "rotation"):
			if(obj and isinstance(obj, (list, tuple))):
				rotation = [0,0,0]
				for i in range(0, min(3, len(obj))):
					try:
						rotation[i] = float(obj[i]) % 360
					except:
						raise TypeError("Cannot parse value to a number: " + obj[i])
				return rotation
			else:
				raise TypeError("Invalid rotation: " + obj)
		
		elif(obj_type == "color"):
			Colors.validate_color(obj)
		
		elif(obj_type == "boolean"):
			if(obj):
				return True
			else:
				return False
		else:
			raise TypeError("Invalid obj_type: " + obj_type)
		
	
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
		parsed_translation = Workspace.__validation_check(translation, "position")
		for i in range(0,len(parsed_translation)):
			self._position[i]+= parsed_translation[i]
			
		return self.clone()
	
	def move(self, new_position):
		self._position = Workspace.__validation_check(new_position, "position")
		return self.clone()
		
	
	def rotate(self, rotation):
		parsed_rotation = Workspace.__validation_check(rotation, "rotation")
		for i in range(0,len(parsed_rotation)):
			self._rotation[i]+= parsed_rotation[i]
			
		return self.clone()
	
	def set_rotation(self, new_rotation)
		self._rotation = Workspace.__validation_check(new_rotation, "rotation")
		return self.clone()
	def color(self, color):
		self._color = Workspace.__validation_check(color, "color")
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
			face.append(self.__get_point_index(p))
		
		self.faces.append(face)
		return face
	
	def build(self):
		return Polyhedron(points = self.points, faces = self.faces)
			
