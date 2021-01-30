# -*- coding: utf-8 -*-
import openpyscad.base as base

__all__ = ['Translate', 'Rotate', 'Scale', 'Resize', 'Mirror', 'Color', 'Offset', 'Hull', 'Minkowski', 'Linear_Extrude', 'Rotate_Extrude']


class _Transformation(base.BaseObject):
	pass


Transformation = _Transformation


# Transformations
class Translate(_Transformation):
	pass


class Rotate(_Transformation):
	pass


class Scale(_Transformation):
	pass


class Resize(_Transformation):
	pass


class Mirror(_Transformation):
	pass


class Color(_Transformation):
	pass


class Offset(_Transformation):
	def _validate_append(self, obj):
		
		if not isinstance(obj, base.BaseObject) or not obj._is_2d():
			raise TypeError('Appended object must be a 2D shape.')


class Hull(_Transformation):
	pass


class Minkowski(_Transformation):
	pass


class Linear_Extrude(_Transformation):
	def _validate_append(self, obj):

		if not isinstance(obj, base.BaseObject) or not obj._is_2d():
			raise TypeError('Appended object must be a 2D shape.')


class Rotate_Extrude(_Transformation):
	def _validate_append(self, obj):

		if not isinstance(obj, base.BaseObject) or not obj._is_2d():
			raise TypeError('Appended object must be a 2D shape.')
