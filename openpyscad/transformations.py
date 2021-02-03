# -*- coding: utf-8 -*-
import openpyscad.base as base

__all__ = ['Translate', 'Rotate', 'Scale', 'Resize', 'Mirror', 'Color', 'Offset', 'Hull', 'Minkowski', 'Linear_Extrude', 'Rotate_Extrude', 'Colors']


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
			
class Colors:
	lavender = "Lavender"
	thistle = "Thistle"
	plum = "Plum"
	violet = "Violet"
	orchid = "Orchid"
	fuchsia = "Fuchsia"
	magenta = "Magenta"
	medium_orchid = "MediumOrchid"
	medium_purple = "MediumPurple"
	blue_violet = "BlueViolet"
	dark_violet = "DarkViolet"
	dark_orchid = "DarkOrchid"
	dark_magenta = "DarkMagenta"
	purple = "Purple"
	indigo = "Indigo"
	dark_slate_blue = "DarkSlateBlue"
	slate_blue = "SlateBlue"
	medium_slate_blue = "MediumSlateBlue"
	pink = "Pink"
	light_pink = "LightPink"
	hot_pink = "HotPink"
	deep_pink = "DeepPink"
	medium_violet_red = "MediumVioletRed"
	pale_violet_red = "PaleVioletRed"
	aqua = "Aqua"
	cyan = "Cyan"
	light_cyan = "LightCyan"
	pale_turquoise = "PaleTurquoise"
	aquamarine = "Aquamarine"
	turquoise = "Turquoise"
	medium_turquoise = "MediumTurquoise"
	dark_turquoise = "DarkTurquoise"
	cadet_blue = "CadetBlue"
	steel_blue = "SteelBlue"
	light_steel_blue = "LightSteelBlue"
	powder_blue = "PowderBlue"
	light_blue = "LightBlue"
	sky_blue = "SkyBlue"
	light_sky_blue = "LightSkyBlue"
	deep_sky_blue = "DeepSkyBlue"
	dodger_blue = "DodgerBlue"
	cornflower_blue = "CornflowerBlue"
	royal_blue = "RoyalBlue"
	blue = "Blue"
	medium_blue = "MediumBlue"
	dark_blue = "DarkBlue"
	navy = "Navy"
	midnight_blue = "MidnightBlue"
	indian_red = "IndianRed"
	light_coral = "LightCoral"
	salmon = "Salmon"
	dark_salmon = "DarkSalmon"
	light_salmon = "LightSalmon"
	red = "Red"
	crimson = "Crimson"
	fire_brick = "FireBrick"
	dark_red = "DarkRed"
	green_yellow = "GreenYellow"
	chartreuse = "Chartreuse"
	lawn_green = "LawnGreen"
	lime = "Lime"
	lime_green = "LimeGreen"
	pale_green = "PaleGreen"
	light_green = "LightGreen"
	medium_spring_green = "MediumSpringGreen"
	spring_green = "SpringGreen"
	medium_sea_green = "MediumSeaGreen"
	sea_green = "SeaGreen"
	forest_green = "ForestGreen"
	green = "Green"
	dark_green = "DarkGreen"
	yellow_green = "YellowGreen"
	olive_drab = "OliveDrab"
	olive = "Olive"
	dark_olive_green = "DarkOliveGreen"
	medium_aquamarine = "MediumAquamarine"
	dark_sea_green = "DarkSeaGreen"
	light_sea_green = "LightSeaGreen"
	dark_cyan = "DarkCyan"
	teal = "Teal"
	light_salmon = "LightSalmon"
	coral = "Coral"
	tomato = "Tomato"
	orange_red = "OrangeRed"
	dark_orange = "DarkOrange"
	orange = "Orange"
	gold = "Gold"
	yellow = "Yellow"
	light_yellow = "LightYellow"
	lemon_chiffon = "LemonChiffon"
	light_goldenrod_yellow = "LightGoldenrodYellow"
	papaya_whip = "PapayaWhip"
	moccasin = "Moccasin"
	peach_puff = "PeachPuff"
	pale_goldenrod = "PaleGoldenrod"
	khaki = "Khaki"
	dark_khaki = "DarkKhaki"
	cornsilk = "Cornsilk"
	blanched_almond = "BlanchedAlmond"
	bisque = "Bisque"
	navajo_white = "NavajoWhite"
	wheat = "Wheat"
	burly_wood = "BurlyWood"
	tan = "Tan"
	rosy_brown = "RosyBrown"
	sandy_brown = "SandyBrown"
	goldenrod = "Goldenrod"
	dark_goldenrod = "DarkGoldenrod"
	peru = "Peru"
	chocolate = "Chocolate"
	saddle_brown = "SaddleBrown"
	sienna = "Sienna"
	brown = "Brown"
	maroon = "Maroon"
	white = "White"
	snow = "Snow"
	honeydew = "Honeydew"
	mint_cream = "MintCream"
	azure = "Azure"
	alice_blue = "AliceBlue"
	ghost_white = "GhostWhite"
	white_smoke = "WhiteSmoke"
	seashell = "Seashell"
	beige = "Beige"
	old_lace = "OldLace"
	floral_white = "FloralWhite"
	ivory = "Ivory"
	antique_white = "AntiqueWhite"
	linen = "Linen"
	lavender_blush = "LavenderBlush"
	misty_rose = "MistyRose"
	gainsboro = "Gainsboro"
	light_grey = "LightGrey"
	silver = "Silver"
	dark_gray = "DarkGray"
	gray = "Gray"
	dim_gray = "DimGray"
	light_slate_gray = "LightSlateGray"
	slate_gray = "SlateGray"
	dark_slate_gray = "DarkSlateGray"
	black = "Black"

	@staticmethod
	def custom(r, g, b, a=255):
		"""
		Return custom color. RGBA [0-255]
		"""
		red_c = max(0,min(255,r))
		green_c = max(0,min(255,g))
		blue_c = max(0,min(255,b))
		alpha_c = max(0,min(255,a))
		
		return "#" + hex(int(red_c))[2::] + hex(int(green_c))[2::] + hex(int(blue_c))[2::] + hex(int(alpha_c))[2::]
	
	@staticmethod
	def all_colors():
		"""
		Return a dict containing all valid colors.
		"""
		colors = {}
		for c in dir(Colors):
			if(not c.startswith("__")):
				attribute = getattr(Colors, c)
				if(isinstance(attribute, str)):
					colors[c] = attribute
		return colors
		
	
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
		for k, v in Colors.all_colors().values():
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
				return ("#" if(offset_1) else "") + red + green + blue + alpha
			except:
				raise TypeError("Invalid color: " + color)
		else:
			raise TypeError("Invalid color: " + color)
			
