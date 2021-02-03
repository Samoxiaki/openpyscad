import openpyscad as ops
from openpyscad.ops_math import OpsMath as math

class Custom2dShapes(object):

	@staticmethod
	def regular_polygon(num, r, offset=0):
		"""
		Return a regular polygon with 'num' sides and 'r' radius.
		Set a value to offset in degrees to apply a rotation.  
		"""
		points = []
		for i in range(num):
			angle = 90 + ((i * 360) / num)
			points.append([r * math.cos(angle), r * math.sin(angle)])
		
		res = ops.Polygon(points)
		return res if(offset == 0) else res.rotate([0,0,offset])

	# Rehacerlo
	@staticmethod
	def star(num, r_out, r_in, offset=0):
		"""
		Return a star with 'num' vertex, 'r_out' outer radius and 'r_in' inner radius.
		Set a value to offset in degrees to apply a rotation.  
		"""
		points = []
		iterations = num*2
		
		for i in range(iterations):
			angle = 90 + ((i * 360) / iterations)
			r = r_out if (i % 2 == 0) else r_in
			points.append([r * math.cos(angle), r * math.sin(angle)])
		
		res = ops.Polygon(points)
		return res if(offset == 0) else res.rotate([0,0,offset])

	@staticmethod
	def sector(r, angle, sectors=360, offset=0):
		"""
		Return a circular sector with 'r' radius and 'angle' as angle length.
		Set a value to offset in degrees to apply a rotation. 
		"""
		circle = ops.Circle(r=r, _fn=sectors, center=True)
		angle = angle%360
		height = math.sin(angle)*r
		width = math.cos(angle)*r
		
		points = [[0,0], [0,0],[0,0]]
		
		if(angle<=90):
			max_height = math.sin(angle)*r
			points = [
				[0,0],
				[r, 0],
				[r, max_height],
				[width, height],
			]
		
		elif(90 < angle <= 180):
			max_height = r
			points = [
				[0,0],
				[r, 0],
				[r, max_height],
				[width, max_height],
				[width, height],
			] 
		else:
			points = [
				[0, 0],
				[r, 0],
				[r, r],
				[-r, r],
				[-r, 0],
			]
			
			if(180 < angle < 270):
				max_height = math.sin(angle)*r
				points.append([-r, max_height])
				points.append([width, height])
				
			else:
				max_height = -r
				points.append([-r, max_height])
				points.append([width, max_height])
				points.append([width, height])
			
		polygon = ops.Polygon(points=points)
		res = circle & polygon
		
		return res if(offset == 0) else res.rotate([0,0,offset])

	@staticmethod
	def arc(r, d, angle, sectors=360, offset=0):
		"""
		Return a circular arc with 'r' radius, 'd' as the difference between inner and outer radius and 'angle' as angle length.
		Set a value to offset in degrees to apply a rotation. 
		"""
		res = Custom2dShapes.sector(r, angle, sectors) - Custom2dShapes.sector(r-d, angle, sectors)
		return res if(offset == 0) else res.rotate([0,0,offset])
	
	
	
	
	
	
	
	
	
	
	
	
