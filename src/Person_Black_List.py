import Image_Face_Black_List
class Person_Black_List(object):
	"""docstring for ClassName"""
	def __init__(self, id,name,age,address,image,image_cmt):
		self.id = id
		self.name = name
		self.address = address
		self.age = age
		self.image = image
		self.image_cmt= image_cmt
	def serialize(self):  
		return {           
        	'id': self.id.encode("utf8"), 
            'name': self.name.encode("utf8"),
			'address': self.address.encode("utf8"),
			'age': self.age,
			'image': self.image.serialize(),
			'image_cmt': self.image_cmt.serialize()
        }
	def serialize_new(self):  
		return {           
        	'id': self.id.encode("utf8"), 
            'name': self.name.encode("utf8"),
			'address': self.address.encode("utf8"),
			'age': self.age,
			'images': self.image.serialize_new(),
			'image_cmt': self.image_cmt.serialize_new()
        }