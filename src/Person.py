import Image_Face
class Person(object):
	"""docstring for ClassName"""
	def __init__(self, id_cmt,name,address,age,job,phone_number,last_address_transaction,first_address_transaction,image,firtdate_transaction,lastdate_transaction):
		self.id_cmt = id_cmt
		self.name = name
		self.address = address
		self.age = age
		self.job = job
		self.phone_number = phone_number
		self.last_address_transaction = last_address_transaction
		self.first_address_transaction = first_address_transaction
		self.image = image
		self.firtdate_transaction = firtdate_transaction
		self.lastdate_transaction = lastdate_transaction
	def serialize(self):  
		return {           
        	'id_cmt': self.id_cmt.encode("utf8"), 
            'name': self.name.encode("utf8"),
			'address': self.address.encode("utf8"),
			'age': self.age,
			"job": self.job.encode("utf8"),
			"phone_number": self.phone_number.encode("utf8"),
			"last_address_transaction": self.last_address_transaction.encode("utf8"),
			"first_address_transaction": self.first_address_transaction.encode("utf8"),
			"firtdate_transaction": str(self.firtdate_transaction),
			"lastdate_transaction": str(self.lastdate_transaction)
        }
	def serialize_new(self):
		return {           
        	'id_cmt': self.id_cmt.encode("utf8"), 
            'name': self.name.encode("utf8"),
			'address': self.address.encode("utf8"),
			'age': self.age,
			"job": self.job.encode("utf8"),
			"phone_number": self.phone_number,
			"last_address_transaction": self.last_address_transaction.encode("utf8"),
			"first_address_transaction": self.first_address_transaction.encode("utf8"),
			"image" : [ob.serialize() for ob in self.image],
			"firtdate_transaction": str(self.firtdate_transaction),
			"lastdate_transaction":str(self.lastdate_transaction)
        }