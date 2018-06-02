import numpy as np
def convertNumpytoVec(vector):
    vector = np.reshape(vector, -1)
    result = ''
    for i in range(len(vector)):
        result += str(vector[i])+','
    result = result[:result.rfind(',')]
    return result
class Image_Face(object):
    """docstring for ClassName"""
    def __init__(self, name,type,path_image,vector,date,location_transaction):
        self.name = name
        self.type = type
        self.path_image = path_image
        self.vector = vector
        self.date = date
        self.location_transaction = location_transaction
    def serialize(self):  
        convert = convertNumpytoVec(self.vector)
        return {
        "name":self.name.encode("utf8"),
        "type":int(self.type),
        "path_image": self.path_image.encode("utf8"),
        "vector": str(convert),
        "date": str(self.date),
        "location_transaction" :self.location_transaction.encode("utf8")
    }
        