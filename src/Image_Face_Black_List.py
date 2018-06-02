import numpy as np
def convertNumpytoVec(vector):
    vector = np.reshape(vector, -1)
    result = ''
    for i in range(len(vector)):
        result += str(vector[i])+','
    result = result[:result.rfind(',')]
    return result
class Image_Face_Black_List(object):
    """docstring for ClassName"""
    def __init__(self,path_image,emb_vector):
        self.path_image = path_image
        self.emb_vector = emb_vector
    def serialize(self):
        return {
        "path_image": self.path_image
    }
    def serialize_new(self):
        convert = convertNumpytoVec(self.emb_vector)
        return {
        "emb_vector": str(convert),
        "path_image": self.path_image
    }
        