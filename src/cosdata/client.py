class Client(object):
    def __init__(self,host, base_url):
        self.host = host
        self.base_url = base_url
    
    def create_db(self, vector_name, description, dimension: int):
        pass

    def create_index(self, index_name, distance_metric):
        pass