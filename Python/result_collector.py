class Result(object):
    def __init__(self):
        self.res_dict = {
            'shoplifting_action': dict(),
            'arson_action': dict(),
            'det_face': [],
            'rec_face': []
        }

    def update(self, res, name):
        if name == 'det_face' or 'rec_face':
            self.res_dict[name] = res
        else:
            self.res_dict[name].update(res)

    def get(self, name):
        if name in self.res_dict and len(self.res_dict[name]) > 0:
            return self.res_dict[name]
        return None

    def clear(self, name):
        self.res_dict[name].clear()