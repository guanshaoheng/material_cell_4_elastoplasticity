from utilSelf.general import echo


class cylicLoadController:
    def __init__(self,
    unload_point ,
    reload_point):
        '''
        :param unload_point:  [-0.07, -0.08, -0.09]
        :param reload_point:  [-0.04, -0.04, -0.01]
        '''
        self.unload_point = unload_point
        self.reload_point =reload_point
        self.unloaded = False
        self.load_flag = 1.0
        self.num_point = 0
        self.un_load_point_temp, self.re_load_point_temp = unload_point[self.num_point], reload_point[self.num_point]

    def move(self, eps):
        if eps < self.un_load_point_temp and self.unloaded == False and self.load_flag == 1:  # change to unload
            self.load_flag = -1
            self.unloaded = True
            echo('Unloading')

        if eps > self.re_load_point_temp and self.unloaded == True and self.load_flag == -1:  # change to load
            line = 'Reloading'
            self.load_flag = 1
            if self.num_point < len(self.unload_point) - 1:  # if this is not the last unloading
                self.unloaded = False  # only by this can unloaded again
                self.num_point += 1
                self.un_load_point_temp, self.re_load_point_temp = self.unload_point[self.num_point], \
                                                                   self.reload_point[self.num_point]
            else:
                line += '_Last reloading'

            echo(line)
        return self.load_flag