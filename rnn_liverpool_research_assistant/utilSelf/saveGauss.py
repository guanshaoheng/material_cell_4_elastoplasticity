import os

def saveGauss2D(name='', pos=(),special_str=None,  **kwargs):
    '''
    saveGauss2D(name='./result/gauss/time_' + str(t) + '.dat', strain=strain, stress=stress, fabric=fabric)
    :param name:
    :param pos:
    :param kwargs:
    :return:
    '''
    fout = open(name, 'w')
    for key in kwargs:  # strain, stress, fabric
        try:
            data = kwargs[key].toListOfTuples()
        except:
            data = kwargs[key]
        if len(pos) == 0:
            fout.write('%s ' % key + str(len(data)) + '\n')
            for i in range(len(data)):
                if key == 'vR' or key == 'index_large_error' or key == 'H_0' or key == 'H_1' or key == 'numg_index':
                    fout.write('%s\n' % data[i])
                elif key == 'tangent' or key == 'D':
                    temp = [data[i][0][0][0][0], data[i][0][1][0][0], data[i][1][1][0][0],
                            data[i][0][1][0][1], data[i][0][1][1][1], data[i][1][1][1][1]]
                    fout.write(' '.join('%s' % x for x in temp)+'\n')
                elif key == 'frobeniusNorm' or key == 'iteration' or key=='epsPlastic':
                    fout.write('%.4e\n' % data[i])
                elif key == 'epsPlasticVector':
                    fout.write(' '.join('%s' % x for x in data[i])+'\n')
                else:
                    # if special_str is None:
                    #     fout.write(' '.join('%s %s' % (x[0], x[1]) for x in data[i]) + '\n')  #
                    # else:
                    #     fout.write(' '.join('%s' % x for x in data[i]) + '\n')  #
                    fout.write(' '.join('%s %s' % (x[0], x[1]) for x in data[i]) + '\n')
        else:
            fout.write('%s ' % key + str(len(pos)) + '\n')
            for i in pos:
                fout.write(' '.join('%s %s' % x for x in data[i]) + '\n')
    fout.close()


def saveGauss3D(name='', pos=(), **kwargs):
    fout = open(name, 'w')
    for key in kwargs:
        data = kwargs[key].toListOfTuples()
        if len(pos) == 0:
            fout.write('%s ' % key + str(len(data)) + '\n')
            for i in range(len(data)):
                fout.write(' '.join('%s %s %s' % x for x in data[i]) + '\n')
        else:
            fout.write('%s ' % key + str(len(pos)) + '\n')
            for i in pos:
                fout.write(' '.join('%s %s %s' % x for x in data[i]) + '\n')
    fout.close()


def save_loading(save_path, t: int, iter=None, special_str=None,  **kwargs):
    '''
        t: is the time step
        iter: is the iteration num in implicit mode
        kwargs: are values to be saved
    '''
    dir_name = 'iteration_gauss' if special_str is None else 'added_points'
    if iter == None:
        fname = os.path.join(
            save_path, '%s/time_%d' % (dir_name, t))
    else:
        fname = os.path.join(
            save_path, '%s/time_%d_iter_%d' % (dir_name, t, iter))
    if special_str is not None:
        fname += '_%s.dat' % special_str
    else:
        fname += '.dat'
    saveGauss2D(name=fname,special_str=special_str, **kwargs)