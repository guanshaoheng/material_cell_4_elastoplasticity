import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 200
# fix random seeds
axes = {'labelsize' : 'large'}
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 17}
legend = {'fontsize': 'large'}
lines = {'linewidth': 3,
         'markersize': 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)

colorlist = ['#008080', '#FF7F50', '#4169E1', '#DA70D6', '#808000', '#4B0082',
             '#FF8C00', '#FF1493', '#4682B4', '#DAA520', '#9370DB',
             '#2E8B57', '#483D8B', '#FF6347', '#008B8B', '#BA55D3',
             '#B8860B', '#1E90FF', '#3CB371'
             ]


'''
    Work in  Liverpool Univeristy
    
    This is used to plot the loading path used in the data generation

'''

path = './random_strain_path/paths_201.npy'
with open(path, mode='rb') as f:
    loading_paths = np.load(f)

# plot
index = [118, 36, 196]
colorlist_temp = np.random.choice(colorlist, 3, replace=False)
# index = np.random.randint(0, 200, 3)
fig, axs = plt.subplots(figsize=[14, 5], nrows=1, ncols=3, sharey=True)
for i, ax in enumerate(axs):
    ax.plot(loading_paths[index[i], :, 0], color=colorlist_temp[0])
    ax.plot(loading_paths[index[i], :, 1], color=colorlist_temp[1])
    ax.plot(loading_paths[index[i], :, 2], color=colorlist_temp[2])
    ax.set_xlabel('Loading steps')
    ax.grid()
plt.legend(['$\epsilon_{00}$', '$\epsilon_{01}$', '$\epsilon_{11}$'])
# plt.ylabel('Strain')
plt.tight_layout()
plt.show()
print()