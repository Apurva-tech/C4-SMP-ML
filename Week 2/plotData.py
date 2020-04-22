import matplotlib.pyplot as plt
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
	# Get indexes for class 0 and class 1
	neg = data[:,2] == 0
	pos = data[:,2] == 1

	# If no specific axes object has been passed, get the current axes.
	if axes == None:
	    axes = plt.gca()
	#Plot data as scatter plot
	axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
	axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
	# set label values
	axes.set_xlabel(label_x)
	axes.set_ylabel(label_y)
	axes.legend(frameon= True, fancybox = True)
