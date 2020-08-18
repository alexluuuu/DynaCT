"""utils.py

	did not get this to run lol
"""

import os
import pickle 
import numpy as np
# import slicer
# from slicer.util import *

# import vtk.util.numpy_support as nps
	

def compute_disparity(cloud1, cloud2): 

	return np.linalg.norm(cloud1 - cloud2, axis=1)
	

def write_to_file(name, obj):
	'''
		Write object of specified name to a pickle file 
	'''

	print 'writing structures to pickle'
	print '----------------------------'

	path = os.getcwd() + '/pickles/' + name + '.pkl'
	file = open(path, 'wb')
	pickle.dump(obj, file)
	file.close()


def procrustes(X, tol=.01): 
    """procrustes transformation
    
    Args:
        X (TYPE): Description
        tol (float, optional): Description
    
    Returns:
        TYPE: Description
    
    """
    nshape, nfeat = X.shape
    initial_shape = X[0].reshape((nfeat/1000, 3), order="F")

    aligned = np.zeros(X)

    # procrustes loop
    for iteration in range(10): 

    	#  alignment to mean shape or initial guess
	    for i, shape in enumerate(X[1:]):
	    	shape = shape.reshape((nfeat/1000, 3), order="F")
	    	initial_shape, shape, disparity = procrustes(initial_shape, shape)

	    	if i == 0: 
	    		aligned[0] = initial_shape.reshape(nfeat*3)

	    	aligned[i+1] = shape.reshape(nfeat*3)

	    # (re)compute the mean shape
	    mean_shape = np.mean(aligned, axis=0)
	    mean_shape, initial_shape, disparity = procrustes(mean_shape, initial_shape)
	    
	    # check if mean shape has changed
	    if disparity > tol: 
	    	initial_shape = mean_shape

	    else:
	    	break

    return X


def main(): 

	segmentation_base = "/Users/Alex/Downloads/"
	segmentation_list = [
		"IR_134", 
		"IR_138", 
	#    "IR_139", 
		"IR_140", 
	#    "IR_142", 
		"IR_144",
		"IR_146", 
		"IR_148", 
		"IR_150",
		"IR_144\ but\ not\ actually"
	]

	options = [
		"Segmentation RT ",
		"Segmentation LT ", 
	]

	output_base = "/Users/Alex/Documents/TaylorLab/segmentation_raw/"

	for scan in segmentation_list: 
		scan_id = scan[-3:]
		for option in options: 
			file_name = segmentation_base + scan + "/" + option + scan_id + "/" + option + scan_id + ".seg.nrrd"

			try: 
				segmentation = slicer.util.loadSegmentation(file_name)
			except: 
				import sys 
				print(sys.modules.keys())

			segmentation_node = getNode(option + scan_id)
			segmentId = segmentationNode.GetSegmentation().GetNthSegmentID(0)
			segmentPolyData = segmentationNode.GetClosedSurfaceRepresentation(segmentId)
			pointData = segmentPolyData.GetPoints().GetData()
			pointCoordinates = nps.vtk_to_numpy(pointData)

			with open(output_base + option + scan_id + '.npy', 'wb') as f: 
				np.save(f, pointCoordinates)


			# data, header = nrrd.read(file_name)			
			# print data
			# print header
			
			break
		break
	

if __name__ == "__main__": 
	main()
