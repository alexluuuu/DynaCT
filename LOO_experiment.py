"""LOO_experiment.py

"""

import numpy as np
import os 

from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes

from pycpd import DeformableRegistration, RigidRegistration

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import mpl_toolkits.mplot3d.axes3d as p3
# import mpl_toolkits.mplot3d as plt3d
# import matplotlib.animation as animation

from utils import write_to_file, compute_disparity


def initial_rigid_registration(segmentation_list, save_path, n_target=500, n_source=800, keep_id=4): 

	print "rigid registration"

	# prepare the source point cloud
	seg_path, identifier = segmentation_list[0]
	identifier = identifier.replace(" ", "")
	target_cloud = np.load(save_path + identifier + "_" + str(keep_id) + ".npy")
	indices = np.random.randint(0, target_cloud.shape[0]-1, size=n_target)
	target_cloud = target_cloud - np.mean(target_cloud, axis=0)
	target_cloud = target_cloud[indices]

	# data structures for storing transformations, correspondences
	ss = []
	Rs = []
	ts = []
	Ps = []
	corresponded_sources = []
	disparities = []
	transformed_sources = []

	# iterate over the target point clouds
	for seg_path, identifier in segmentation_list[1:]:
		
		identifier = identifier.replace(" ", "")

		print '-- ' + identifier 

		source_cloud = np.load(save_path + identifier + "_" + str(keep_id) + ".npy")
		indices_source = np.random.randint(0, source_cloud.shape[0]-1, size=n_source)
		source_cloud = source_cloud - np.mean(source_cloud, axis=0)
		if "LT" in identifier: 
			source_cloud[:,0] = -1.*source_cloud[:,0]

		source_cloud = source_cloud[indices_source]

		reg = RigidRegistration(**{'X': target_cloud, 'Y': source_cloud})
		reg.register()
		s, R, t = reg.get_registration_parameters()
		P = reg.P

		ss.append(s)
		Rs.append(R)
		ts.append(t)
		Ps.append(P)
		
		corresponded = source_cloud[np.argmax(P, axis=0), :]
		corresponded_sources.append(corresponded) 

		transformed = s*np.dot(source_cloud, R) + t
		transformed = transformed[np.argmax(P, axis=0), :]
		transformed_sources.append(transformed)

		disparity = compute_disparity(corresponded, transformed)
		disparities.append(disparity)

	transformed_sources.append(target_cloud)
	all_initial = np.array(transformed_sources)

	mean_shape = np.mean(all_initial, axis=0)

	return ss, Rs, ts, Ps, corresponded_sources, disparities, transformed_sources, mean_shape


def deformable_registration(segmentation_list, save_path, Rs, ss, ts, mean_shape, n_source=500, n_target=1000, keep_id=4, apply_scale=True): 

	print "deformable registration"

	source_cloud = mean_shape

	# data structures for storing transformations, correspondences
	Gs = []
	Ws = []
	Ps = []
	corresponded_targets = []

	disparities = []

	# iterate over point clouds, setting each as the target for the mean shape to be deformed to
	for i, (seg_path, identifier) in enumerate(segmentation_list):
		identifier = identifier.replace(" ", "")

		print '-- ' + identifier 
		target_cloud = np.load(save_path + identifier + "_" + str(keep_id) + ".npy")
		indices_target = np.random.randint(0, target_cloud.shape[0]-1, size=n_target)
		target_cloud = target_cloud - np.mean(target_cloud, axis=0)
		if "LT" in identifier: 
			target_cloud[:,0] = -1.*target_cloud[:,0]
			
		# access the rough rotation, scale, translations from the rigid CPD previously executed
		R = Rs[i-1]
		s = ss[i-1]
		t = ss[i-1]
		
		if i == 0: 
			R = np.eye(3)
			s = 1.
			t = 0.

		if not apply_scale: 
			s = 1.
			t = 0.


		# apply the similarity transform
		target_cloud = s * np.dot(target_cloud, R) + t
		target_cloud = target_cloud[indices_target]
	   
	   	# deformable registration
		reg = DeformableRegistration(**{'X': target_cloud, 'Y': source_cloud})
		reg.register()
		G, W = reg.get_registration_parameters()
		P = reg.P

		Gs.append(G)
		Ws.append(W)
		Ps.append(P)

		# apply correspondence, deformable registration
		corresponded = target_cloud[np.argmax(P, axis=1), :]
		corresponded_targets.append(corresponded) 

		transformed = source_cloud + np.dot(G, W)

		# compute the residual error before / after deformable transformation
		disparity = compute_disparity(corresponded, transformed)
		disparities.append(disparity)

		print np.sqrt(np.sum(disparity)) 

	return Gs, Ws, Ps, corresponded_targets, disparities


def LOO(segmentation_list, save_path, model_save_prefix="LOO_"): 

	# for each in segmentation list, omit and train separate model 

	# 	run the rigid registration

	# 	run the deformable registration 

	# 	save the model parameters

	for loo_id, (seg_path, identifier) in enumerate(segmentation_list): 

		identifier = identifier.replace(" ", "")

		print "executing LOO model with ", identifier, " left out"
		print "---"*10

		rest = segmentation_list[:loo_id] + segmentation_list[loo_id+1:]

		ss, Rs, ts, _, _, _, _, mean_shape = initial_rigid_registration(rest, save_path)

		Gs, Ws, Ps, corresponded_targets, disparities = deformable_registration(rest, save_path, Rs, ss, ts, mean_shape, apply_scale=False)

		write_to_file(model_save_prefix + identifier, (mean_shape, Gs, Ws, Ps, corresponded_targets, disparities))

	return 
	
		


def main():
	np.random.seed(2020) 

	save_path = "/Users/Alex/Documents/TaylorLab/segmentation_raw/"
	segmentation_list = [
			("/Users/Alex/Downloads/IR_134/Segmentation RT 134/Segmentation RT 134.seg.nrrd", "Segmentation RT 134"),
			("/Users/Alex/Downloads/IR_134/Segmentation LT 134/Segmentation LT 134.seg.nrrd", "Segmentation LT 134"),
			("/Users/Alex/Downloads/IR_138/Segmentation RT 138/Segmentation RT 138.seg.nrrd", "Segmentation RT 138"),
			("/Users/Alex/Downloads/IR_138/Segmentation LT 138/Segmentation LT 138.seg.nrrd", "Segmentation LT 138"),
			("/Users/Alex/Downloads/IR_140/Segmentation RT 140/Segmentation RT 140.seg.nrrd", "Segmentation RT 140"),
			("/Users/Alex/Downloads/IR_140/Segmentation LT 140/Segmentation LT 140.seg.nrrd", "Segmentation LT 140"),
			("/Users/Alex/Downloads/IR_144/Segmentation RT 144/Segmentation RT 144.seg.nrrd", "Segmentation RT 144"),
			("/Users/Alex/Downloads/IR_144/Segmentation LT 144/Segmentation LT 144.seg.nrrd", "Segmentation LT 144"),
			("/Users/Alex/Downloads/IR_146/Segmentation RT 146/Segmentation RT 146.seg.nrrd", "Segmentation RT 146"),
			("/Users/Alex/Downloads/IR_146/Segmentation LT 146/Segmentation LT 146.seg.nrrd", "Segmentation LT 146"),
			("/Users/Alex/Downloads/IR_148/Segmentation LT 148/Segmentation LT 148.seg.nrrd", "Segmentation LT 148"),
			("/Users/Alex/Downloads/IR_150/Segmentation RT 150/Segmentation RT 150.seg.nrrd", "Segmentation RT 150"),
			("/Users/Alex/Downloads/IR_150/Segmentation LT 150/Segmentation LT 150.seg.nrrd", "Segmentation LT 150"),
			("/Users/Alex/Downloads/IR_144\ but\ not\ actually/Segmentation RT lly/Segmentation RT 144.seg.nrrd", "Segmentation RT 147"),
			("/Users/Alex/Downloads/IR_144\ but\ not\ actually/Segmentation LT 144/Segmentation LT 144.seg.nrrd", "Segmentation LT 147"),
	]

	LOO(segmentation_list, save_path)



if __name__ == "__main__": 
	main()
