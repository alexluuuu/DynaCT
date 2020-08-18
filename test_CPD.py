from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
import numpy as np


def visualize(iteration, error, X, Y, ax):
	plt.cla()
	ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
	ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
	ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
		iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
	ax.legend(loc='upper left', fontsize='x-large')
	plt.draw()
	plt.pause(0.001)


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

	keep_id = 4
	n_source= 1500
	n_target = 6000

	# prepare the source point cloud
	seg_path, identifier = segmentation_list[0]
	identifier = identifier.replace(" ", "")
	source_cloud = np.load(save_path + identifier + "_" + str(keep_id) + ".npy")
	indices = np.random.randint(0, source_cloud.shape[0]-1, size=n_source)
	source_cloud = source_cloud - np.mean(source_cloud, axis=0)
	source_cloud = source_cloud[indices]

	# data structures for storing transformations, correspondences
	Gs = []
	Ws = []
	Ps = []
	corresponded_targets = []

	disparities = []

	# iterate over the target point clouds
	for seg_path, identifier in segmentation_list[1:]:
		identifier = identifier.replace(" ", "")

		target_cloud = np.load(save_path + identifier + "_" + str(keep_id) + ".npy")
		indices_target = np.random.randint(0, target_cloud.shape[0]-1, size=n_target)
		target_cloud = target_cloud - np.mean(target_cloud, axis=0)
		if "LT" in identifier: 
		    target_cloud[:,0] = -1.*target_cloud[:,0]

		target_cloud = target_cloud[indices_target]

		reg = DeformableRegistration(**{'X': target_cloud, 'Y': source_cloud})
		reg.register()
		G, W = reg.get_registration_parameters()
		P = reg.P

		Gs.append(G)
		Ws.append(W)
		Ps.append(P)

		corresponded = target_cloud[np.argmax(P, axis=1), :]
		corresponded_targets.append(corresponded) 

		transformed = source_cloud + np.dot(G, W)

		disparity = np.sum(np.square(corresponded - transformed))
		disparities.append(disparity)


	

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# ax.scatter()


	# seg_path_LT, identifier_LT = segmentation_list[1]
	# identifier_LT = identifier_LT.replace(" ", "")

	

	# point_cloud_LT = np.load(save_path + identifier_LT + "_" + str(keep_id) + ".npy")
	# indices_LT = np.random.randint(0, point_cloud_LT.shape[0]-1, size=n_downsampled)
	# point_cloud_LT = point_cloud_LT[indices_LT]
	# point_cloud_LT = point_cloud_LT - np.mean(point_cloud_LT, axis=0)

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# callback = partial(visualize, ax=ax)

	# reg = DeformableRegistration(**{'X': point_cloud, 'Y': point_cloud_LT})
	# reg.register()

	# G, W = reg.get_registration_parameters()
	# P = reg.P

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# point_cloud_LT = point_cloud_LT + np.dot(G, W)
	# ax.scatter(point_cloud_LT[:,0], point_cloud_LT[:,1], point_cloud_LT[:,2], color='k')
	# ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
	# plt.show()



if __name__ == '__main__':
	main()