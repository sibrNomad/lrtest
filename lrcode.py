import math
import numpy as np

def readData(data_path):
	X = []; label = []
	f = open(data_path, 'r')
	for l in f.readlines():
		x1,x2,y = l.strip().split('\t')
		X.append([1, float(x1), float(x2)])
		label.append(int(y))
	return [X,label]
	
def sigmoidFun(z):
	return 1/(1 + np.exp(-z))

def stoGradient(X,y,max_iter,alpha):
	X_mat = np.matrix(X); y_mat = np.matrix(y).transpose()
	m,n = X_mat.shape
	theta = np.ones((n,1))
	for i in range(max_iter):
		h = sigmoidFun(X_mat*theta)
		error = h - y_mat
		theta = theta - alpha*(X_mat.transpose()*error)
	return theta
