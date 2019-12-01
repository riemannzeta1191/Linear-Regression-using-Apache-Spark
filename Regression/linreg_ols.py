import sys
import  numpy as np
from pyspark import SparkContext


def computeFirstPart(A):
	c = np.array(A).astype('float')
	c_1 =np.asmatrix(c)
	X = np.insert(c_1, 0, 1, axis=1)
	d = X.T
	return np.dot(d,X)


def computeSecondPart(B,C):
	Bm = np.asmatrix(np.array(B).astype('float'))
	B_1 = np.insert(Bm,0,1,axis=1)
	BT = B_1.T
	Cm = np.asmatrix(np.array(C).astype('float'))
	dot = np.dot(BT,Cm)
	return dot




if __name__=="__main__":
	sc = SparkContext(appName="linreg")
	yxinput = sc.textFile(sys.argv[1])
	# print(yxinput.collect())
	li = yxinput.collect()
	lines = yxinput.map(lambda p:p.split(','))
	# print(first_part.collect())
	first_part = lines.map(lambda x:("elemA",computeFirstPart(x[1]))).reduce(lambda e,f :np.add(e,f))
	print("first_part",first_part)
	second_part = lines.map(lambda l:computeSecondPart(l[1:],l[0])).reduce(lambda e,f :np.add(e,f))

	print("second_part",second_part)
	# second_part = np.asmatrix(second_part)
	first_part_inverse = np.linalg.inv((first_part[1]))
	print("first_part_inverse",first_part_inverse)

	product = np.dot(first_part_inverse,second_part).tolist()
	print("product",product)


