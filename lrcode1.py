import test0704 as tm
reload(tm)

testSet='/Users/wumengling/Documents/machinelearninginaction-master/Ch05/testSet.txt'
X,y = tm.readData(testSet)
# tm.sigmoidFun(100)
theta = tm.stoGradient(X,y,500,0.001)
