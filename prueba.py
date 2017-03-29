import recsys
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data


#CARGANDO LOS DATOS
filename = './ratings.dat'
svd = SVD()
svd.load_data(filename=filename, sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})

#HACIENDO SPLIT DATASSET PARA ENTRENAMIENTO Y test_20 
filename = './ratings.dat'
data = Data()
format = {'col':0, 'row':1, 'value':2, 'ids': int}
data.load(filename, sep='::', format=format)
train_80, test_20 = data.split_train_test(percent=80) # 80% train, 20% test

svd = SVD()
svd.set_data(train_80)

from recsys.utils.svdlibc import SVDLIBC
svdlibc = SVDLIBC('./ratings.dat')
svdlibc.to_sparse_matrix(sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})
svdlibc.compute(k=100)
svd = svdlibc.export()

K=100
print "usando k=100"
svdlibc.compute(k=100)
svd = svdlibc.export()
print svd
print "se acaba de imprimir svd "


#print svd.compute(k=K, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True, savefile=None)
# se quita 

#HACIENDO PREDICCIONES
ITEMID=595
USERID=1

print "haciendo predicciones"
print svd.predict(ITEMID, USERID, MIN_RATING=0.0, MAX_RATING=5.0)

#HACIENDO RECOMENDACIONES
print svd.recommend(USERID, n=10, only_unknowns=True, is_row=False)

