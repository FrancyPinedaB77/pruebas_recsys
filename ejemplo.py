import recsys.algorithm
from recsys.datamodel.data import Data
from recsys.algorithm.factorize import SVD
recsys.algorithm.VERBOSE = True

#Load a dataset


svd = SVD()
svd.load_data(filename='./data/ratings.dat', sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})


#Haciendo el split al dataset
filename = './data/ratings.dat'
data = Data()
format = {'col':0, 'row':1, 'value':2, 'ids': int}
data.load(filename, sep='::', format=format)
train_80, test_20 = data.split_train_test(percent=80) # 80% train, 20% test
svd = SVD()
svd.set_data(train_80)

#Ingresando  variables para crear la matrizx
k = 100
svd.compute(k=k, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True)

k = 100
svd.compute(k=k, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True, savefile='./temporal/')

#Hallando similitud entre  2 items 
from recsys.algorithm.factorize import SVD
svd2 = SVD(filename='./temporal/') # Loading already computed SVD model
# Get two movies, and compute its similarity:
ITEMID1 = 1    # Toy Story (1995)
ITEMID2 = 2355 # A bug's life (1998)
print "similaridad entre items  sin usar la matrix que ya se genero "
print svd2.similarity(ITEMID1, ITEMID2)

print "similaridad entre items usando la matrix guardada"
print svd.similarity(ITEMID1, ITEMID2)


print "Recomendaciones para el itemid1"
print svd.similar(ITEMID1)

#Haciendo las predicciones 
MIN_RATING = 0
MAX_RATING = 5
ITEMID = 1
USERID = 1
print svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING)
print svd.get_matrix().value(ITEMID, USERID)

#HACUIENDO RECOMENDACIONES AL USUARIO Y POR TITEM 
print svd.recommend(USERID, is_row=False) #cols are users and rows are items, thus we set is_row=False
print svd.recommend(ITEMID)
print "se deben mostrar 5 recomendaciones para el item 1"
print svd.recommend(USERID, n=5, only_unknowns=True, is_row=False) 


#usando la matriz que ya esta generada 
from recsys.utils.svdlibc import SVDLIBC
svdlibc = SVDLIBC('./data/ratings.dat')
svdlibc.to_sparse_matrix(sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})
svdlibc.compute(k=100)
svd = svdlibc.export()
print "hallando similitud con el itemd1"
print svd.similar(ITEMID1) # results might be different than example 4. as there's no min_values=10 set here
