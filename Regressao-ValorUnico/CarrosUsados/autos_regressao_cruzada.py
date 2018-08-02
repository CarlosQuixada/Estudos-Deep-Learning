import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)
# Conta o numero de valores distintos
# base['name'].value_counts()
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value=valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

label_enconder_previsores = LabelEncoder()
one_hot_encoder = OneHotEncoder(categorical_features=[0, 1, 3, 5, 8, 9, 10])

previsores[:, 0] = label_enconder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = label_enconder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = label_enconder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = label_enconder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = label_enconder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = label_enconder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = label_enconder_previsores.fit_transform(previsores[:, 10])

previsores = one_hot_encoder.fit_transform(previsores).toarray()
