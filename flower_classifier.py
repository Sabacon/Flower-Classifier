import streamlit as st
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

header=st.container()
dataset=st.container()
features=st.container()
model=st.container()
prediction=st.container()

@st.cache
def get_data(file_name):
	iris=pd.read_csv(file_name)
	return iris

with header:
	st.title('Classify Your Flowers')
	st.text('I classify different flowers based on their physical features')


with dataset:
	st.header('Iris dataset')
	st.text('I use the Iris dataset from Kaggle')
	st.text("Here's a snippet")

	iris=get_data('iris.csv')
	st.write(iris.tail())


with features:
	st.header('Relevant features')
	
	st.text('Enter the dimensions of your flower')
	sep_len,sep_wid=st.columns(2)
	sepal_length=sep_len.number_input('sepal length', min_value=0.1, max_value=10.0, step=0.1)
	sepal_width=sep_wid.number_input('sepal width', min_value=0.1, max_value=10.0, step=0.1)

	pet_len,pet_wid=st.columns(2)
	petal_length=pet_len.number_input('petal length', min_value=0.1, max_value=10.0, step=0.1)
	petal_width=pet_wid.number_input('petal width', min_value=0.1, max_value=10.0, step=0.1)


with model:
	st.header('K Neighbors Classifier model')
	st.text('Select the number of neighbors to train the model')

	n_neighbors=st.slider('n_neighbors', min_value=1, max_value=10, step=1, value=5)

	X=iris.drop('species',axis=1)
	y=iris['species']

	knn=KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(X,y)

	
with prediction:
	st.header('Prediction')

	if st.button('Run'):
		pred=knn.predict([[sepal_length,sepal_width,petal_length,petal_width]])


		st.text('Your flower is of species: {}'.format(pred))

