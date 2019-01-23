import csv 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

def makeModel(i):
	"makes a model for team i"
	y=[]
	with open('TM.csv','r') as csvfile: 
		plots = csv.reader(csvfile, delimiter=',') 
		for row in plots:
			if row[3]==str(i):
				t=row[3]
				row[3]=row[2]
				row[2]=t
			if row[2]==str(i):
				if row[2]==row[4]:
					row[4]=1
				else:
					row[4]=0
				if row[2]==row[7]:
					row[7]=1
				else:
					row[7]=0
				y.append(row)	
	myFile = open('NB'+str(i)+'.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(y)


	data=pd.read_csv('NB'+str(i)+'.csv')
	X = data.iloc[:, [0,1,3,4,5,6,8]].values
	Y = data.iloc[:, 7].values	
	X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state=0) 

	# Random Forest Model
	rf_model =  RandomForestClassifier(max_depth=14,n_estimators=500, min_samples_split=2, random_state=1)
	return rf_model.fit(X_train,y_train)


model=[]
for i in range(11):
	model.append(makeModel(i+1))


data=pd.read_csv('TM.csv')

X = data.iloc[:, [0,1,2,3,4,5,6,8]].values
Y = data.iloc[:, 7].values
for i in range(len(Y)):
	if Y[i]==X[i][2]:
		Y[i]=1
	else:
		Y[i]=0	
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)
y_pred=[]
for val in X_test:
	v1=model[val[2]-1].predict_proba([[val[0]   ,val[1]    ,val[3]    ,int(val[4]==val[2])    ,val[5]    ,val[6]   ,val[7]]])
	v2=model[val[3]-1].predict_proba([[val[0]   ,val[1]    ,val[2]    ,int(val[4]==val[3])    ,val[5]    ,val[6]   ,val[7]]])
	if (v1[0][1]+v2[0][0])>(v1[0][0]+v2[0][1]):
		y_pred.append(1)
	else:
		y_pred.append(0)

t=0
for i in range(len(y_pred)):
	if y_pred[i]==y_test[i]:
		t=t+1
print('Taking 20% of total data as testing set and training on 80% data gives the result: ')
print('Accuracy='+str((t*100)/len(y_pred))+'%')


# plot
x=range(len(y_pred))
plt.plot(x,y_pred, label='Predicted', color="blue")
plt.scatter(x,y_test, label='real', color="red") 
plt.xlabel('Matches')
plt.ylabel('Winner') 
plt.title('Winner prediction') 
plt.legend() 
plt.show()