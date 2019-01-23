import csv 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import matplotlib.pyplot as plt 

def makeModel(i):
	"makes a model for team i"
	y=[]
	with open('TM.csv','r') as csvfile: 
		plots = csv.reader(csvfile, delimiter=',') 
		for row in plots:
			if row[3]==str(i): #swap team2 and team1 if team1 is the ith team
				t=row[3]
				row[3]=row[2]
				row[2]=t
			if row[2]==str(i): #if team(i) has won the toss modify row as 1 else 0
				if row[2]==row[4]:
					row[4]=1
				else:
					row[4]=0
				if row[2]==row[7]: #if team(i) has won the match modify row as 1 else 0
					row[7]=1
				else:
					row[7]=0
				y.append(row)	
	#creating a csv file for each team
	myFile = open('NB'+str(i)+'.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(y)

	##reading the csv file to fit the model    
	data=pd.read_csv('NB'+str(i)+'.csv')
	X = data.iloc[:, [0,1,3,4,5,6,8]].values
	Y = data.iloc[:, 7].values	
	# Random Forest Model Creation
	rf_model =  RandomForestClassifier(max_depth=20,n_estimators=500, min_samples_split=2, random_state=1)
	return rf_model.fit(X,Y)

#generate an array of models corresponding to each team
model=[]
for i in range(11):
	model.append(makeModel(i+1))

#test the data
data=pd.read_csv('TestM.csv')

X_test = data.iloc[:, [0,1,2,3,4,5,6,8]].values
y_pred=[]
for val in X_test:
	v1=model[val[2]-1].predict_proba([[val[0]   ,val[1]    ,val[3]    ,int(val[4]==val[2])    ,val[5]    ,val[6]   ,val[7]]])#predicting the probability for team1
	v2=model[val[3]-1].predict_proba([[val[0]   ,val[1]    ,val[2]    ,int(val[4]==val[3])    ,val[5]    ,val[6]   ,val[7]]])#predicting the probability for team2
	if (v1[0][1]+v2[0][0])>(v1[0][0]+v2[0][1]): #if winning probability of team1>team2 then prediction is 1 else 0
		y_pred.append(1)
	else:
		y_pred.append(0)
#creating the submission file
r=[]
r.append(['match_id','team_1_win_flag'])
for i in range(501,501+len(y_pred)):
	r.append([i,y_pred[i-501]])
myFile = open('submission.csv', 'w') 
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(r)

# plot the generated outputs
x=range(len(y_pred))
plt.plot(x,y_pred, label='RF', color="red")
plt.xlabel('Match')
plt.ylabel('Winner') 
plt.title('Cricket Match\n Winner Prediction') 
plt.legend() 
plt.show()