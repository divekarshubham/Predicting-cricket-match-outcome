# Predicting-cricket-match-outcome
Probabilistic Classification using Regression Forest Ensemble built for each team playing the match.


Pre-processing:
Pre-processing on the data (training and testing)yields numeric entries for all the columns and relevant columns are selected as shown by the snapshot
Pre-processing is done so as to avoid encoding each time we train the data and for easy processing of numeric values.It also provides for direct feed into the model.

This classification approach predicts the winner of a certain match by taking into parameters :
1.	Season- The play of each team may vary per season due to change in players and their forms
2.	City: Homeground or familiar areas have a great impact on the performance of the team.
3.	Stadium: Similar to the choice of city as a parameter
4.	Opponent team: The performance of the team is dependant on the previous matches with the opposing teams.
5.	Toss winner: while plotting the data we see there is a corelation between the toss winner and the winning team. 
6.	Toss Decision:Similar to the above
7.	DL applied: if the match is affected by weather conditions Duckworth–Lewis plays an important role in deciding the winning team by providing an advantage

Attributes such as Player of the match are not chosen due to low correlation, we cannot determine to which team the player belongs to hence we rule out the column. Also win by run/wicket play a less significant role in the decision making process and can be seen as the outcomes rather than the inputs of an event.

Creation of Models:
1) For each team playing a separate ensemble of random forest is created taking into consideration the parameters as mentioned above. Method:
For each team ti playing:
Make a new csv file for team ti
Select all the rows which contain ti as one of the playing 	teams
Append attributes of the row to the file
2)	Fit the data generated from each file for team ti into model mi which is a RandomTreeClassifier class imported from scikit-learn 

Using this approach we can calculate the individual winning probabilities of both the teams playing a match.
This provides as a way to avoid recomputing the model each time during testing of the data. These model represent the probability of each team winning against others given a certain circumstances.

Prediction:
For each match the teams playing are individually considered for evaluation. Consider the teami and teamj are playing a match. Using the model mi and mj winning(P(W)) and loosing(P(L)) probabilities of both the teams are calculated .
Hence the overall probability of teami winning is:
(P(W)i+P(L)j)/2
and the overall probability of teamj winning is:
(P(W)j+P(L)i)/2
Thus the team with the higher chances is chosen as the winner of the match.   

Code:
Predictor.py contains the code to generate the winners of the TestMatches.csv testing set and creates the submission file.
It first creates an array of models named ‘model’ and for every match in TestMatches.csv calculates the probability of winning for both the teams ie team1 and team2 using the predict_proba() function. The team with the maximum likelihood of winning is chosen as the winner.  

Model and Hyperparameters:
The random tree generator provides a more robust and reliable way to generate the probabilities. It avoids the problem of overfitting as encountered in Decision trees like CART. Due to the greater number of trees generated it is easy to estimate the probability accurately. We have chosen the maximum depth as 14 i.e. 2 times the number of attributes for better accuracy and the number of trees =500 to avoid over fitting and better approximation. The hyperparameters were chosen on a trial and error basis and the parameters with the maximum accuracy were chosen for implementation.


Results:(Tester.py)
Taking 20% of total data as testing set and training on 80% data gives the Accuracy of 90% .
