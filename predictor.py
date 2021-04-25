### Custom definitions and classes if any ###

def predictRuns(testInput):
    prediction = 0
    ### Your Code Here ###

    import numpy as np
    import pandas as pd

   
    dataset = pd.read_csv('mo5.csv')
    datasetbkp = pd.read_csv('mo5.csv')
    inputdataset = pd.read_csv(testInput)


    input_x = inputdataset.iloc[:,:].values
    
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le1 = LabelEncoder()
    dataset['venue']= le1.fit_transform(dataset['venue'])
    dataset['batting_team']= le.fit_transform(dataset['batting_team'])
    dataset['bowling_team']= le.fit_transform(dataset['bowling_team'])


    x = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values
    y = dataset.iloc[:,-1].values
    

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=1)


    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    
    team1avgscore = np.array(datasetbkp.loc[ (datasetbkp['batting_team'] == input_x[0][2])].iloc[[-1],[-7]])[0][0]
    team2avgecoscore = np.array(datasetbkp.loc[ (datasetbkp['bowling_team'] == input_x[0][3])].iloc[[-1],[-6]])[0][0]
    team1vsteam2avgscore = np.array(datasetbkp.loc[ (datasetbkp['batting_team'] ==input_x[0][2]) & (datasetbkp['bowling_team'] == input_x[0][3])].iloc[[-1],[-5]])[0][0]
    team1venueavgscore = np.array(datasetbkp.loc[ (datasetbkp['batting_team'] == input_x[0][2]) & (datasetbkp['venue'] == input_x[0][0])].iloc[[-1],[-4]])[0][0]
    team2venueavgscore = np.array(datasetbkp.loc[ (datasetbkp['bowling_team'] == input_x[0][3]) & (datasetbkp['venue'] == input_x[0][0])].iloc[[-1],[-3]])[0][0]
    venueavgscore = np.array(datasetbkp.loc[ (datasetbkp['venue'] == input_x[0][0])].iloc[[-1],[-2]])[0][0]
        

    teamslabel = le.classes_.tolist()
    team1l = teamslabel.index(input_x[0][2])
    team2l = teamslabel.index(input_x[0][3])
    venuelabel = le1.classes_.tolist()
    venuel = venuelabel.index(input_x[0][0])
    
    
        

        
    if input_x[0][1] == 1:
        prediction_model_score = model.predict([[venuel,1,team1l,team2l,team1avgscore, team2avgecoscore, team1vsteam2avgscore,
                            team1venueavgscore, team2venueavgscore, venueavgscore]])
    else:
        prediction_model_score = model.predict([[venuel,2,team1l,team2l,team1avgscore, team2avgecoscore, team1vsteam2avgscore,
                            team1venueavgscore, team2venueavgscore, venueavgscore]])


    dataset_players = pd.read_csv('iplcrichack.csv')

   
    #batsman
    
    batsman_list = list(map(str,input_x[0][4].split(",")))

    batsman_length = len(batsman_list)

    batsman_avg=0
    
    for batsman in batsman_list:

        if batsman[0]==" ":
            batsman = batsman[1:]


        balist= list(map(str,batsman.split(" ")))
        ba = balist[-1]
       
       
        try:
            bat = np.array(dataset_players.loc[ (dataset_players['Batsman_Name'].str.contains(ba)) & (dataset_players['Team_Name'] == input_x[0][2])].iloc[[-1],[2]])[0][0]
           
            
        except:
            bat = np.array(dataset_players.loc[ (dataset_players['Batsman_Name'] == 'DNF') & (dataset_players['Team_Name'] == input_x[0][2])].iloc[[-1],[2]])[0][0]
            
            
        batsman_avg+=bat
            
            
    batsman_avg = batsman_avg//batsman_length


    #bowler
    bowler_list = list(map(str,input_x[0][5].split(",")))

    bowler_length = len(bowler_list)
    bowler_eco = 0
    bowler_list2 = []
    bowler_list2.extend(bowler_list)

    
    for i in range(0,6-bowler_length):
        bowler_list2.append(bowler_list[i%bowler_length])
    
            
    for bowler in bowler_list2:
        if bowler[0]==" ":
            bowler = bowler[1:]

        bolist= list(map(str,bowler.split(" ")))
        bo = bolist[-1]
        

        try:
            bowl = np.array(dataset_players.loc[ (dataset_players['Bowler_Name'].str.contains(bo)) & (dataset_players['Team_Name'] == input_x[0][3])].iloc[[-1],[4]])[0][0]
            
        except:
            bowl = np.array(dataset_players.loc[ (dataset_players['Bowler_Name'] == 'DNF') & (dataset_players['Team_Name'] == input_x[0][3])].iloc[[-1],[4]])[0][0]
        
           
        bowler_eco += bowl
        
            

    
    final_score = (prediction_model_score[0]+batsman_avg+bowler_eco)//3


    return int(final_score)
