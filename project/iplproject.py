import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

# Load data using raw string for paths
match = pd.read_csv(r"C:\VS Code\matches.csv")
delivery = pd.read_csv(r"C:\VS Code\deliveries.csv")

# Total 1st innings scores
total_score_df = delivery.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]

# Merge with match data
match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

# Standard team names
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

match_df['team1'] = match_df['team1'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})
match_df['team2'] = match_df['team2'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})

# Filter only matches between selected teams
match_df = match_df[match_df['team1'].isin(teams) & match_df['team2'].isin(teams)]

# Exclude D/L matches
match_df = match_df[match_df['dl_applied'] == 0]

# Keep needed columns
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]

# Merge delivery data
delivery_df = match_df.merge(delivery, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]

# Feature engineering
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

# Handle wickets
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1).astype(int)
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets'] = 10 - wickets

# Run rate metrics
delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']

# Binary result
delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)

# Final dataset
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'result']]
final_df = final_df.dropna()
final_df = final_df[final_df['balls_left'] != 0]
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Split features/labels
X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Pipeline with correct OneHotEncoder param
trf = ColumnTransformer([
    ('onehot', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('preprocessor', trf),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Train model
pipe.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, pipe.predict(X_test)))

# Match progression plot function
def match_progression(x_df, match_id, pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[match['ball'] == 6]  # End of each over
    
    temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)
    
    target = temp_df['total_runs_x'].values[0]
    
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)
    wickets.append(0)
    temp_df['wickets_in_over'] = (np.array(new_wickets) - np.array(wickets))[:temp_df.shape[0]]

    return temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']], target

# Example match plot
temp_df, target = match_progression(delivery_df, 74, pipe)

plt.figure(figsize=(18, 8))
plt.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], color='yellow', linewidth=3)
plt.plot(temp_df['end_of_over'], temp_df['win'], color='#00a65a', linewidth=4)
plt.plot(temp_df['end_of_over'], temp_df['lose'], color='red', linewidth=4)
plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'])
plt.title('Target - ' + str(target))
plt.xlabel("Over")
plt.ylabel("Runs / Wickets / Probability")
plt.legend(['Wickets in Over', 'Win %', 'Lose %', 'Runs in Over'])
plt.grid()
plt.show()

# Save model
pickle.dump(pipe, open('pipe.pkl', 'wb'))
