import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
import joblib

random.seed(10)

def players(files):
    over = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        df = df.sort_values(by=["GameId", "Quarter", "Minute", "Second"], ascending=[True, True, False, False])
        df = df[df["PlayType"].isin(['RUSH', 'PASS', 'SACK', 'PUNT', 'FIELD GOAL', 'SCRAMBLE', 'FUMBLES', 'PENALTY'])]
        df.reset_index(drop=True, inplace=True)
        df['Play_Num'] = 0
        
        current_team = None
        play_counter = 0
        for index, row in df.iterrows():
            if row['OffenseTeam'] != current_team:
                current_team = row['OffenseTeam']
                play_counter = 0
            play_counter += 1
            df.at[index, 'Play_Num'] = play_counter
        over = pd.concat([over,df])
    return over

def process_series(df):

    df.reset_index(drop=True, inplace=True)
    
    series_touchdown = []
    series_interception = []
    series_fumble = []
    series_punt = []
    series_fg = []
    series_redzone = []

    current_team = None
    for index, row in df.iterrows():
        if row['OffenseTeam'] != current_team:
        
            if any(series_touchdown) or any(series_interception) or any(series_fumble) or any(series_punt) or any(series_fg) or any(series_redzone):
                df.loc[index - len(series_touchdown): index - 1, 'SeriesTouchdown'] = int(any(series_touchdown))
                df.loc[index - len(series_interception): index - 1, 'SeriesInterception'] = int(any(series_interception))
                df.loc[index - len(series_fumble): index - 1, 'SeriesFumble'] = int(any(series_fumble))
                df.loc[index - len(series_punt): index - 1, 'SeriesPunt'] = int(any(series_punt))
                df.loc[index - len(series_fg): index - 1, 'SeriesFG'] = int(any(series_fg))
                df.loc[index - len(series_redzone): index - 1, 'SeriesRedzone'] = int(any(series_redzone))

        
            series_touchdown = []
            series_interception = []
            series_fumble = []
            series_punt = []
            series_fg = []
            series_redzone = []

            current_team = row['OffenseTeam']

    
        series_touchdown.append(row['IsTouchdown'])
        series_interception.append(row['IsInterception'])
        series_fumble.append(row['IsFumble'])
        series_punt.append(row['PlayType'] == 'PUNT')
        series_fg.append(row['PlayType'] == 'FIELD GOAL')
        series_redzone.append(row['YardLine'] >= 80)


    if any(series_touchdown) or any(series_interception) or any(series_fumble) or any(series_punt) or any(series_fg) or any(series_redzone):
        df.loc[len(df) - len(series_touchdown):, 'SeriesTouchdown'] = int(any(series_touchdown))
        df.loc[len(df) - len(series_interception):, 'SeriesInterception'] = int(any(series_interception))
        df.loc[len(df) - len(series_fumble):, 'SeriesFumble'] = int(any(series_fumble))
        df.loc[len(df) - len(series_punt):, 'SeriesPunt'] = int(any(series_punt))
        df.loc[len(df) - len(series_fg):, 'SeriesFG'] = int(any(series_fg))
        df.loc[len(df) - len(series_redzone):, 'SeriesRedzone'] = int(any(series_redzone))

    return df

def cluster(df, k):
    features = ['Down', 'ToGo', 'Yards','SeriesFirstDown','IsRush', 'IsPass','SeriesTouchdown', 'SeriesInterception','SeriesFumble', 'SeriesPunt', 'SeriesFG', 'SeriesRedzone']

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=k,random_state=10) 
    clusters = kmeans.fit_predict(df_scaled)

    df['Cluster'] = clusters
    cluster_summary = df.groupby('Cluster')[['Down', 'ToGo', 'Yards',"YardLine",'SeriesFirstDown','IsRush', 'IsPass', 'IsIncomplete', 'IsTouchdown','IsSack','SeriesTouchdown', 'SeriesInterception','SeriesFumble', 'SeriesPunt', 'SeriesFG', 'SeriesRedzone']].mean()
    counts = df.groupby('Cluster')['Quarter'].count()
    counts = pd.DataFrame(counts).reset_index()
    counts.columns = ["Cluster", "NumberofPlays"]
    return cluster_summary, counts

all = players(["pbp-2014.csv","pbp-2015.csv","pbp-2016.csv","pbp-2017.csv","pbp-2018.csv","pbp-2019.csv","pbp-2020.csv","pbp-2021.csv","pbp-2022.csv","pbp-2023.csv"])
all = process_series(all).fillna(0)
all_select = all.loc[:,['Quarter', 'Minute', 'Second','Down', 'ToGo', 'YardLine','SeriesFirstDown', 'Yards', 'Formation','PlayType', 'IsRush', 'IsPass', 'IsIncomplete', 'IsTouchdown','PassType', 'IsSack','IsInterception', 'IsFumble','IsPenalty','RushDirection', 'YardLineFixed','Play_Num','SeriesTouchdown', 'SeriesInterception','SeriesFumble', 'SeriesPunt', 'SeriesFG', 'SeriesRedzone']]
    
def threshholder(down,thresh,num):
    down,counts = cluster(all_select[all_select["Down"]==down],num)
    top85 = down[down["SeriesRedzone"]>=thresh].reset_index()
    merged = pd.merge(top85,counts,on = "Cluster", how = "left")
    merged['thresehold'] = (merged["Yards"]/merged['ToGo']) * merged["NumberofPlays"]
    down_threshsold = merged['thresehold'].sum() / merged["NumberofPlays"].sum()
    return down_threshsold

first_down = threshholder(1,.92,45)
second_down = threshholder(2,.92,27)
third_down = min(threshholder(3,.92,45),1)
forth_down = max(threshholder(4,.92,45),1)


data = all.loc[:,['Quarter', 'Minute', 'Second', 'OffenseTeam','DefenseTeam', 'Down', 'ToGo','Yards','Formation','YardLine', 'IsRush', 'IsTouchdown','RushDirection']]
for x in range(0,len(data)):
    if data.loc[x,"Down"]==1:
        if data.loc[x,'IsTouchdown'] ==1 or (data.loc[x,'Yards']>= data.loc[x,'ToGo']*first_down):
            data.loc[x,"Success"] =1
        else:
            data.loc[x,"Success"] =0
    if data.loc[x,"Down"]==2:
        if data.loc[x,'IsTouchdown'] ==1 or (data.loc[x,'Yards']>= data.loc[x,'ToGo']*second_down):
            data.loc[x,"Success"] =1
        else:
            data.loc[x,"Success"] =0         
    else:
        if data.loc[x,'IsTouchdown'] ==1 or (data.loc[x,'Yards']>= data.loc[x,'ToGo']):
            data.loc[x,"Success"] =1
        else:
            data.loc[x,"Success"] =0     

offense_dumm = pd.get_dummies(data['OffenseTeam'], prefix="OFF_",dtype=bool)
data = pd.concat([data, offense_dumm], axis=1)
data.drop(columns=['OffenseTeam'], inplace=True)

defense_dumm = pd.get_dummies(data['DefenseTeam'],prefix="DEF_", dtype=bool)
data = pd.concat([data, defense_dumm], axis=1)
data.drop(columns=['DefenseTeam'], inplace=True)

Formation_dumm = pd.get_dummies(data['Formation'], dtype=bool)
data = pd.concat([data, Formation_dumm], axis=1)
data.drop(columns=['Formation'], inplace=True)

RushDirection_dumm = pd.get_dummies(data['RushDirection'], dtype=bool)
data = pd.concat([data, RushDirection_dumm], axis=1)
data.drop(columns=['RushDirection'], inplace=True)

data.drop(columns=['OFF__0','DEF__0',0,'IsTouchdown','Yards'],inplace=True)

data.columns

first = data[data["Down"]==4]
first["Success"].sum()/len(first)

X = data.drop(columns=['Success'])
y = data['Success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(
    n_estimators=175,    
    max_depth=15,        
    min_samples_split=10,
    min_samples_leaf=3,  
    max_features='sqrt', 
    bootstrap=True,      
    random_state=40      
)

rf_classifier.fit(X_train_scaled, y_train)

y_pred = rf_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)

importances = rf_classifier.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 16))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in RandomForest Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Failed Play', 'Predicted Successful Play'],
            yticklabels=['Actual Failed Play', 'Actual Successful Play'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

tn, fp, fn, tp = cm.ravel() 

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity (Recall or TPR): {sensitivity:.2f}")
print(f"Specificity (TNR): {specificity:.2f}")


probabilities = rf_classifier.predict_proba(X_test_scaled)

print("Sample probabilities for 'Success':", probabilities[:,:1])

dump(rf_classifier, 'rf_6242.joblib')


joblib.dump(scaler, 'scaler.joblib')

rf_classifier_loaded = load('rf_6242.joblib')

final_user_input = []
scaler = StandardScaler()
scaled = scaler.transform(final_user_input) 


probabilities = rf_classifier_loaded.predict_proba(scaled)
probabilities[:,:1]





















X_test["Down"] = X_test["Down"].astype('int')
X_test['Predicted_Success'] = y_pred.astype('int')
X_test["Actual_Success"] = y_test.astype('int')
X_test["C_Succ_Pred"] = (X_test['Predicted_Success'] & X_test["Actual_Success"])

X_test["C_Pred"] = (X_test['Predicted_Success'] == X_test["Actual_Success"])
X_test = X_test[X_test["Down"].isin([1,2,3,4])]


X_test.groupby(["SeasonYear","Down"])[["C_Succ_Pred","Actual_Success"]].mean()

ovr= X_test.groupby(["SeasonYear"])[["C_Pred"]].mean()


TPR = X_test.groupby(["SeasonYear"])[["C_Succ_Pred","Actual_Success"]].mean().reset_index()
TPR["TPR"] = TPR["C_Succ_Pred"]/TPR["Actual_Success"]
TPR.iloc[:,[0,-1]]
X_test.groupby("SeasonYear")[["C_Pred","Actual_Success"]].mean()



grouped_data = X_test.groupby(["SeasonYear", "Down"])[["C_Succ_Pred", "Actual_Success"]].mean().reset_index()

grouped_data['C_Succ_Pred_Percent'] = grouped_data['C_Succ_Pred'] / grouped_data['Actual_Success']

grouped_data['index1'] = range(len(grouped_data))

plt.figure(figsize=(16, 6))

green_palette = sns.color_palette("Greens", n_colors=len(grouped_data['Down'].unique()))
red_palette = sns.color_palette("Reds", n_colors=len(grouped_data['Down'].unique()))

bars = []
for i, down in enumerate(grouped_data['Down'].unique()):
    down_data = grouped_data[grouped_data['Down'] == down]
    bar = plt.bar(down_data['index1'], down_data['Actual_Success'], color=green_palette[i], label=f'Actual Success Down {down}')
    bars.append(bar)

for i, down in enumerate(grouped_data['Down'].unique()):
    down_data = grouped_data[grouped_data['Down'] == down]
    plt.bar(down_data['index1'], down_data['Actual_Success'] * down_data['C_Succ_Pred_Percent'], color=red_palette[i], label=f'Correct Prediction Down {down}')

adjusted_index = grouped_data['index1'] - 0.5
plt.xticks(adjusted_index, labels=[f"{year} Down {down}" for year, down in zip(grouped_data['SeasonYear'], grouped_data['Down'])])

plt.xticks(rotation=45)

plt.title('Percentage of Correct Predictions Within Actual Successes by Season and Down')
plt.xlabel('Season Year and Down')
plt.ylabel('Percentage')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles)) 
plt.legend(by_label.values(), by_label.keys(), title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()



data = data[data["SeasonYear"]==2020]
data = data[data["Down"].isin([2,3])]
X = data.drop(columns=['Success'])
y = data['Success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Failed Play', 'Predicted Successful Play'],
            yticklabels=['Actual Failed Play', 'Actual Successful Play'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()



years = TPR.iloc[:, 0].values 
tpr_values = TPR.iloc[:, -1].values 

plt.figure(figsize=(10, 6)) 
plt.plot(years, tpr_values, marker='o', linestyle='-', color='b') 
plt.title('True Positive Rate (TPR) Over Years') 
plt.xlabel('Year') 
plt.ylabel('TPR') 
plt.grid(True) 
plt.xticks(years) 
plt.tight_layout() 
plt.show() 
