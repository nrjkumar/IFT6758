import json

import pandas as pd
import requests, sys
import streamlit as st
from sklearn import metrics
import matplotlib.pyplot as plt
#from ift6758.ift6758.client.feature_engineering import main_feature_engg
currentdir = os.path.dirname(os.path.realpath(__file__))


parentdir = os.path.dirname(currentdir)

sys.path.append(parentdir)
from features.utility import *
from features.tidy_data_features import  tidy_dataframe
from features.feature_eng  import *
#from ift6758.ift6758.client.game_client import GameClient
from client.serving_client import ServingClient
from client.game_client1 import Game_Client

if 'gc_obj' not in st.session_state:
    st.session_state.gc_obj = Game_Client()

drop_features_for_display = ['about_time_remaining', 'home_team', 'away_team', 'action_team_name',
                             'event_type_id', 'about_goal_away', 'about_goal_home']

all_imp_features = ['angle', 'distance_from_last_event', 'empty_net', 'shot_type_Wrap-around', 'y_coordinate', 'speed',
                    'distance', 'x_coordinate', 'game_period', 'shot_type_Tip-In', 'shot_type_Wrist Shot',
                    'game_seconds']

st.title("Hockey Visualization App")
with st.sidebar:
    workspace = st.text_input("Workspace", value="data-science-workspace")
    model = st.text_input("Model", value="xgboost-feature-selection-class-weights")
    version = st.text_input("Version", value="1.0.0")
    data = {"workspace": workspace, "model": model, "version": version}
    if st.button('Load Model'):
        # Load model from Comet ML here using the 3 inputs above
        response = requests.post(
            "http://serving:8080/download_registry_model",
            # "http://localhost:8080/download_registry_model",
            json=data
        )
        st.write(response.json()["message"])
    option = st.multiselect(label='Visualize model evaluation',options=['ROC curve', 'Goal rate by predicted decile', 'Cumulative goal rate by predicted decile'])
    st.subheader("Below Model Names the Milestone 3 is configured for")
    st.markdown("- xgboost-feature-selection-class-weights")
    st.markdown("- neural-network-model")
gameID = st.text_input("Game ID", value="2021020329", max_chars=10)
if gameID:
    if st.button('Ping game'):
        # call the game client here in order to get a new sample of events differnt from previous ones
        # and return the 9 parameters below
        data, message = st.session_state.gc_obj.get_live_data(game_id=gameID)
        # st.write(data.shape)
        if message != "Success":
            st.write(message)
        else:
            if len(data) < 1:
                st.write("There is no new data to displayed. Please check for another game!")
            else:
                df_input = feature_engineer(df=data)
                if len(df_input) > 1:
                    df_imp_features = df_input[all_imp_features]
                    response = requests.post(
                        "http://serving:8080/predict",
                        # "http://localhost:8080/predict",
                        json=json.loads(df_imp_features.to_json())
                    )
                    # df_input = pd.DataFrame.from_dict(response.json())
                    df_input["goal_probabilities"] = response.json()["goal_probabilities"]
                    last_row_data = df_input.iloc[-1]
                    time_left = last_row_data["about_time_remaining"]
                    sum_df = df_input.groupby(['action_team_name'])['goal_probabilities'].sum()
                    json_data_total_prob = json.loads(sum_df.to_json())
                    home_team_name = df_input["home_team"].unique().tolist()[0]
                    away_team_name = df_input["away_team"].unique().tolist()[0]
                    period = last_row_data["game_period"]
                    home_team_current_score = last_row_data["about_goal_home"]
                    away_team_current_score = last_row_data["about_goal_away"]
                    if home_team_name in json_data_total_prob:
                        home_team_sum_of_expected_goals = json_data_total_prob[home_team_name]
                    else:
                        home_team_sum_of_expected_goals = 0.0
                    if away_team_name in json_data_total_prob:
                        away_team_sum_of_expected_goals = json_data_total_prob[away_team_name]
                    else:
                        away_team_sum_of_expected_goals = 0.0
                    # Display:
                    st.subheader(str('Game #' + str(gameID) + "\:  " + str(home_team_name) + " vs\. " + str(away_team_name)))
                    st.subheader(str('Period: ' + str(period) + "   -   " + str(time_left) + " minutes left"))
                    col1, col2 = st.columns(2)
                    col1.metric(label=str(str(home_team_name) + " xG (actual)"), value=str(
                        str(round(home_team_sum_of_expected_goals, 1)) + " (" + str(home_team_current_score) + ")"),
                                delta=round(home_team_sum_of_expected_goals - home_team_current_score, 1))
                    col2.metric(label=str(str(away_team_name) + " xG (actual)"), value=str(
                        str(round(away_team_sum_of_expected_goals, 1)) + " (" + str(away_team_current_score) + ")"),
                                delta=round(away_team_sum_of_expected_goals - away_team_current_score, 1))
                    st.header("Data and Predictions")

                    df_input = df_input.drop(drop_features_for_display, axis=1)
                    st.write(df_input)
                else:
                    st.write("No relevant data retrieved during this instance, please refresh and retry later !!")

        # Bonus: display graphs - the input is the dataframe "df_input"
        if 'df_input' in locals():
            if option.count('ROC curve')>0:
                st.subheader('ROC curve:')
                st.write("This curve displays the performance of a the model at all classification thresholds. The Area Under the Curve (AUC) reflects the overall performance. A random model would have an AUC=0.5; the better the model, the more its AUC will be close to 1.")
                fpr, tpr, thresholds = metrics.roc_curve(df_input[['is_goal']], df_input[['goal_probabilities']])
                roc_auc = metrics.auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
                plt.legend(loc = 'lower right')
                plt.plot([0, 1], [0, 1],'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                st.pyplot(fig)
            if option.count('Goal rate by predicted decile')>0:
                st.subheader('Goal rate by predicted decile:')
                st.write("The observations are binned into deciles, and for each, we compute the proportion of goals "
                        "(rate). Note that lower deciles are those with the highest probability of goal. "
                        "If the model performs well, there should be a relationship such that the lower the decile,"
                        " the higher the goal rate.")
                fig, ax = plt.subplots(1, 1)
                goal_prob = df_input[['goal_probabilities']]
                df = df_input[['is_goal']]
                df["probablity_of_goal"] = goal_prob
                df['decile_of_goal'] = round(df["probablity_of_goal"].rank(pct = True)*10)
                goal_rate = round((df.groupby(by='decile_of_goal').sum() / df.groupby(by='decile_of_goal').count())*10)
                goal_rate['decile'] = goal_rate.index
                ax.plot(goal_rate["decile"], goal_rate["is_goal"])
                st.pyplot(fig)
            if option.count('Cumulative goal rate by predicted decile')>0:
                st.subheader('Cumulative goal rate by predicted decile:')
                st.write("This graph reflects the proportion of goals for all events which were predicted to be below or at a given decile. If the model performs well, there should be a relationship such that the lower the decile, the higher the goal rate.")    
                fig, ax = plt.subplots(1, 1)
                goal_prob = df_input[['goal_probabilities']]
                df = df_input[['is_goal']]
                df["probablity_of_goal"] = goal_prob
                df['decile_of_goal'] = round(df["probablity_of_goal"].rank(pct = True)*10)
                goal_rate = round((df.groupby(by='decile_of_goal').sum() / df.groupby(by='decile_of_goal').count())*10)
                goal_rate['decile'] = goal_rate.index
                goal_rate['cum_sum'] = goal_rate.loc[::-1, 'is_goal'].cumsum()[::-1]
                goal_rate['cum_perc'] = 10*goal_rate['cum_sum'] / goal_rate['is_goal'].sum()
                ax.plot(goal_rate["decile"], goal_rate["cum_perc"])
                st.pyplot(fig)


