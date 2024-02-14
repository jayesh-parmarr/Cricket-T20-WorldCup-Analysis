import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Pickle files

with open('wc_raw_data_dict.pkl', 'rb') as file:
    wc_dicts = pd.read_pickle(file)

with open('wc_2007_dict.pkl', 'rb') as file:
    wc_2007_dict = pd.read_pickle(file)    

with open('wc_2009_dict.pkl', 'rb') as file:
    wc_2009_dict = pd.read_pickle(file)

with open('wc_2010_dict.pkl', 'rb') as file:
    wc_2010_dict = pd.read_pickle(file)

with open('wc_2012_dict.pkl', 'rb') as file:
    wc_2012_dict = pd.read_pickle(file)

with open('wc_2014_dict.pkl', 'rb') as file:
    wc_2014_dict = pd.read_pickle(file)

with open('wc_2016_dict.pkl', 'rb') as file:
    wc_2016_dict = pd.read_pickle(file)

with open('wc_2021_dict.pkl', 'rb') as file:
    wc_2021_dict = pd.read_pickle(file)

with open('team_analysis_df.pkl','rb') as file:
    team_analysis_df = pd.read_pickle(file)

with open('main_df.pkl','rb') as file:
    main_df = pd.read_pickle(file)

with open('teams_wins_and_lost_count.pkl', 'rb') as file:
    teams_wins_and_lost_count = pd.read_pickle(file)






user_menu = st.sidebar.header('T20 Cricket WorldCup Analysis(2007-2021)')

user_menu = st.sidebar.radio('Select', ('Home','Points Table','Team Wise Analysis','Head To Head Comparison','FeedBack'))



def get_point_table(year):
    if year == '2007':
        for group, dataframe in wc_2007_dict.items():
            st.header(group)
            st.dataframe(dataframe)
    if year == '2009':
        for group, dataframe in wc_2009_dict.items():
            st.header(group)
            st.dataframe(dataframe)
    if year == '2010':
        for group, dataframe in wc_2009_dict.items():
            st.header(group)
            st.dataframe(dataframe)
    if year == '2012':
        for group, dataframe in wc_2009_dict.items():
            st.header(group)
            st.dataframe(dataframe)
    if year == '2014':
        for group, dataframe in wc_2014_dict.items():
            st.header(group)
            st.dataframe(dataframe)
    if year == '2016':
        for group, dataframe in wc_2016_dict.items():
            st.header(group)
            st.dataframe(dataframe)
    if year == '2021':
        for group, dataframe in wc_2021_dict.items():
            st.header(group)
            st.dataframe(dataframe)
    
def plot_total_run_graph(team,edition,df):
    if (edition == 'Overall'):
        st.header(f'Total Runs Scored By {team} in Each Editions')
        layout = go.Layout(title = 'Total Runs Scored in World Cup',xaxis=dict(title='Editions'), yaxis=dict(title='Total Runs Scored'))
        fig1 = go.Figure(data=go.Scatter(y=df[df['Team']==team]['Total Runs'], x=df[df['Team']==team]['Year'] ), layout =layout)
        return fig1
    else:
        ls = []
        ls.append(df[(df['Team']==team) & (df['Year']==edition)]['Runs in Wins'].values)
        ls.append(df[(df['Team']==team) & (df['Year']==edition)]['Runs in Loss'].values)
        lss = []
        for sublist in ls:
            lss.extend(sublist)
        if not lss:
            st.write(f'Was Not The Part Of {edition} World Cup')
        else:
            st.header(f'{team} Score In Each Matches In {edition} T20 World Cup')
            layout = go.Layout(
            title = f'Runs Scored in {edition} World Cup',xaxis=dict(title='Matches'), yaxis=dict(title='Runs Scored In a Match')
            )
            fig = go.Figure(data=go.Scatter(y=lss),layout =layout)
            return fig


def plot_avg_run_graph(team,df):
    st.header(f'Average Run Scored By {team} In Each Edition')
    layout = go.Layout(title = 'Average Scored in World Cup',xaxis=dict(title='Editions'), yaxis=dict(title='Average Runs Scored'))
    fig1 = go.Figure(data=go.Scatter(y=df[df['Team']==team]['Total Avg Runs'], x=df[df['Team']==team]['Year'] ), layout =layout)
    return fig1
    

    
def plot_run_in_winAndLoss_graph(team,edition,df):
    if (edition == 'Overall'):
        st.header(f"{team}'s Total Runs In Wins And Losses In An Edition")
        plt1 = go.Scatter(x=df[df['Team']==team]['Year'], y=df[df['Team']==team]['Runs in Wins'], mode='lines',name='Run Score In wins')
        plt2 = go.Scatter(x=df[df['Team']==team]['Year'], y=df[df['Team']==team]['Runs in Loss'], mode='lines',name='Run Score In loss')
        layout = go.Layout(title='Runs in Wins and Loss',xaxis=dict(title='Editions'), yaxis=dict(title='Runs Scored'))
        fig2 = go.Figure(data=[plt1,plt2],layout=layout)
        return fig2
    else:
        st.header(f"{team}'s Total Runs In Wins And Losses In {edition} T20 World Cup")
        winning_runs = []
        losing_runs = []
        winning_runs.append(df[(df['Team']==team) & (df['Year']==edition)]['Runs in Wins'].values)
        losing_runs.append(df[(df['Team']==team) & (df['Year']==edition)]['Runs in Loss'].values)
        wins = []
        loss =[]
        for sublist in winning_runs:
            wins.extend(sublist)
        for sublist in losing_runs:
            loss.extend(sublist)
    
        # ploting using plotly
        x_wins = list(range(1,len(wins)+1))
        trace_wins = go.Bar(x=x_wins, y=wins, name='Runs in Wins',marker=dict(color='blue'))
        x_loss = list(range(1,len(loss)+1))
        trace_loss = go.Bar(x=x_loss, y=loss, name='Runs in loss',marker=dict(color='red'))
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Runs in wins', 'Runs in Losses'])
        fig.add_trace(trace_wins, row=1,col=1)
        fig.add_trace(trace_loss, row=1,col=2)
        fig.update_layout(title='Runs in Wins and Losses',xaxis=dict(title='Matches'),xaxis2=dict(title='Matches'),yaxis=dict(title='Runs'),yaxis2=dict(title='Runs'))
        return fig
    

def plot_bar_wins_loss(team,edition,df,state):
    if edition =='Overall':
        temp_df = df[df['Team']==team].groupby('Stages').agg({'Wins':'sum','Loss':'sum'}).reset_index()
        if state:
            st.title(f"Team {team} Overall Analysis")
            # st.subheader(f"Group Stage Total Matches: {int(temp_df[temp_df['Stages']=='Group']['Wins'][1]) + int(temp_df[temp_df['Stages']=='Group']['Loss'][1])}")
            # st.subheader(f"Group Stage Wins: {int(temp_df[temp_df['Stages']=='Group']['Wins'][1])}")
            # st.subheader(f"Group Stage Loss: {int(temp_df[temp_df['Stages']=='Group']['Loss'][1])}")
            st.subheader("Group Stage:")
            st.subheader(f"-----Total Matches Played:{int(temp_df[temp_df['Stages']=='Group']['Wins'][1]) + int(temp_df[temp_df['Stages']=='Group']['Loss'][1])}")
            st.subheader(f"-----(Win/Lost):({int(temp_df[temp_df['Stages']=='Group']['Wins'][1])} /{int(temp_df[temp_df['Stages']=='Group']['Loss'][1])}")
            st.subheader(f"-----Semi Final          :  Total Matches:{int(temp_df[temp_df['Stages']=='Semi Final']['Wins'].values) + int(temp_df[temp_df['Stages']=='Semi Final']['Loss'].values)}Wins-{int(temp_df[temp_df['Stages']=='Semi Final']['Wins'].values)} Lost-{int(temp_df[temp_df['Stages']=='Semi Final']['Loss'].values)}")
            st.subheader(f"-----Finals              : Total Matches:{int(temp_df[temp_df['Stages']=='Final']['Wins'].values) + int(temp_df[temp_df['Stages']=='Final']['Loss'].values)} Wins-{int(temp_df[temp_df['Stages']=='Final']['Wins'].values)} Lost-{int(temp_df[temp_df['Stages']=='Final']['Loss'].values)}")
        df_tidy = pd.melt(temp_df, id_vars='Stages', var_name='Outcome', value_name='Count')
        fig = px.bar(df_tidy, x='Stages', y='Count', color='Outcome',title='Wins and Losses at Different Stages',labels={'Count': 'Wins'},barmode='group')
        return fig
    else:
        temp_df = df[(df['Team']==team) & (df['Year']==edition)].groupby('Stages').agg({'Wins':'sum','Loss':'sum'}).reset_index()
        
        if state:
            st.subheader(f"Team {team} Group Stage Total Matches: {int(temp_df[temp_df['Stages']=='Group']['Wins'][1]) + int(temp_df[temp_df['Stages']=='Group']['Loss'][1])}")
            st.subheader(f"Team {team} Group Stage Wins: {int(temp_df[temp_df['Stages']=='Group']['Wins'][1])}")
            st.subheader(f"Team {team} Group Stage Wins: {int(temp_df[temp_df['Stages']=='Group']['Loss'][1])}")
            # st.title()
        df_tidy = pd.melt(temp_df, id_vars='Stages', var_name='Outcome', value_name='Count')
        fig1 = px.bar(df_tidy, x='Stages', y='Count', color='Outcome',title='Wins and Losses at Different Stages',labels={'Count': 'Wins'},barmode='group')
        return fig1
    
checkbox_counter = 0
def generate_checkbox_id():
    global checkbox_counter 
    checkbox_counter += 1
    return checkbox_counter

def get_more_stats(team,edition,df):
    if edition == 'Overall':
        temp_win = df[df['Winner Team']==team]
        temp_loss = df[df['Losing Team']==team]
        temp_won_chase = df[(df['Winner Team']==team) & (df['Winning Team (Bat/Chase)']=='Chased')]
        temp_lost_chase = df[(df['Losing Team']==team) & (df['Winning Team (Bat/Chase)']=='Bat')]
        temp_won_bat = df[(df['Winner Team']==team) & (df['Winning Team (Bat/Chase)']=='Bat')]
        temp_lost_bat = df[(df['Losing Team']==team) & (df['Winning Team (Bat/Chase)']=='Chased')]

        runs_in_win = []
        runs_in_lost = []
        runs_in_win.append(temp_win['Winning Team Runs'].values)
        runs_in_lost.append(temp_loss['Losing Team Runs'].values)
        scores_win = []
        for sublist in runs_in_win:
            scores_win.extend(sublist)
        scores_lost = []
        for sublist in runs_in_lost:
            scores_lost.extend(sublist)

        max_score_wins = max(scores_win)
        max_score_lost = max(scores_lost)

        min_score_wins = min(scores_win)
        min_score_lost = min(scores_lost)

        max_score = max(max_score_wins, max_score_lost)
        min_score = min(min_score_wins, min_score_lost)

        # higest run chase
        high_run_chase = []
        high_run_chase.append(temp_won_chase['Winning Team Runs'].values)
        f_high_run_chase = []
        for sublist in high_run_chase:
            f_high_run_chase.extend(sublist)
        highest_chase = max(f_high_run_chase)

        # lowest total defend
        low_run_defend = []
        low_run_defend.append(temp_won_bat['Winning Team Runs'].values)
        f_low_run_defend = []
        for sublist in low_run_defend:
            f_low_run_defend.extend(sublist)
        lowest_defend = min(f_low_run_defend)


        # for highest total
        col1, col2 = st.columns(2)
        if max_score in scores_win:
            col1.write(f'1. Highest Total By {team}')
            col1.write(temp_win[temp_win['Winning Team Runs']==max_score]['Winning Team Score'].values[0])
            check = col2.checkbox('Match Details')
            if check:
                col2.write(f"EDITION: T20 World Cup {temp_win[temp_win['Winning Team Runs']==max_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_win[temp_win['Winning Team Runs']==max_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_win[temp_win['Winning Team Runs']==max_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_win[temp_win['Winning Team Runs']==max_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_win[temp_win['Winning Team Runs']==max_score]['Player Of The Match'].values[0]}")
        
        if max_score in scores_lost:
            col1.write(f'1. Highest Total By {team}')
            col1.write(temp_loss[temp_loss['Losing Team Runs']==max_score]['Losing Team Score'].values[0])

            checkbox_id = generate_checkbox_id()    
            check = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check:
                col2.write(f"EDITION: T20 World Cup {temp_loss[temp_loss['Losing Team Runs']==max_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Player Of The Match'].values[0]}")
        
        #  for lowest total
        col1, col2 = st.columns(2)
        if min_score in scores_win:
            col1.write(f'2. Lowest Total By {team}')
            col1.write(temp_win[temp_win['Winning Team Runs']==min_score]['Winning Team Score'].values[0])
            
            checkbox_id = generate_checkbox_id() 
            check1 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check1:
                col2.write(f"EDITION: T20 World Cup {temp_win[temp_win['Winning Team Runs']==min_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_win[temp_win['Winning Team Runs']==min_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_win[temp_win['Winning Team Runs']==min_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_win[temp_win['Winning Team Runs']==min_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_win[temp_win['Winning Team Runs']==min_score]['Player Of The Match'].values[0]}")
        
        if min_score in scores_lost:
            col1.write(f'2. Lowest Total By {team}')
            col1.write(temp_loss[temp_loss['Losing Team Runs']==min_score]['Losing Team Score'].values[0])
            
            checkbox_id = generate_checkbox_id()
            check1 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check1:
                col2.write(f"EDITION: T20 World Cup {temp_loss[temp_loss['Losing Team Runs']==min_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Player Of The Match'].values[0]}")
        
        # for highest run chased
        col1, col2= st.columns(2)

        col1.write(f"3. Highest Total Chased By {team}")
        col1.write(temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Winning Team Score'].values[0])
        
        checkbox_id = generate_checkbox_id()
        check2 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
        if check2:
            col2.write(f"EDITION: T20 World Cup {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Year'].values[0]} ")
            col2.write(f"Date   : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Date'].dt.date.values[0]}")
            col2.write(f"Match  : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Match Between'].values[0]}")
            col2.write(f"Result : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Result'].values[0]}")
            col2.write(f"Player Of The Match : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Player Of The Match'].values[0]}")
        
        # for lowest run defended
            
        col1, col2 = st.columns(2)
        
        col1.write(f"4. Lowest Total Defended By {team}")
        col1.write(temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Winning Team Score'].values[0])
        checkbox_id = generate_checkbox_id()
        check3 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
        if check3:
            col2.write(f"EDITION: T20 World Cup {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Year'].values[0]} ")
            col2.write(f"Date   : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Date'].dt.date.values[0]}")
            col2.write(f"Match  : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Match Between'].values[0]}")
            col2.write(f"Result : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Result'].values[0]}")
            col2.write(f"Player Of The Match : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Player Of The Match'].values[0]}")
        

        # for winning perc while Batting first and while Chasing

        col1, col2 = st.columns(2)
        # col1 = winning perc while batting first

        col1.write(f"5. Win Percentage While Batting First")
        col1.subheader(temp_won_bat.shape[0]/(temp_won_bat.shape[0] + temp_lost_bat.shape[0])*100)
        
        col2.write(f'6. Win Percentage While Chasing')
        col2.subheader(temp_won_chase.shape[0]/(temp_won_chase.shape[0] + temp_lost_chase.shape[0])*100)
    else:
        temp_win = df[(df['Winner Team']==team) & (df['Year']==edition)]
        temp_loss = df[(df['Losing Team']==team) & (df['Year']==edition)]
        temp_won_chase = df[(df['Winner Team']==team) & (df['Winning Team (Bat/Chase)']=='Chased') & (df['Year']==edition)]
        temp_lost_chase = df[(df['Losing Team']==team) & (df['Winning Team (Bat/Chase)']=='Bat') & (df['Year']==edition)]
        temp_won_bat = df[(df['Winner Team']==team) & (df['Winning Team (Bat/Chase)']=='Bat') & (df['Year']==edition)]
        temp_lost_bat = df[(df['Losing Team']==team) & (df['Winning Team (Bat/Chase)']=='Chased') & (df['Year']==edition)]

        runs_in_win = []
        runs_in_lost = []
        runs_in_win.append(temp_win['Winning Team Runs'].values)
        runs_in_lost.append(temp_loss['Losing Team Runs'].values)
        scores_win = []
        for sublist in runs_in_win:
            scores_win.extend(sublist)
        scores_lost = []
        for sublist in runs_in_lost:
            scores_lost.extend(sublist)

        max_score_wins = max(scores_win)
        max_score_lost = max(scores_lost)

        min_score_wins = min(scores_win)
        min_score_lost = min(scores_lost)

        max_score = max(max_score_wins, max_score_lost)
        min_score = min(min_score_wins, min_score_lost)

        # higest run chase
        high_run_chase = []
        high_run_chase.append(temp_won_chase['Winning Team Runs'].values)
        f_high_run_chase = []
        for sublist in high_run_chase:
            f_high_run_chase.extend(sublist)
        highest_chase = max(f_high_run_chase)

        # lowest total defend
        low_run_defend = []
        low_run_defend.append(temp_won_bat['Winning Team Runs'].values)
        f_low_run_defend = []
        for sublist in low_run_defend:
            f_low_run_defend.extend(sublist)
        lowest_defend = min(f_low_run_defend)


        # for highest total
        col1, col2 = st.columns(2)
        if max_score in scores_win:
            col1.write(f'1. Highest Total By {team} in {edition} World Cup')
            col1.write(temp_win[temp_win['Winning Team Runs']==max_score]['Winning Team Score'].values[0])
            checkbox_id = generate_checkbox_id()
            check = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check:
                col2.write(f"EDITION: T20 World Cup {temp_win[temp_win['Winning Team Runs']==max_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_win[temp_win['Winning Team Runs']==max_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_win[temp_win['Winning Team Runs']==max_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_win[temp_win['Winning Team Runs']==max_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_win[temp_win['Winning Team Runs']==max_score]['Player Of The Match'].values[0]}")
        
        if max_score in scores_lost:
            col1.write(f'1. Highest Total By {team} in {edition} World Cup')
            col1.write(temp_loss[temp_loss['Losing Team Runs']==max_score]['Losing Team Score'].values[0])

            checkbox_id = generate_checkbox_id()    
            check = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check:
                col2.write(f"EDITION: T20 World Cup {temp_loss[temp_loss['Losing Team Runs']==max_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Player Of The Match'].values[0]}")
        
        #  for lowest total
        col1, col2 = st.columns(2)
        if min_score in scores_win:
            col1.write(f'2. Lowest Total By {team} in {edition} World Cup')
            col1.write(temp_win[temp_win['Winning Team Runs']==min_score]['Winning Team Score'].values[0])
            
            checkbox_id = generate_checkbox_id()
            check1 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check1:
                col2.write(f"EDITION: T20 World Cup {temp_win[temp_win['Winning Team Runs']==min_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_win[temp_win['Winning Team Runs']==min_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_win[temp_win['Winning Team Runs']==min_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_win[temp_win['Winning Team Runs']==min_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_win[temp_win['Winning Team Runs']==min_score]['Player Of The Match'].values[0]}")
        
        if min_score in scores_lost:
            col1.write(f'2. Lowest Total By {team} in {edition} World Cup')
            col1.write(temp_loss[temp_loss['Losing Team Runs']==min_score]['Losing Team Score'].values[0])
            
            checkbox_id = generate_checkbox_id()
            check1 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}' )
            if check1:
                col2.write(f"EDITION: T20 World Cup {temp_loss[temp_loss['Losing Team Runs']==min_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Match Between'].values[0]}")
                col2.write(f"Result : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Player Of The Match'].values[0]}")
        
        # for highest run chased
        col1, col2= st.columns(2)

        col1.write(f"3. Highest Total Chased By {team} in {edition} World Cup")
        col1.write(temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Winning Team Score'].values[0])
        
        checkbox_id = generate_checkbox_id()
        check2 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
        if check2:
            col2.write(f"EDITION: T20 World Cup {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Year'].values[0]} ")
            col2.write(f"Date   : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Date'].dt.date.values[0]}")
            col2.write(f"Match  : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Match Between'].values[0]}")
            col2.write(f"Result : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Result'].values[0]}")
            col2.write(f"Player Of The Match : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Player Of The Match'].values[0]}")
        
        # for lowest run defended
            
        col1, col2 = st.columns(2)

        col1.write(f"4. Lowest Total Defended By {team} in {edition} World Cup")
        col1.write(temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Winning Team Score'].values[0])

        checkbox_id = generate_checkbox_id()
        check3 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
        if check3:
            col2.write(f"EDITION: T20 World Cup {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Year'].values[0]} ")
            col2.write(f"Date   : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Date'].dt.date.values[0]}")
            col2.write(f"Match  : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Match Between'].values[0]}")
            col2.write(f"Result : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Result'].values[0]}")
            col2.write(f"Player Of The Match : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Player Of The Match'].values[0]}")
        

        # for winning perc while Batting first and while Chasing

        col1, col2 = st.columns(2)
        # col1 = winning perc while batting first

        col1.write(f"5. Win Percentage While Batting First in {edition} World Cup")
        col1.subheader(temp_won_bat.shape[0]/(temp_won_bat.shape[0] + temp_lost_bat.shape[0])*100)
        
        col2.write(f'6. Win Percentage While Chasing in {edition} World Cup')
        col2.subheader(temp_won_chase.shape[0]/(temp_won_chase.shape[0] + temp_lost_chase.shape[0])*100)


if user_menu == 'Home':
    st.title('T20 Cricket World Cup Analysis')
        
if user_menu == 'Points Table':
    Points_Table = st.sidebar.header('Points Table')
    Points_table_year = st.sidebar.selectbox('Select Edition', ['2007','2009','2010','2012','2014','2016','2021'])
    actual_data_year = st.sidebar.checkbox("Show actual data") 
    if actual_data_year:
        st.header('Actual Data')
        st.dataframe(wc_dicts[Points_table_year])
    get_point_table(Points_table_year)

if user_menu == 'Team Wise Analysis':
    team = st.sidebar.selectbox('Your Team',team_analysis_df['Team'].unique())
    edition = st.sidebar.selectbox('Select Year',['2007','2009','2010','2012','2014','2016','2021','Overall'])

    if edition=='Overall':
        st.title(f'Team {team} Analysis')

        st.plotly_chart(plot_total_run_graph(team,edition,team_analysis_df))

        st.plotly_chart(plot_run_in_winAndLoss_graph(team,edition,team_analysis_df))

        st.plotly_chart(plot_avg_run_graph(team,team_analysis_df))

        data_in_text_check = st.checkbox('Textual Data')
        st.plotly_chart(plot_bar_wins_loss(team,edition,teams_wins_and_lost_count,data_in_text_check))

        st.title('More Stats')
        get_more_stats(team,edition,main_df)
        
    else:
        if (team_analysis_df[(team_analysis_df['Team']==team) & (team_analysis_df['Year']==edition)]['Was Part Of WC'].values[0]):
            st.title(f'Team {team} Analysis')
            
            st.plotly_chart(plot_total_run_graph(team,edition,team_analysis_df))
            
            st.plotly_chart(plot_run_in_winAndLoss_graph(team,edition,team_analysis_df))
            
            data_in_text_check = st.checkbox('Textual Data')
            st.plotly_chart(plot_bar_wins_loss(team,edition,teams_wins_and_lost_count,data_in_text_check))

            st.title('More Stats')
            get_more_stats(team,edition,main_df)          
        else:
            st.write(f'{team} Team Was Not The Part Of {edition} T20 World Cup')

if user_menu =='Head To Head Comparison':
    teams1 = pd.concat([main_df['Team1'],main_df['Team2']]).unique()
    team1 = st.sidebar.selectbox("Team 1",teams1)

    teams2 = [team for team in teams1 if team != team1]
    team2 = st.sidebar.selectbox("Team 2",teams2)
    
    edition = st.sidebar.selectbox('Select Year',['2007','2009','2010','2012','2014','2016','2021','Overall'])

    if edition =='Overall':
        st.plotly_chart(plot_total_run_graph(team1,edition,team_analysis_df))
        st.plotly_chart(plot_total_run_graph(team2,edition,team_analysis_df))

        st.plotly_chart(plot_run_in_winAndLoss_graph(team1,edition,team_analysis_df))
        st.plotly_chart(plot_run_in_winAndLoss_graph(team2,edition,team_analysis_df))

        st.plotly_chart(plot_avg_run_graph(team1,team_analysis_df))
        st.plotly_chart(plot_avg_run_graph(team2,team_analysis_df))

        data_in_text_check_team1 = st.checkbox(f'Textual Data For {team1}')
        st.plotly_chart(plot_bar_wins_loss(team1,edition,teams_wins_and_lost_count,data_in_text_check_team1))
        
        data_in_text_check_team2 = st.checkbox(f'Textual Data For {team2}')
        st.plotly_chart(plot_bar_wins_loss(team2,edition,teams_wins_and_lost_count,data_in_text_check_team2))

        
        st.title(f'More Stats For {team1}')
        get_more_stats(team1,edition,main_df)

        st.title(f'More Stats For {team2}')
        get_more_stats(team2,edition,main_df)


    else:
        if (team_analysis_df[(team_analysis_df['Team']==team1) & (team_analysis_df['Year']==edition)]['Was Part Of WC'].values[0]):
            st.plotly_chart(plot_total_run_graph(team1,edition,team_analysis_df))
            st.plotly_chart(plot_total_run_graph(team2,edition,team_analysis_df))

            data_in_text_check_team1 = st.checkbox(f'Textual Data For {team1}')
            st.plotly_chart(plot_bar_wins_loss(team1,edition,teams_wins_and_lost_count,data_in_text_check_team1))
            
            data_in_text_check_team2 = st.checkbox(f'Textual Data For {team2}')
            st.plotly_chart(plot_bar_wins_loss(team2,edition,teams_wins_and_lost_count,data_in_text_check_team2))

            
            st.title(f'More Stats For {team1}')
            get_more_stats(team1,edition,main_df)

            st.title(f'More Stats For {team2}')
            get_more_stats(team2,edition,main_df)

        else:    
            st.write(f'{team} Team Was Not The Part Of {edition} T20 World Cup')

