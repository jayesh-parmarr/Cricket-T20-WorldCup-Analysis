import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time

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

user_menu = st.sidebar.radio('Select', ('Home','Points Table','Team Wise Analysis','Head To Head Team Comparison'))



def get_point_table(year):
    st.title(f'{year} World Cup Points Table')
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
def plot_total_run_two_team_graph(team1,team2,edition,df):
    if (edition == 'Overall'):
        st.header(f'Total Runs Scored By {team1} and {team2} in Each Editions')
        plt1 = go.Scatter(y=df[df['Team']==team1]['Total Runs'], x=df[df['Team']==team1]['Year'], name = f'{team1}')
        plt2 = go.Scatter(y=df[df['Team']==team2]['Total Runs'], x=df[df['Team']==team2]['Year'], name = f'{team2}')
        layout = go.Layout(title=f'Total Runs Scored By {team1} and {team2} in Each Editions',xaxis=dict(title='Editions'), yaxis=dict(title='Total Runs Scored'))
        fig = go.Figure(data=[plt1,plt2],layout=layout)
        return fig
    else:
        ls1 = []
        ls2 = []
        ls1.append(df[(df['Team']==team1) & (df['Year']==edition)]['Runs in Wins'].values)
        ls1.append(df[(df['Team']==team1) & (df['Year']==edition)]['Runs in Loss'].values)
        lss1 = []
        ls2.append(df[(df['Team']==team2) & (df['Year']==edition)]['Runs in Wins'].values)
        ls2.append(df[(df['Team']==team2) & (df['Year']==edition)]['Runs in Loss'].values)
        lss1 = []
        lss2 = []
        for sublist in ls1:
            lss1.extend(sublist)
        for sublist in ls2:
            lss2.extend(sublist)
        if lss1 and lss2:
            st.header(f'{team1} and {team2} Score In Each Matches In {edition} T20 World Cup')
            # fig = go.Figure(data=go.Scatter(y=lss),layout =layout)
            plt1 = go.Scatter(y=lss1,name=f'{team1} Runs')
            plt2 = go.Scatter(y=lss2,name=f'{team2} Runs')
            layout = go.Layout(title=f'Total Runs Scored By {team1} and {team2} in {edition} World Cup',
            xaxis=dict(title='Matches'), yaxis=dict(title=f'Runs Scored By {team1} and {team2} In a Match')
            )
            fig = go.Figure(data=[plt1,plt2],layout=layout)
            return fig
        elif lss1:
            st.write(f'{team2} Was Not The Part Of {edition} World Cup')
            plot_total_run_graph(team1,edition,df)
        elif lss2:
            st.write(f'{team1} Was Not The Part Of {edition} World Cup')
            plot_total_run_graph(team2,edition,df)
        else:
            st.write(f'{team1} and {team2} Was Not The Part Of {edition} World Cup')

            



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
        fig.update_layout(title=f'Runs in Wins and Losses',xaxis=dict(title='Matches'),xaxis2=dict(title='Matches'),yaxis=dict(title='Runs'),yaxis2=dict(title='Runs'))
        return fig

def text_plot_bar_wins_loss(team,edition,df): 
    if edition =='Overall':
        temp_df = df[df['Team']==team].groupby('Stages').agg({'Wins':'sum','Loss':'sum'}).reset_index()      
        st.title(f"Overall Team {team} Win and Loss Count in World Cups")
        col1,col2,col3,col4 = st.columns(4)
        col2.subheader(f"Total Matches")
        col3.subheader(f"Wins")
        col4.subheader(f"Lost")

        col1,col2,col3,col4 = st.columns(4)
        col1.subheader("Group Stage")
        col2.subheader(int(temp_df[temp_df['Stages']=='Group']['Wins'].values) + int(temp_df[temp_df['Stages']=='Group']['Loss'].values))
        col3.subheader(int(temp_df[temp_df['Stages']=='Group']['Wins'].values))
        col4.subheader(int(temp_df[temp_df['Stages']=='Group']['Loss'].values))

        col1,col2,col3,col4 = st.columns(4)
        col1.subheader("Semi Finals")
        col2.subheader(int(temp_df[temp_df['Stages']=='Semi Final']['Wins'].values) + int(temp_df[temp_df['Stages']=='Semi Final']['Loss'].values))
        col3.subheader(int(temp_df[temp_df['Stages']=='Semi Final']['Wins'].values))
        col4.subheader(int(temp_df[temp_df['Stages']=='Semi Final']['Loss'].values))

        col1,col2,col3,col4 = st.columns(4)
        col1.subheader("Finals")
        col2.subheader(int(temp_df[temp_df['Stages']=='Final']['Wins'].values) + int(temp_df[temp_df['Stages']=='Final']['Loss'].values))
        col3.subheader(int(temp_df[temp_df['Stages']=='Final']['Wins'].values))
         
    else:
        temp_df = df[(df['Team']==team) & (df['Year']==edition)].groupby('Stages').agg({'Wins':'sum','Loss':'sum'}).reset_index()        
        st.title(f"Team {team} {edition} World Cup Win and Loss Count")
        col1,col2,col3,col4 = st.columns(4)
        col2.subheader(f"Total Matches")
        col3.subheader(f"Wins")
        col4.subheader(f"Lost")

        col1,col2,col3,col4 = st.columns(4)
        col1.subheader("Group Stage")
        col2.subheader(int(temp_df[temp_df['Stages']=='Group']['Wins'].values) + int(temp_df[temp_df['Stages']=='Group']['Loss'].values))
        col3.subheader(int(temp_df[temp_df['Stages']=='Group']['Wins'].values))
        col4.subheader(int(temp_df[temp_df['Stages']=='Group']['Loss'].values))

        col1,col2,col3,col4 = st.columns(4)
        col1.subheader("Semi Finals")
        col2.subheader(int(temp_df[temp_df['Stages']=='Semi Final']['Wins'].values) + int(temp_df[temp_df['Stages']=='Semi Final']['Loss'].values))
        col3.subheader(int(temp_df[temp_df['Stages']=='Semi Final']['Wins'].values))
        col4.subheader(int(temp_df[temp_df['Stages']=='Semi Final']['Loss'].values))

        col1,col2,col3,col4 = st.columns(4)
        col1.subheader("Finals")
        col2.subheader(int(temp_df[temp_df['Stages']=='Final']['Wins'].values) + int(temp_df[temp_df['Stages']=='Final']['Loss'].values))
        col3.subheader(int(temp_df[temp_df['Stages']=='Final']['Wins'].values))
        col4.subheader(int(temp_df[temp_df['Stages']=='Final']['Loss'].values))
       

def plot_bar_wins_loss(team,edition,df):
    if edition =='Overall':
        st.title(f'{team} Wins and Loss Count at different Stages in World Cup')
        temp_df = df[df['Team']==team].groupby('Stages').agg({'Wins':'sum','Loss':'sum'}).reset_index()     
        df_tidy = pd.melt(temp_df, id_vars='Stages', var_name='Outcome', value_name='Count')
        fig = px.bar(df_tidy, x='Stages', y='Count', color='Outcome',title='Wins and Losses at Different Stages',labels={'Count': 'Wins'},barmode='group')
        return fig
    else:
        st.title(f'{team} Wins and Loss Count at different Stages in {edition} World Cup')
        temp_df = df[(df['Team']==team) & (df['Year']==edition)].groupby('Stages').agg({'Wins':'sum','Loss':'sum'}).reset_index()
        
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
        if not scores_win:
            max_score_wins = -1
        else:
            max_score_wins = max(scores_win)

        if not scores_lost:
            max_score_lost = -1
        else:
            max_score_lost = max(scores_lost)
        
        if not scores_win:
            min_score_wins = float('inf')
        else:
            min_score_wins = min(scores_win)
        if not scores_lost:
            min_score_lost = float('inf')
        else:
            min_score_lost = min(scores_lost)

        max_score = max(max_score_wins, max_score_lost)
        min_score = min(min_score_wins, min_score_lost)

        # higest run chase
        high_run_chase = []
        high_run_chase.append(temp_won_chase['Winning Team Runs'].values)
        f_high_run_chase = []
        for sublist in high_run_chase:
            f_high_run_chase.extend(sublist)
        if not f_high_run_chase:
            highest_chase = -1
        else:
            highest_chase = max(f_high_run_chase)

        # lowest total defend
        low_run_defend = []
        low_run_defend.append(temp_won_bat['Winning Team Runs'].values)
        f_low_run_defend = []
        for sublist in low_run_defend:
            f_low_run_defend.extend(sublist)
        if not f_low_run_defend:
            lowest_defend = float('inf')
        else:
            lowest_defend = min(f_low_run_defend)


        # for highest total Chased
        col1, col2 = st.columns(2)
        if max_score in scores_win:
            col1.write(f'1. Highest Total By {team}')
            col1.write(temp_win[temp_win['Winning Team Runs']==max_score]['Winning Team Score'].values[0])
            checkbox_id = generate_checkbox_id()
            check = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check:
                col2.write(f"EDITION: T20 World Cup {temp_win[temp_win['Winning Team Runs']==max_score]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_win[temp_win['Winning Team Runs']==max_score]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_win[temp_win['Winning Team Runs']==max_score]['Match Between'].values[0]}")
                col2.write(f"Opposition Team Score  : {temp_win[temp_win['Winning Team Runs']==max_score]['Losing Team Score'].values[0]}")
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
                col2.write(f"Opposition Team Score  : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Winning Team Score'].values[0]}")
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
                col2.write(f"Opposition Team Score  : {temp_win[temp_win['Winning Team Runs']==min_score]['Losing Team Score'].values[0]}")
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
                col2.write(f"Opposition Team Score  : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Winning Team Score'].values[0]}")
                col2.write(f"Result : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Player Of The Match'].values[0]}")
        
        # for highest run chased
        col1, col2= st.columns(2)

        col1.write(f"3. Highest Total Chased By {team}")
        if highest_chase == -1:
            col1.write(f"{team} has not won a cricket match while chasing")
        else:
            col1.write(temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Winning Team Score'].values[0])
        
            checkbox_id = generate_checkbox_id()
            check2 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check2:
                col2.write(f"EDITION: T20 World Cup {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Match Between'].values[0]}")
                col2.write(f"Opposition Team Score  : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Losing Team Score'].values[0]}")
                col2.write(f"Result : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Player Of The Match'].values[0]}")
        
        # for lowest run defended
            
        col1, col2 = st.columns(2)
        
        col1.write(f"4. Lowest Total Defended By {team}")
        if lowest_defend == float('inf'):
            col1.write(f"{team} has not won a match while defending")
        else:
            col1.write(temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Winning Team Score'].values[0])
            checkbox_id = generate_checkbox_id()
            check3 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check3:
                col2.write(f"EDITION: T20 World Cup {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Match Between'].values[0]}")
                col2.write(f"Opposition Team Score  : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Losing Team Score'].values[0]}")
                col2.write(f"Result : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Player Of The Match'].values[0]}")
        

        # for winning perc while Batting first and while Chasing

        col1, col2 = st.columns(2)
        # col1 = winning perc while batting first

        col1.write(f"5. Win Percentage While Batting First")
        perc1 = temp_won_bat.shape[0]/(temp_won_bat.shape[0] + temp_lost_bat.shape[0])*100
        col1.subheader(f'{round(perc1,2)}%')
        
        col2.write(f'6. Win Percentage While Chasing')
        perc2 = temp_won_chase.shape[0]/(temp_won_chase.shape[0] + temp_lost_chase.shape[0])*100
        col2.subheader(f'{round(perc2,2)}%')
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
        
        if not scores_win:
            max_score_wins = -1
        else:
            max_score_wins = max(scores_win)

        if not scores_lost:
            max_score_lost = -1
        else:
            max_score_lost = max(scores_lost)
        
        if not scores_win:
            min_score_wins = float('inf')
        else:
            min_score_wins = min(scores_win)
        if not scores_lost:
            min_score_lost = float('inf')
        else:
            min_score_lost = min(scores_lost)

        max_score = max(max_score_wins, max_score_lost)
        min_score = min(min_score_wins, min_score_lost)

        # higest run chase
        high_run_chase = []
        high_run_chase.append(temp_won_chase['Winning Team Runs'].values)
        f_high_run_chase = []
        for sublist in high_run_chase:
            f_high_run_chase.extend(sublist)
        if not f_high_run_chase:
            highest_chase = -1
        else:
            highest_chase = max(f_high_run_chase)

        # lowest total defend
        low_run_defend = []
        low_run_defend.append(temp_won_bat['Winning Team Runs'].values)
        f_low_run_defend = []
        for sublist in low_run_defend:
            f_low_run_defend.extend(sublist)
        if not f_low_run_defend:
            lowest_defend = float('inf')
        else:
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
                col2.write(f"Opposition Team Score  : {temp_win[temp_win['Winning Team Runs']==max_score]['Losing Team Score'].values[0]}")
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
                col2.write(f"Opposition Team Score  : {temp_loss[temp_loss['Losing Team Runs']==max_score]['Winning Team Score'].values[0]}")
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
                col2.write(f"Opposition Team Score  : {temp_win[temp_win['Winning Team Runs']==min_score]['Losing Team Score'].values[0]}")
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
                col2.write(f"Opposition Team Score  : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Winning Team Score'].values[0]}")
                col2.write(f"Result : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_loss[temp_loss['Losing Team Runs']==min_score]['Player Of The Match'].values[0]}")
        
        # for highest run chased
        col1, col2= st.columns(2)

        col1.write(f"3. Highest Total Chased By {team} in {edition} World Cup")
        if highest_chase == -1:
            col1.write(f"{team} has not won a match while chasing in {edition} World Cup")
        else:
            col1.write(temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Winning Team Score'].values[0])
        
            checkbox_id = generate_checkbox_id()
            check2 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check2:
                col2.write(f"EDITION: T20 World Cup {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Match Between'].values[0]}")
                col2.write(f"Opposition Team Score  : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Losing Team Score'].values[0]}")
                col2.write(f"Result : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_won_chase[temp_won_chase['Winning Team Runs']==highest_chase]['Player Of The Match'].values[0]}")
        
        # for lowest run defended
            
        col1, col2 = st.columns(2)

        col1.write(f"4. Lowest Total Defended By {team} in {edition} World Cup")
        if lowest_defend == float('inf'):
            col1.write(f"{team} has not won a match while Defending a Total in {edition} World Cup")
        else:
            col1.write(temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Winning Team Score'].values[0])

            checkbox_id = generate_checkbox_id()
            check3 = col2.checkbox('Match Details',key=f'checkbox_{checkbox_id}')
            if check3:
                col2.write(f"EDITION: T20 World Cup {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Year'].values[0]} ")
                col2.write(f"Date   : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Date'].dt.date.values[0]}")
                col2.write(f"Match  : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Match Between'].values[0]}")
                col2.write(f"Opposition Team Score  : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Losing Team Score'].values[0]}")
                col2.write(f"Result : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Result'].values[0]}")
                col2.write(f"Player Of The Match : {temp_won_bat[temp_won_bat['Winning Team Runs']==lowest_defend]['Player Of The Match'].values[0]}")
        

        # for winning perc while Batting first and while Chasing

        col1, col2 = st.columns(2)
        # col1 = winning perc while batting first

        col1.write(f"5. Win Percentage While Batting First in {edition} World Cup")
        perc1 = temp_won_bat.shape[0]/(temp_won_bat.shape[0] + temp_lost_bat.shape[0])*100
        col1.subheader(f'{round(perc1,2)}%')
        
        col2.write(f'6. Win Percentage While Chasing in {edition} World Cup')
        perc2 = temp_won_chase.shape[0]/(temp_won_chase.shape[0] + temp_lost_chase.shape[0])*100
        col2.subheader(f'{round(perc2,2)}%')

def get_more_stats_two_teams(team1,team2,edition,df):
    if edition == 'Overall':
        temp_df = df[((df['Team1']==team1) & (df['Team2']==team2)) | ((df['Team1']==team2) & (df['Team2']==team1))]
        temp_df2 = temp_df.groupby(['Winner Team','Stages'])['Winning Team Runs'].count().reset_index()
        temp_df2 = temp_df2.rename(columns={'Winning Team Runs':'Wins'})

        meta_data = temp_df[['Match No', 'Date', 'Year','Match Between', 'Venue', 'Winning Team Score','Losing Team Score', 'Winner Team','Losing Team', 'Result', 'Player Of The Match','Winning Team (Bat/Chase)','Stages']].reset_index(drop=True)
        meta_data = meta_data.rename(columns={'Year':'Edition'})
        meta_data['Date'] = meta_data['Date'].dt.date
        
        

        st.header(f'World Cup Face-Off: Win Count Comparison for {team1} and {team2} in Various Stages')
        col1,col2,col3 = st.columns(3)
        col2.subheader(f'{team1}')
        col3.subheader(f'{team2}')


        col1,col2,col3 = st.columns(3)
        col1.subheader("Group Stage Won")
        gt1 = col2.subheader(temp_df2.loc[(temp_df2['Winner Team']==team1) & (temp_df2['Stages']=='Group'),'Wins'].values[0] if any((temp_df2['Winner Team']==team1) &(temp_df2['Stages']=='Group')) else 0)
        gt2 =col3.subheader(temp_df2.loc[(temp_df2['Winner Team']==team2) & (temp_df2['Stages']=='Group'),'Wins'].values[0] if any((temp_df2['Winner Team']==team2) &(temp_df2['Stages']=='Group')) else 0)
        if gt1==gt2==0:
            st.write(f"Both Teams {team1} and {team2} haven't played any Group Stages Matches in World Cup")
        col1,col2,col3 = st.columns(3)
        col1.subheader("Semi Final Won")
        st1=col2.subheader(temp_df2.loc[(temp_df2['Winner Team']==team1)&(temp_df2['Stages']=='Semi Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team1) &(temp_df2['Stages']=='Semi Final')) else 0)
        st2=col3.subheader(temp_df2.loc[(temp_df2['Winner Team']==team2)&(temp_df2['Stages']=='Semi Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team2) &(temp_df2['Stages']=='Semi Final')) else 0)
        if st1==st2==0:
            st.write(f"Both Teams {team1} and {team2} haven't played any Semi Finals against each other in World Cup")
        col1,col2,col3 = st.columns(3)
        col1.subheader("Final Won")
        ft1=col2.subheader(temp_df2.loc[(temp_df2['Winner Team']==team1)&(temp_df2['Stages']=='Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team1) &(temp_df2['Stages']=='Final')) else 0)
        ft2=col3.subheader(temp_df2.loc[(temp_df2['Winner Team']==team2)&(temp_df2['Stages']=='Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team2) &(temp_df2['Stages']=='Final')) else 0)
        if ft1==ft2==0:
            st.write(f"Both Teams {team1} and {team2} haven't played Finals against each other in World Cup")
        checkbox_id = generate_checkbox_id()
        check = st.checkbox('More Details',key=f'checkbox_{checkbox_id}')
        if check:
            if meta_data.empty:
                st.write(f"{team1} and {team2} haven't played any matches against each other in World Cup")
            else:
                st.dataframe(meta_data)
    else:
        temp_df = df[((df['Team1']==team1) & (df['Team2']==team2) & (df['Year']==edition)) | ((df['Team1']==team2) & (df['Team2']==team1) & (df['Year']==edition))]
        temp_df2 = temp_df.groupby(['Winner Team','Stages'])['Winning Team Runs'].count().reset_index()
        temp_df2 = temp_df2.rename(columns={'Winning Team Runs':'Wins'})

        meta_data = temp_df[['Match No', 'Date', 'Year','Match Between', 'Venue', 'Winning Team Score','Losing Team Score', 'Winner Team','Losing Team', 'Result', 'Player Of The Match','Winning Team (Bat/Chase)','Stages']].reset_index(drop=True)
        meta_data = meta_data.rename(columns={'Year':'Edition'})
        meta_data['Date'] = meta_data['Date'].dt.date
        
        
        st.header(f'{edition} World Cup Face-Off: Win Count Comparison for {team1} and {team2} in Various Stages')
        col1,col2,col3 = st.columns(3)
        col2.subheader(f'{team1}')
        col3.subheader(f'{team2}')


        col1,col2,col3 = st.columns(3)
        col1.subheader("Group Stage Won")
        gt1 = col2.subheader(temp_df2.loc[(temp_df2['Winner Team']==team1) & (temp_df2['Stages']=='Group'),'Wins'].values[0] if any((temp_df2['Winner Team']==team1) &(temp_df2['Stages']=='Group')) else 0)
        gt2 = col3.subheader(temp_df2.loc[(temp_df2['Winner Team']==team2) & (temp_df2['Stages']=='Group'),'Wins'].values[0] if any((temp_df2['Winner Team']==team2) &(temp_df2['Stages']=='Group')) else 0)
        if (gt1==0) and (gt2==0):
            st.write(f"Both Teams {team1} and {team2} haven't played Group Stages Matches against each other in {edition} World Cup")
        col1,col2,col3 = st.columns(3)
        col1.subheader("Semi Final Won")
        st1 = col2.subheader(temp_df2.loc[(temp_df2['Winner Team']==team1)&(temp_df2['Stages']=='Semi Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team1) &(temp_df2['Stages']=='Semi Final')) else 0)
        st2 =col3.subheader(temp_df2.loc[(temp_df2['Winner Team']==team2)&(temp_df2['Stages']=='Semi Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team2) &(temp_df2['Stages']=='Semi Final')) else 0)
        if st1==st2==0:
            st.write(f"Both Teams {team1} and {team2} haven't played Semi Finals against each other in {edition} World Cup")
        col1,col2,col3 = st.columns(3)
        col1.subheader("Final Won")
        ft1=col2.subheader(temp_df2.loc[(temp_df2['Winner Team']==team1)&(temp_df2['Stages']=='Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team1) &(temp_df2['Stages']=='Final')) else 0)
        ft2=col3.subheader(temp_df2.loc[(temp_df2['Winner Team']==team2)&(temp_df2['Stages']=='Final'),'Wins'].values[0] if any((temp_df2['Winner Team']==team2) &(temp_df2['Stages']=='Final')) else 0)
        if ft1==ft2==0:
            st.write(f"Both Teams {team1} and {team2} haven't played Fianls against each other in {edition} World Cup")
        checkbox_id = generate_checkbox_id()
        check = st.checkbox('More Details',key=f'checkbox_{checkbox_id}')
        if check:
            if meta_data.empty:
                st.write(f"{team1} and {team2} haven't played any matches against each other in {edition} World Cup")
            else:
                st.dataframe(meta_data)
                
def welcome_message():
    st.markdown(
        """
        ## Welcome to Your T20 World Cup Analysis Hub! where cricket meets data!🏏
        """
    )  
    st.write("Explore in-depth insights, team performances, and exciting stats. Dive into the world of T20 cricket with our comprehensive analysis. Let the stats tell the story of this thrilling tournament.")
    features = {
        "Points Table 🏏": "Explore the tournament's unfolding drama with our dynamic Points Table. Witness the rise and fall of teams in real-time, track net run rates, and immerse yourself in the excitement as your favorite team battles for supremacy.",
        "Team Wise Analysis 📊": "Uncover the hidden strategies and standout performances with our Team Wise Analysis. From batting brilliance to bowling mastery, dive into the stats that define each team's journey. Whether you're a stat enthusiast or a strategic thinker, this feature awaits your exploration.",
        "Head to Head Team Comparison 🤼‍♂️": "Gear up for thrilling matchups! Our Team Comparison feature lets you dive into the heart of team rivalries. Explore historical clashes, key player face-offs, and discover which team holds the upper hand. Anticipate the excitement as you prepare for the next chapter in head-to-head battles."
    }

    st.subheader("What Awaits You:")
    for feature, description in features.items():
        st.write(f"**{feature}**\n{description}")
    st.write('Let the innings begin! 🎉"')
     


def slide_effect(images,caption,iterations):
    for _ in range(iterations):
        caption_container = st.empty()
        image_container = st.empty()
        for i in range(len(images)):
            caption_container.title(caption[i])
            image_container.image(images[i], caption=caption[i], use_column_width=True)
            time.sleep(5)
            image_container.empty()
            caption_container.empty()





    


        


if user_menu == 'Home':
    winner_images = ["https://staticimg.amarujala.com/assets/images/2018/09/05/750x506/2007-wc-win_1536132009.jpeg","https://akm-img-a-in.tosshub.com/indiatoday/images/story/202110/Pakistan_2009_1200x768.jpeg?size=690:388","https://resources.pulse.icc-cricket.com/photo-resources/2021/09/16/b189824c-9670-4e39-adf8-fd555ceddbf6/GettyImages-957280818.jpg?width=845&height=509","https://i2.wp.com/www.xyj.in/wp-content/uploads/2016/02/ICC-T20-World-Cup-2012-Winner-West-Indies-Team-Image.jpg?strip=all","https://www.cricwindow.com/images/photo_gallary/icc-worldt20-2014-winner.jpg","https://resources.pulse.icc-cricket.com/photo-resources/2021/09/16/87f7e1bf-1737-4324-9238-f594494a1604/GettyImages-518896824.jpg?width=2000&height=1125","https://img1.hscicdn.com/image/upload/f_auto,t_ds_wide_w_640,q_50/lsci/db/PICTURES/CMS/330500/330570.6.jpg"]
    caption = ["2007 World Cup Winner: India","2009 World Cup Winner: Pakistan","2010 World Cup Winner: England","2012 World Cup Winner: West Indies","2014 World Cup Winner: Sri Lanka","2016 World Cup Winner: West Indies","2021 World Cup Winner: Australia"]
    st.sidebar.header('Home')
    st.title('T20 Cricket World Cup Analysis')
    welcome_message()
    start_slideshow = st.sidebar.checkbox("World Cup Winners")
    iterations = 10
    if start_slideshow:
        slide_effect(winner_images,caption,iterations)
        st.experimental_rerun()

    
        
if user_menu == 'Points Table':
    Points_Table = st.sidebar.header('Points Table')
    Points_table_year = st.sidebar.selectbox('Select Edition', ['2007','2009','2010','2012','2014','2016','2021'])
    actual_data_year = st.sidebar.checkbox("Show actual data") 
    if actual_data_year:
        st.header('Actual Data')
        st.dataframe(wc_dicts[Points_table_year])
    get_point_table(Points_table_year)

if user_menu == 'Team Wise Analysis':
    st.sidebar.header('Team Wise Analysis')
    team = st.sidebar.selectbox('Your Team',team_analysis_df['Team'].unique())
    edition = st.sidebar.selectbox('Select Year',['2007','2009','2010','2012','2014','2016','2021','Overall'])

    if edition=='Overall':
        st.title(f'Team {team} Analysis')

        st.plotly_chart(plot_total_run_graph(team,edition,team_analysis_df))

        st.plotly_chart(plot_run_in_winAndLoss_graph(team,edition,team_analysis_df))

        st.plotly_chart(plot_avg_run_graph(team,team_analysis_df))

        st.plotly_chart(plot_bar_wins_loss(team,edition,teams_wins_and_lost_count))
        data_in_text_check = st.checkbox('Textual Representation')
        if data_in_text_check:
            text_plot_bar_wins_loss(team,edition,teams_wins_and_lost_count)

        st.title('More Stats')
        get_more_stats(team,edition,main_df)
        
    else:
        if (team_analysis_df[(team_analysis_df['Team']==team) & (team_analysis_df['Year']==edition)]['Was Part Of WC'].values[0]):
            st.title(f'Team {team} Analysis')
            
            st.plotly_chart(plot_total_run_graph(team,edition,team_analysis_df))
            
            st.plotly_chart(plot_run_in_winAndLoss_graph(team,edition,team_analysis_df))
            
            st.plotly_chart(plot_bar_wins_loss(team,edition,teams_wins_and_lost_count))
            data_in_text_check = st.checkbox('Textual Representation')
            if data_in_text_check:
                text_plot_bar_wins_loss(team,edition,teams_wins_and_lost_count)

            st.title('More Stats')
            get_more_stats(team,edition,main_df)          
        else:
            st.write(f'{team} Team Was Not The Part Of {edition} T20 World Cup')

if user_menu =='Head To Head Team Comparison':
    st.sidebar.header('Head To Head Team Comparison')
    teams1 = pd.concat([main_df['Team1'],main_df['Team2']]).unique()
    team1 = st.sidebar.selectbox("Team 1",teams1)

    teams2 = [team for team in teams1 if team != team1]
    team2 = st.sidebar.selectbox("Team 2",teams2)
    
    edition = st.sidebar.selectbox('Select Year',['2007','2009','2010','2012','2014','2016','2021','Overall'])

    if edition =='Overall':
        st.title(f"{team1} and {team2} Head To Head Comparison")
        get_more_stats_two_teams(team1,team2,edition,main_df)

        st.plotly_chart(plot_total_run_two_team_graph(team1,team2,edition,team_analysis_df))

        st.plotly_chart(plot_run_in_winAndLoss_graph(team1,edition,team_analysis_df))
        st.plotly_chart(plot_run_in_winAndLoss_graph(team2,edition,team_analysis_df))

        st.plotly_chart(plot_avg_run_graph(team1,team_analysis_df))
        st.plotly_chart(plot_avg_run_graph(team2,team_analysis_df))

        st.plotly_chart(plot_bar_wins_loss(team1,edition,teams_wins_and_lost_count))
        data_in_text_check = st.checkbox(f'Textual Representation For {team1}')
        if data_in_text_check:
            text_plot_bar_wins_loss(team1,edition,teams_wins_and_lost_count)
       
        st.plotly_chart(plot_bar_wins_loss(team2,edition,teams_wins_and_lost_count))
        data_in_text_check = st.checkbox(f'Textual Representation For {team2}')
        if data_in_text_check:
            text_plot_bar_wins_loss(team2,edition,teams_wins_and_lost_count)
        
        st.title(f'More Stats For {team1}')
        get_more_stats(team1,edition,main_df)

        st.title(f'More Stats For {team2}')
        get_more_stats(team2,edition,main_df)


    else:
        if (team_analysis_df[(team_analysis_df['Team']==team1) & (team_analysis_df['Year']==edition)]['Was Part Of WC'].values[0]) and (team_analysis_df[(team_analysis_df['Team']==team2) & (team_analysis_df['Year']==edition)]['Was Part Of WC'].values[0]):
            st.title(f"{team1} and {team2} Head To Head Comparison")
            get_more_stats_two_teams(team1,team2,edition,main_df)

            st.plotly_chart(plot_total_run_two_team_graph(team1,team2,edition,team_analysis_df))
            
            st.plotly_chart(plot_run_in_winAndLoss_graph(team1,edition,team_analysis_df))
            st.plotly_chart(plot_run_in_winAndLoss_graph(team2,edition,team_analysis_df))

            
            st.plotly_chart(plot_bar_wins_loss(team1,edition,teams_wins_and_lost_count))
            data_in_text_check = st.checkbox(f'Textual Representation For {team1}')
            if data_in_text_check:
                text_plot_bar_wins_loss(team1,edition,teams_wins_and_lost_count)
            
            
            st.plotly_chart(plot_bar_wins_loss(team2,edition,teams_wins_and_lost_count))
            data_in_text_check = st.checkbox(f'Textual Representation For {team2}')
            if data_in_text_check:
                text_plot_bar_wins_loss(team2,edition,teams_wins_and_lost_count)
            
            st.title(f'More Stats For {team1}')
            get_more_stats(team1,edition,main_df)

            st.title(f'More Stats For {team2}')
            get_more_stats(team2,edition,main_df)

        elif (team_analysis_df[(team_analysis_df['Team']==team1) & (team_analysis_df['Year']==edition)]['Was Part Of WC'].values[0]):
            st.write(f'"{team2}" Was Not The Part Of {edition} T20 World Cup')
            st.write("Don't Worry :)")
            st.write(f"-- You can see Overall Analysis of {team1} and {team2}")
            st.write(f"-- Select a team which was a part of {edition} World Cup instead of {team2}")
            st.write(team_analysis_df[(team_analysis_df['Year']==edition) & (team_analysis_df['Was Part Of WC'])]['Team'].unique())
        elif (team_analysis_df[(team_analysis_df['Team']==team2) & (team_analysis_df['Year']==edition)]['Was Part Of WC'].values[0]):
            st.write(f'"{team1}" Was Not The Part Of {edition} T20 World Cup')
            st.write("Don't Worry :)")
            st.write(f"-- You can see Overall Analysis of {team1} and {team2}")
            st.write(f"-- Select a team which was a part of {edition} World Cup instead of {team1}")
            st.write(team_analysis_df[(team_analysis_df['Year']==edition) & (team_analysis_df['Was Part Of WC'])]['Team'].unique())

        else:    
            st.write(f'"{team1}" and "{team2}" Was Not The Part Of {edition} T20 World Cup')
            st.write("Don't Worry :)")
            st.write(f"-- You can see Overall Analysis of {team1} and {team2}")
            st.write(f"-- Select a team which was a part of {edition} World Cup instead of {team1} and {team2}")
            st.write(team_analysis_df[(team_analysis_df['Year']==edition) & (team_analysis_df['Was Part Of WC'])]['Team'].unique())
