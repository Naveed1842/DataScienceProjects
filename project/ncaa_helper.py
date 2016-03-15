# Script file for functions that need to be imported accross multiple python notebooks
# Created By Deniz Tumer 3/13/16

# imports that we need
import pandas as pd

# Machine Learning Functions
# ----------------------------------
# This function is the main runner for converting season and tournament data
def convert_season_tourney_data(games, tourney, team_season_data):
    game_data = _convert_game_data(games, team_season_data).reset_index()
    tourney_data = _convert_game_data(tourney, team_season_data).reset_index()
    
    joined_data = game_data.append(tourney_data)
    joined_data["index"] = [i for i in range(len(joined_data))]
    joined_data.set_index("index", inplace=True)
    
    return joined_data

# This function is a helper for converting seasonal or tournament data into a single, organized, dataframe
def _convert_game_data(games, team_season_data):
    basic_cols = ["Season", "Daynum", "Numot"]
    loc_col = "Wloc"
    wp_col = "Win_Percentage"
    w_team_cols = _get_team_columns(True)
    l_team_cols = _get_team_columns(False)
    
    #grab winning and losing data
    w_team = games[w_team_cols]
    l_team = games[l_team_cols]
    l_team.columns = w_team.columns
    
    win_percent = team_season_data[["Season", "Win_Percentage"]].reset_index().set_index(["Season", "Team_Id"])
    
    # concatenate data together
    w_team_join = pd.concat([games[basic_cols], w_team], axis=1)
    w_team_join = _calc_fg_percents(w_team_join)
    w_team_join = w_team_join.merge(win_percent, left_on=["Season", "Wteam"], right_index=True)
    w_team_join["Win"] = 1
    l_team_join = pd.concat([games[basic_cols], l_team], axis=1)
    l_team_join = _calc_fg_percents(l_team_join)
    l_team_join = l_team_join.merge(win_percent, left_on=["Season", "Wteam"], right_index=True)
    l_team_join["Win"] = 0
    
    #final append of data
    joined_data = w_team_join.append(l_team_join)
    joined_data["index"] = [i for i in range(len(joined_data))]
    joined_data.set_index("index", inplace=True)
    
    return joined_data

# This function calculates the column names depending on if the team is a winning team or a losing team for a given game
def _get_team_columns(is_winning_team=False):
    team_cols = ["team", "score", "fgm", "fga", "fgm3", "fga3", "ftm", "fta", "or", "dr", "ast", "to", "stl", "blk", "pf"]
    ret_cols = []
    prefix = "L"
    
    if is_winning_team:
        prefix = "W"
    
    for col in team_cols:
        ret_cols.append(prefix + col)
    
    return ret_cols

# This function calculates the difference field goal percentages for a game
def _calc_fg_percents(joined_data):
    joined_data = _calc_fg(joined_data)
    joined_data = _calc_ft(joined_data)
    joined_data = _calc_3s(joined_data)
    
    return joined_data

# This function calculates the field goal percentage of a team for a game
def _calc_fg(joined_data):
    temp = joined_data.copy()
    temp["Wfgp"] = temp["Wfgm"] / temp["Wfga"]
    temp.drop(["Wfgm", "Wfga"], axis=1, inplace=True)
    temp.fillna(0, axis=1, inplace=True)
    
    return temp

# This function calculates the free throw percentage of a team for a game
def _calc_ft(joined_data):
    temp = joined_data.copy()
    temp["Wftp"] = temp["Wftm"] / temp["Wfta"]
    temp.drop(["Wftm", "Wfta"], axis=1, inplace=True)
    temp.fillna(0, axis=1, inplace=True)
    
    return temp

# This function calculates the 3-pointer percentage of a team for a game
def _calc_3s(joined_data):
    temp = joined_data.copy()
    temp["Wfgp3"] = temp["Wfgm3"] / temp["Wfga3"]
    temp.drop(["Wfgm3", "Wfga3"], axis=1, inplace=True)
    temp.fillna(0, axis=1, inplace=True)
    
    return temp

# Calculating Season Data Functions
# ----------------------------------
# This function is the main runner for converting team data and game data into seasonal data for a team
def calc_year_data(year, detailed_season_results, teams):
    year_data = teams.copy(True)
    
    games = _get_games(year, detailed_season_results)
    games_won = games.groupby("Wteam")
    games_lost = games.groupby("Lteam")
    
    year_data["Season"] = year
    year_data["Win_Percentage"] = _win_percent(games_won, games_lost)
    year_data["Points_Per_Game"] = _points_per_game(games_won, games_lost)
    year_data["Percent_FG_Made"] = _field_goals(games_won, games_lost)
    year_data["Percent_FT_Made"] = _free_throws(games_won, games_lost)
    year_data["Percent_3s_Made"] = _three_pointers(games_won, games_lost)
    year_data["OR_Per_Game"] = _off_rebounds(games_won, games_lost)
    year_data["DR_Per_Game"] = _def_rebounds(games_won, games_lost)
    year_data["Assists_Per_Game"] = _assists(games_won, games_lost)
    year_data["TO_Per_Game"] = _turnovers(games_won, games_lost)
    year_data["Steals_Per_Game"] = _steals(games_won, games_lost)
    year_data["Blocks_Per_Game"] = _blocks(games_won, games_lost)
    year_data["Fouls_Per_Game"] = _fouls(games_won, games_lost)
    
    # delete rows with None values (means they weren't in NCAA Division 1 that year)
    return year_data.dropna()

# This function is a helper for grabbing all games played in a year
def _get_games(year, detailed_season_results):
    return detailed_season_results[detailed_season_results.Season == year]

# This function calculates win percentage
def _win_percent(games_won, games_lost):
    num_games_won = games_won.count()["Season"]
    num_games_lost = games_lost.count()["Season"]

    return num_games_won / (num_games_won + num_games_lost)

# This function calculates points per game
def _points_per_game(games_won, games_lost):
    sum_points = 0
    
    if len(games_won) > 0:
        sum_points += games_won.sum()["Wscore"]
    
    if len(games_lost) > 0:
        sum_points += games_lost.sum()["Lscore"]
    
    return sum_points / (games_won.count()["Season"] + games_lost.count()["Season"])

# This function calculates percent field goals made
def _field_goals(games_won, games_lost):
    sum_fg_made = 0
    sum_fg_attempted = 0
    
    if len(games_won) > 0:
        sum_fg_made += games_won.sum()["Wfgm"]
        sum_fg_attempted += games_won.sum()["Wfga"]
    
    if len(games_lost) > 0:
        sum_fg_made += games_lost.sum()["Lfgm"]
        sum_fg_attempted += games_lost.sum()["Lfga"]
    
    return sum_fg_made / sum_fg_attempted

# This function calculates percent free throws made
def _free_throws(games_won, games_lost):
    sum_ft_made = 0
    sum_ft_attempted = 0
    
    if len(games_won) > 0:
        sum_ft_made += games_won.sum()["Wftm"]
        sum_ft_attempted += games_won.sum()["Wfta"]
    
    if len(games_lost) > 0:
        sum_ft_made += games_lost.sum()["Lftm"]
        sum_ft_attempted += games_lost.sum()["Lfta"]
    
    return sum_ft_made / sum_ft_attempted

# This function calculates percent three pointers made
def _three_pointers(games_won, games_lost):
    sum_3s_made = 0
    sum_3s_attempted = 0
    
    if len(games_won) > 0:
        sum_3s_made += games_won.sum()["Wfgm3"]
        sum_3s_attempted += games_won.sum()["Wfga3"]
        
    if len(games_lost) > 0:
        sum_3s_made += games_lost.sum()["Lfgm3"]
        sum_3s_attempted += games_lost.sum()["Lfga3"]
    
    return sum_3s_made / sum_3s_attempted

# This function calculates offensive rebounds per game
def _off_rebounds(games_won, games_lost):
    sum_rebounds = 0
    num_games = games_won.count()["Season"] + games_lost.count()["Season"]
    
    if len(games_won) > 0:
        sum_rebounds += games_won.sum()["Wor"]
    
    if len(games_lost) > 0:
        sum_rebounds += games_won.sum()["Lor"]
    
    return sum_rebounds / num_games

# This function calculates defensive rebounds per game
def _def_rebounds(games_won, games_lost):
    sum_rebounds = 0
    num_games = games_won.count()["Season"] + games_lost.count()["Season"]
    
    if len(games_won) > 0:
        sum_rebounds += games_won.sum()["Wdr"]
    
    if len(games_lost) > 0:
        sum_rebounds += games_won.sum()["Ldr"]
    
    return sum_rebounds / num_games

# This function calculates assists per game
def _assists(games_won, games_lost):
    sum_assists = 0
    num_games = games_won.count()["Season"] + games_lost.count()["Season"]
    
    if len(games_won) > 0:
        sum_assists += games_won.sum()["Wast"]
    
    if len(games_lost) > 0:
        sum_assists += games_lost.sum()["Last"]
    
    return sum_assists / num_games

# This function calculates turnovers per game
def _turnovers(games_won, games_lost):
    sum_turnovers = 0
    num_games = games_won.count()["Season"] + games_lost.count()["Season"]
    
    if len(games_won) > 0:
        sum_turnovers += games_won.sum()["Wto"]
    
    if len(games_lost) > 0:
        sum_turnovers += games_lost.sum()["Lto"]
    
    return sum_turnovers / num_games

# This function calculates steals per game
def _steals(games_won, games_lost):
    sum_steals = 0
    num_games = games_won.count()["Season"] + games_lost.count()["Season"]
    
    if len(games_won) > 0:
        sum_steals += games_won.sum()["Wstl"]
        
    if len(games_lost) > 0:
        sum_steals += games_won.sum()["Lstl"]
        
    return sum_steals / num_games

# This function calculates blocks per game
def _blocks(games_won, games_lost):
    sum_blocks = 0
    num_games = games_won.count()["Season"] + games_lost.count()["Season"]
    
    if len(games_won) > 0:
        sum_blocks += games_won.sum()["Wblk"]
        
    if len(games_lost) > 0:
        sum_blocks += games_lost.sum()["Lblk"]
    
    return sum_blocks / num_games

# This function calculates fouls per game
def _fouls(games_won, games_lost):
    sum_fouls = 0
    num_games = games_won.count()["Season"] + games_lost.count()["Season"]
    
    if len(games_won) > 0:
        sum_fouls += games_won.sum()["Wpf"]
        
    if len(games_lost) > 0:
        sum_fouls += games_lost.sum()["Lpf"]
    
    return sum_fouls / num_games