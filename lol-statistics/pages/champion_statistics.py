import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import dataset
import numpy as np

@st.cache_data
def version_selector(match_tbl : pd.DataFrame, version : str) :
  return match_tbl[match_tbl["Patch"] == version]

@st.cache_data
def rank_selector(rank_tbl : pd.DataFrame, tier : str, filtered_match : pd.DataFrame) :
  if tier == "Any Rank" :
    return filtered_match
  
  rank = rank_tbl.loc[rank_tbl["RankName"] == tier, "RankId"].item()
  return filtered_match[filtered_match["RankFk"] == rank]

@st.cache_data
def get_full_match_details(
  match_stats_tbl : pd.DataFrame, 
  summoner_match_tbl : pd.DataFrame, 
  filtered_match : pd.DataFrame
  ) :
  
  merged_champion_matches = pd.merge(
    filtered_match,
    summoner_match_tbl,
    left_on = "MatchId",
    right_on = "MatchFk",
    how = "inner"
  )
  
  result = pd.merge(
    merged_champion_matches,
    match_stats_tbl,
    left_on = "SummonerMatchId",
    right_on = "SummonerMatchFk",
    how = "inner"
  )
  
  return result.drop_duplicates(subset="MatchId").reset_index(drop=True)

@st.cache_data
def get_pick_rate_per_champion(merged_dataset: pd.DataFrame):
  champion_counts = merged_dataset.groupby('ChampionName')['MatchFk'].count()
  total_picks = champion_counts.sum()
  
  pick_rate_df = ((champion_counts / total_picks) * 100).round(2).to_frame(name='PickRate(%)')
  pick_rate_df = pick_rate_df.reset_index()
  
  return pick_rate_df
  
@st.cache_data
def get_main_lane_per_champion(merged_dataset: pd.DataFrame):
  champion_lane_counts = merged_dataset.groupby(['ChampionName', 'Lane']).size().reset_index(name='PlayCount')
  idx = champion_lane_counts.groupby(['ChampionName'])['PlayCount'].idxmax()
  main_lane_df = champion_lane_counts.loc[idx]
  main_lane_df = main_lane_df[['ChampionName', 'Lane']].rename(columns={'Lane': 'MainLane'})
  
  return main_lane_df.reset_index(drop=True)

def convert_numeric_to_int(target) :
  numeric_cols = target.select_dtypes(include=['number']).columns
  for col in numeric_cols:
    target[col] = target[col].fillna(target.mean()).astype(int)

def scale_to_per_10minute(table, columns : list[str]) :
  for column in columns :
    try :
      table[column + '(10m)'] = table[column] / table['GameDuration'] * 60 * 10
    except :
      print("Table does not have column " + column)

@st.cache_data
def calculate_tier(final_stats_df : pd.DataFrame, features:list[str]) :
  final_stats_df['Tier'] = -1
  fig = None
  try :
    X = final_stats_df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    final_stats_df['Tier'] = kmeans.fit_predict(X_scaled) + 1
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    pca_df['Cluster'] = final_stats_df['Tier']
    
    pca_df['ChampionName'] = final_stats_df.reset_index()['ChampionName']
    
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=pca_df['Cluster'].astype(str),
        hover_data={
            'ChampionName': True, 
            'Cluster': True, 
            'PC1': ':.2f', 
            'PC2': ':.2f'
        },
        title=f'Clustering Champion Visuallization (PCA, k={k})'
    )
    
    fig.update_layout(legend_title_text='Cluster ID')
  except :
    st.error("Data scarcity")
  
  return final_stats_df[['ChampionName', 'Tier']].set_index('ChampionName'), fig

def sort_version_key(version_str):
    try:
        return tuple(map(int, version_str.split('.')))
    except ValueError:
        return version_str

def draw_win_rate_plot(champion_rate_information_tbl : pd.DataFrame, selected_champion : str, is_searched : bool) :
  df_for_highlight = champion_rate_information_tbl.copy()
  df_for_highlight['Color_Group'] = df_for_highlight['ChampionName'].apply(
    lambda x: 'Selected' if x == selected_champion else 'Others'
  )

  fig_winrate = px.bar(
    df_for_highlight.reset_index(),
    x='ChampionName',
    y='WinRate(%)',
    color='Color_Group',
    title='Win Rate %',
    text='WinRate(%)',
    color_discrete_map={'Selected': 'red', 'Others': 'gray'} if is_searched else {}
  )
  fig_winrate.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
  fig_winrate.update_layout(xaxis={'categoryorder':'total descending'})

  st.plotly_chart(fig_winrate, use_container_width=True)

def draw_pick_rate_plot(champion_rate_information_tbl : pd.DataFrame, selected_champion : str, is_searched : bool) :
  df_for_highlight = champion_rate_information_tbl.copy()
  df_for_highlight['Color_Group'] = df_for_highlight['ChampionName'].apply(
    lambda x: 'Selected' if x == selected_champion else 'Others'
  )
  
  fig_pickrate = px.bar(
    df_for_highlight.reset_index(),
    x='ChampionName',
    y='PickRate(%)',
    color='Color_Group',
    title='Pick Rate %',
    text='PickRate(%)',
    color_discrete_map={'Selected': 'red', 'Others': 'gray'} if is_searched else {}
  )
  fig_pickrate.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
  fig_pickrate.update_layout(xaxis={'categoryorder':'total descending'})
  
  st.plotly_chart(fig_pickrate, use_container_width=True)
  

def run() :
  st.header("Champion Statistics Page")
  
  # get datasets
  champion_tbl = dataset.get_dataset_by_filename("ChampionTbl.csv")
  match_tbl = dataset.get_dataset_by_filename("MatchTbl.csv")
  rank_tbl = dataset.get_dataset_by_filename("RankTbl.csv")
  match_stats_tbl = dataset.get_dataset_by_filename("MatchStatsTbl.csv")
  summoner_match_tbl = dataset.get_dataset_by_filename("SummonerMatchTbl.csv")
  
  # generate selectbox
  selected_version = st.selectbox("Version", sorted(match_tbl["Patch"].unique().tolist(), key=sort_version_key, reverse=True))
  selected_rank = st.selectbox("Rank", ["Any Rank"] + rank_tbl["RankName"].unique().tolist())
  selected_champion = st.selectbox("Champion", champion_tbl["ChampionName"].unique().tolist())
  
  # get selected data
  filtered_match = version_selector(match_tbl, selected_version)
  filtered_match = rank_selector(rank_tbl, selected_rank, filtered_match)
  
  # merge each dataset
  merged_dataset = get_full_match_details(
    match_stats_tbl, 
    summoner_match_tbl, 
    filtered_match
  )
  
  # scale data to per 10 min
  scale_to_per_10minute(merged_dataset, [
    'DmgDealt',
    'DmgTaken',
    'MinionsKilled',
    'TotalGold',
    'kills',
    'deaths',
    'assists',
    'visionScore'
  ])
  
  # convert ChampionId to ChampionName
  champion_map = champion_tbl.set_index('ChampionId')['ChampionName']
  merged_dataset['ChampionName'] = merged_dataset['ChampionFk'].map(champion_map)
  
  # searching stats
  stats = {
    "DmgDealt(10m)" : "mean",
    "DmgTaken(10m)" : "mean",
    "MinionsKilled(10m)" : "mean",
    "TotalGold(10m)" : "mean",
    "kills(10m)" : "mean",
    "deaths(10m)" : "mean",
    "assists(10m)" : "mean",
    "visionScore(10m)" : "mean",
  }
  
  champion_rate_information_tbl = pd.DataFrame()
  
  # get win rate
  champion_rate_information_tbl[['ChampionName', 'WinRate(%)']] = (merged_dataset.groupby('ChampionName', as_index=False)['Win'].mean())
  champion_rate_information_tbl['WinRate(%)'] = (champion_rate_information_tbl['WinRate(%)'] * 100).round(2)
  
  # get pick rate
  champion_rate_information_tbl = pd.merge(
      champion_rate_information_tbl,
      get_pick_rate_per_champion(merged_dataset),
      on='ChampionName',
      how='inner'
  )
  
  # get main lane
  champion_rate_information_tbl = pd.merge(
    champion_rate_information_tbl,
    get_main_lane_per_champion(merged_dataset),
    on='ChampionName',
    how='inner'
  )
  
  # get merged final champion info data
  final_stats_df = pd.merge(
    merged_dataset.groupby("ChampionName").agg(stats),
    champion_rate_information_tbl,
    on='ChampionName',
    how='inner'
  )
  
  # get tier with clustering
  features = ['WinRate(%)', 'PickRate(%)', 'DmgDealt(10m)', 'TotalGold(10m)']
  champion_tier_tbl, cluster_fig = calculate_tier(final_stats_df, features)
  
  # search champion
  searched_stats = pd.DataFrame()
  searched_information = pd.DataFrame()
  searched_tier = pd.DataFrame()
  is_searched = selected_champion != "No Champion" 
  
  if is_searched :
    try :
      # search stats
      selected_champion_details : pd.DataFrame = merged_dataset[merged_dataset['ChampionName'] == selected_champion]
      searched_stats = selected_champion_details.groupby("ChampionName").agg(stats)
      searched_information = champion_rate_information_tbl[champion_rate_information_tbl['ChampionName'] == selected_champion]
      searched_tier = champion_tier_tbl.loc[selected_champion]
    except :
      print("err in search")
    
    st.text(str(selected_champion_details.shape[0]) + " match detected")
  else :
    try :
      searched_stats = merged_dataset.groupby("ChampionName").agg(stats)
      searched_information = champion_rate_information_tbl
      searched_tier = champion_tier_tbl
    except :
      print("err in search")
    
    st.text(str(merged_dataset.shape[0]) + " match detected")
  
  # convert data float to int
  convert_numeric_to_int(searched_stats)
      
  st.divider()
  
  st.subheader("champion mean stats per 10 min")
  st.dataframe(searched_stats)
  
  st.divider()
  
  st.subheader("champion information")
  st.dataframe(searched_information.set_index('ChampionName')[['WinRate(%)', 'PickRate(%)', 'MainLane']])
  
  draw_win_rate_plot(champion_rate_information_tbl, selected_champion, is_searched)
  draw_pick_rate_plot(champion_rate_information_tbl, selected_champion, is_searched)
  
  st.divider()
  
  st.subheader("champion tier")
  st.dataframe(searched_tier)
  if(cluster_fig != None) :
    st.plotly_chart(cluster_fig, use_container_width=True)

if __name__ == "__main__" :
  run()