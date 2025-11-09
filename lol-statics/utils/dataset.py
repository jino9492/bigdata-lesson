import kagglehub
import pandas as pd
import os
import streamlit as st

@st.cache_resource
def get_path() :
  path = kagglehub.dataset_download("nathansmallcalder/lol-match-history-and-summoner-data-80k-matches")
  return path

def get_names() :
  return os.listdir(get_path())

def get_count() :
  return os.listdir(get_path()).__len__()

def get_dataset_by_id(id) :
  files = os.listdir(get_path())
  for i, file in enumerate(files) :
    if id == i :
      return pd.read_csv(get_path() + "/" + file)

def get_dataset_by_filename(file) :
  return pd.read_csv(get_path() + "/" + file)