import streamlit as st
from utils import dataset

def run():
  st.header("Main Page")
  st.subheader("Related Datasets")

  for i in range(0, dataset.get_count()) :
    st.text(dataset.get_names()[i])
    df = dataset.get_dataset_by_id(i)
    st.dataframe(df)



if __name__ == "__main__" :
  run()
