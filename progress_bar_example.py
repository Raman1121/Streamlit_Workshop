import numpy as np
import time
import streamlit as st

# Example of progress bar
def progress_bar_example():
    st.write("Example of progress bar")
    my_bar = st.progress(0)
    num = 50

    for i in range(num):
        time.sleep(0.1)
        my_bar.progress( (i+1)/num )

def spinner_example():
    st.write("Example of spinner")
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')

def balloons_example():
    st.write("Loading ... ")
    time.sleep(3)
    st.write("Loading Completed ")
    st.balloons()

def snow_example():
    st.write("Loading ... ")
    time.sleep(3)
    st.write("Loading Completed ")
    st.snow()


if __name__ == "__main__":
    progress_bar_example()
    #spinner_example()
    #balloons_example()
    #snow_example()
