import streamlit as st


def fn_take_picture():
    picture = st.camera_input("Take a picture")

    if picture:
        st.image(picture)

def fn_upload_file():
    uploaded_files = st.file_uploader("Choose an image", accept_multiple_files=True)

    for uploaded_file in uploaded_files:

        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            st.write(bytes_data)

if __name__ == "__main__":

    st.write("How do you want to input the image?")

    camera_checkbox_answer = st.checkbox('Camera feed')
    photo_upload_answer = st.checkbox('Upload a photo')

    if(camera_checkbox_answer == True):
        
        fn_take_picture()

        if(photo_upload_answer == True):
            fn_upload_file()
        
    elif(photo_upload_answer == True):
        fn_upload_file()

        if(camera_checkbox_answer == True):
            fn_take_picture()