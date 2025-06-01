import cv2
import streamlit as st
from pathlib import Path
from utils import file_to_bits, bitstream_to_file
from encoder import encode, decode

st.title("Steganography Demo")

st.write("This application demonstrates basic steganography techniques for hiding data in images.")

nav_options = ["Encode Data", "Decode Data"]
selected_option = st.sidebar.selectbox("Choose an option", nav_options)
if selected_option == "Encode Data":
    st.header("Encode Data into Image")
    st.write("Upload an image and a file to hide within the image.") 

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image file")

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            image_extension = Path(uploaded_image.name).suffix
            
            with open(f"uploaded_image{image_extension}", "wb") as f:
                f.write(uploaded_image.getbuffer())

            st.success("Image uploaded successfully!")

            st.session_state.uploaded_image_path = f"uploaded_image{image_extension}"

    with col2:
        st.header("Upload File")
        
        uploaded_data = st.file_uploader("Choose a file to hide")

        # Or allow text input
        if uploaded_data is None:
            uploaded_data = st.text_input("Or enter the text here", "")

        if uploaded_data is not None:

            if isinstance(uploaded_data, str):
                st.write(f"Text inputted successfully!")
                # If text input, save it as a temporary file
                with open("uploaded_data.txt", "w") as f:
                    f.write(uploaded_data)
                uploaded_data = "uploaded_data.txt"
                st.session_state.uploaded_data_path = uploaded_data

            else:
                st.write(f"File {uploaded_data.name} uploaded successfully!")
                data_extension = Path(uploaded_data.name).suffix

                with open(f"uploaded_data{data_extension}", "wb") as f:
                    f.write(uploaded_data.getbuffer())

                st.success("Data file uploaded successfully!")

                data_bits = file_to_bits(f"uploaded_data{data_extension}")
                st.write(f"Data size in bits: {len(data_bits)} bits")

                st.session_state.uploaded_data_path = f"uploaded_data{data_extension}"

    st.markdown("---")

    if uploaded_image is not None and uploaded_data is not None:
        if st.button("Encode"):
            try:
                in_img = cv2.imread(st.session_state.uploaded_image_path)
                data = st.session_state.uploaded_data_path
                encoded_image, mask = encode(in_img, data, n=-1, VERBOSE=False)

                cv2.imwrite("encoded_image.png", encoded_image)
                st.image("encoded_image.png", caption="Encoded Image", width=300)

                st.success("Data successfully encoded!")

                with open("encoded_image.png", "rb") as f:
                    st.download_button(
                        label="Download Encoded Image",
                        data=f,
                        file_name="encoded_image.png",
                        mime="image/png"
                    )

            except Exception as e:
                st.error(f"Error during encoding: {e}")

else:
    st.header("Decode Data from Image")
    st.write("Upload an image to extract hidden data from it.")

    uploaded_image = st.file_uploader("Choose an image file for decoding", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", width=300)

        with open("uploaded_image.png", "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.success("Image uploaded successfully!")

        st.markdown("---")

        if st.button("Decode"):
            try:
                in_img = cv2.imread("uploaded_image.png")
                if in_img is None:
                    st.error("Failed to read the image. Please check the file format.")
                else:
                    decoded_data, (add_data_bitstream, extension) = decode(in_img, VERBOSE=False)
                    if add_data_bitstream:
                        st.write(f"Extracted data size: {len(add_data_bitstream)} bits")
                        st.write(f"File type: {extension}")

                        output_path = bitstream_to_file(add_data_bitstream, extension, output_path='recovered_file')

                        st.success(f"Data successfully extracted and saved as {output_path}!")
                        st.download_button(
                            label="Download Recovered File",
                            data=open(output_path, "rb").read(),
                            file_name=f"recovered_file{extension}",
                            mime="application/octet-stream"
                        )
                    else:
                        st.warning("No hidden data found in the image.")

                    st.success("Data successfully decoded!")
            except Exception as e:
                st.error(f"Error during decoding: {e}")
