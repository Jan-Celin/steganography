import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import streamlit as st
from pathlib import Path

def convert_bytes(size_in_bits):
    size_in_bytes = size_in_bits / 8
    if size_in_bytes < 1024:
        return f"{size_in_bytes} Bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_in_bytes / 1024**3:.2f} GB"
# vraca memorijsko zauzece u odgovarajucem formatu

def file_to_bits(file_path):
    with open(file_path, "rb") as file:
        byte_data = file.read()  
    bit_stream = ''.join(format(byte, '08b') for byte in byte_data)  # Convert to bits
    return bit_stream
# vraca info koliko bita ima zadani file

def optimal_n_finder(original_img_num_bits, data_add_num_bits, metadata_bits=48):
    # kalkulira optimalan n (q) s obzirom na velicinu originalne slike i dodatnih podataka
    # # uzima u obzir da nije cijela slika na raspolaganju, nego se u početku dio rezervira za metadata
    n_opt = (8 * data_add_num_bits) / (original_img_num_bits - (metadata_bits/4) * 8)

    if data_add_num_bits <= 0 or original_img_num_bits <= 0:
        return None

    if n_opt > 4 and n_opt < 8:
        print("TREBA UPSCALEat x2")
        return -2  # Optional: Suggest upscaling or give warning

    if n_opt >= 8 or n_opt <= 0:
        return None
    else:
        return n_opt
    
def possible_n(old_n, bits=12): 
    # not all n values are posible to save exactly, so to ensure we encode/decode with exactly same n, we introduce this wrapper function
    max_scaled = 2 ** bits
    if old_n>=8:
        print(f"Error: n has to be lower than 8!")
        print(f"Will autocast n to 7.9")
    old_n = min(old_n, 7.9)
    n_scale_real = (old_n * (max_scaled/8) )  
    n_scale_real_round = math.ceil(n_scale_real)
    new_n = n_scale_real_round * (8/max_scaled) 
    # treba voditi i racuna da je u range [0, 8>
    return new_n

def n_to_bits(n, bits = 12):
    # this is specific converter, mapping from custom ranges
    max_scaled = 2 ** bits
    if n>=8:
        print(f"Error: n has to be lower than 8!")
        print(f"Will autocast n to 7.9")
    n = min(n, 7.9)
    n_scale_real = (n * (max_scaled/8) )  
    n_scale_real_round = math.ceil(n_scale_real)
    #print("\t", n_scale_real_round)
    return format(n_scale_real_round, f'0{bits}b') 

def bits_to_n(stream_b, bits = 12):
    # this is specific converter, mapping from custom ranges
    max_scaled = 2 ** bits
    if len(stream_b) != bits:
        print(f"Error: stream of bits has size {len(stream_b)} but {bits} is expected!")
        return None
    n_scale_real = int(stream_b, 2)
    #print("\t", n_scale_real)
    n = (n_scale_real * (8/max_scaled) )
    return n


def bits_to_integer(bit_stream):
    # general purpose use

    #bit_stream = bit_stream.strip()
    
    if not all(bit in '01' for bit in bit_stream):
        raise ValueError("Input must be a string of bits (only '0' and '1') -> recieved:", bit_stream)
    
    return int(bit_stream, 2)


def integer_to_bits(number, num_bits):
    # general purpose use

    if number < 0:
        raise ValueError("Only non-negative integers are supported.")
    if number >= 2 ** num_bits:
        raise ValueError(f"Number {number} cannot be represented in {num_bits} bits.")
    
    return format(number, f'0{num_bits}b')

# podrzane ekstenzije za dodatne podatke
EXTENSION_MAP = {
    '.txt':  '0000',
    '.png':  '0001',
    '.jpg':  '0010',
    '.jpeg': '0011',
    '.pdf':  '0100',
    '.csv':  '0101',
    '.json': '0110',
    '.mp3':  '0111',
    '.wav':  '1000',
    '.zip':  '1001',
    '.mp4':  '1010',
    '.docx': '1011',
    '.xlsx': '1100',
    '.pptx': '1101',
    '.bin':  '1110',
    '.log':  '1111',
}

def is_supported_extension(ext):
    """
    Check if the file extension is supported in the EXTENSION_MAP.
        
    Returns:
        bool: True if supported, False otherwise
    """
    return ext.lower() in EXTENSION_MAP

def encode_extension(ext):
    ext = ext.lower()
    if ext not in EXTENSION_MAP:
        raise ValueError(f"Unsupported extension: {ext}")
    return EXTENSION_MAP[ext]

def decode_extension(bits):
    if len(bits) != 4 or not all(b in '01' for b in bits):
        raise ValueError("Extension bits must be a 4-bit binary string.")
    
    inv_map = {v: k for k, v in EXTENSION_MAP.items()}
    if bits not in inv_map:
        raise ValueError(f"Unknown 4-bit extension code: {bits}")
    return inv_map[bits]


def bitstream_to_file(bitstream, extension, output_path='recovered_file'):
    # Rekonstruira additional file iz bitstreama
    # Odmah sprema u memoriju

    # Validate the extension
    EXTENSION_MAP = {
        '.txt':  '0000',
        '.png':  '0001',
        '.jpg':  '0010',
        '.jpeg': '0011',
        '.pdf':  '0100',
        '.csv':  '0101',
        '.json': '0110',
        '.mp3':  '0111',
        '.wav':  '1000',
        '.zip':  '1001',
        '.mp4':  '1010',
        '.docx': '1011',
        '.xlsx': '1100',
        '.pptx': '1101',
        '.bin':  '1110',
        '.log':  '1111',
    }

    if extension not in EXTENSION_MAP:
        raise ValueError(f"Unsupported extension '{extension}'. Supported: {list(EXTENSION_MAP.keys())}")

    # Make sure bitstream length is a multiple of 8
    if len(bitstream) % 8 != 0:
        bitstream = bitstream[:len(bitstream) - (len(bitstream) % 8)]

    # Convert bitstream to bytes
    byte_data = bytes(int(bitstream[i:i+8], 2) for i in range(0, len(bitstream), 8))

    # Save to file
    full_path = output_path + extension
    with open(full_path, 'wb') as f:
        f.write(byte_data)
    
    print(f"Recovered file saved at: {full_path}")

    return full_path

# * * * * * * * * * 
# ENKODIRANJE (main code)


# def encode (..... n=-1) -> n=-1 znaci da treba naci optimalan n, inace koristi zadani n

def encode(original_image, additional_data_path, n=-1, VERBOSE = False): # -> n=-1 znaci da treba naci optimalan n, inace koristi zadani n

    # provjera je li additional data file type podrzan!
    ext = os.path.splitext(additional_data_path)[1] 
    if not is_supported_extension(ext):
        print(f"[!] Unsupported extension: {ext}")
        print("Supported extensions are:", ', '.join(sorted(EXTENSION_MAP.keys())))
        return None

    data_mask_img = np.zeros(in_img.shape, dtype=np.uint8) # 24 bits per pixel # first 4 pixels are to store metadata (n_opt)
    bit_stream = file_to_bits(additional_data_path)    # add_data_path)

    if n == -1: # nađi optimalni n_opt_poss

        orig_img_shape = original_image.shape
        if len(orig_img_shape) == 3:
            original_img_num_bits = orig_img_shape[0] * orig_img_shape[1] * orig_img_shape[2] * 8
        else: # for b/w photos
            original_img_num_bits = orig_img_shape[0] * orig_img_shape[1] * 8

        n_opt = optimal_n_finder(original_img_num_bits, len(bit_stream))
        if n_opt == -2:
            # upscale x2 image and work with that
            print("[!] Not enough space. Automatically upscaling the image by 2x...")


            upscaled_img = cv2.resize(orig_img, (orig_img.shape[1]*2, orig_img.shape[0]*2), interpolation=cv2.INTER_LINEAR)

            orig_img = upscaled_img  # replace the original
            orig_img_shape = orig_img.shape  # update shape
            original_img_num_bits = orig_img.size * 8  # update bit count

            # Recompute after upscaling
            n_opt = optimal_n_finder(original_img_num_bits, len(bit_stream))
            if n_opt is None:
                raise ValueError("Still not enough space or invalid input after upscaling.")
        
        n_poss = possible_n(n_opt) 

    else:
        n_poss = n

    


    len_add_data = len(bit_stream)
    
    # !!
    full_add_data = n_to_bits(n=n_poss, bits=12) + integer_to_bits(len_add_data, num_bits=32) + encode_extension(ext) + bit_stream # metdata + add_data # MAPIRAJ file type
    # !!
    

    n_real = n_poss // 1
    n_decimal = n_poss % 1
    pix_ch_i = -12
    stream_pointer = 0

    if True:
        print("llen:", len_add_data)
        print("LEN FULL DATA:", len(full_add_data))
        print("metadata:", n_to_bits(n=n_poss, bits=12))
        print("n_opt_poss:", n_poss, "->", n_real, n_decimal)

        print("-------------")
        print("-------------")
        print()

    for h in range(0, in_img.shape[0]):
        for w in range(0, in_img.shape[1]):
            for cc in range(0, in_img.shape[2]):
                
                # if h>3 or w>4: ###OBRISI!!!
                #     break

                if stream_pointer >= len(full_add_data): # very important, to know when whole data is loaded
                    break

                #zapiši metapodatak (n)
                if pix_ch_i < 0:
                    n_curr = 4
                    q = 2 ** n_curr
                    bits_curr = full_add_data[stream_pointer:(stream_pointer+n_curr)]
                    stream_pointer += n_curr

                    original_image[h][w][cc] = original_image[h][w][cc] - (original_image[h][w][cc]  % q)
                    data_mask_img[h][w][cc] = bits_to_integer(bits_curr) #
                    if VERBOSE:
                        print("bits_curr:", bits_curr)
                        print("quan_img[h][w][cc]:", original_image[h][w][cc])
                        print("bits_to_integer(bits_curr):", bits_to_integer(bits_curr))
                        print("MMMMMM")
                        print("MMMMMM")
                        print("MMMMMM")
                
                    
                    
                else:
                    # n je decimalan
                    if n_decimal > 0:
                        if VERBOSE:
                            print("HEYYY", n_poss, "----", n_decimal)
                        
                        #poseban pixo-kanal di zapisujemo više info
                        if pix_ch_i % ( math.floor(1 / n_decimal) ) == 0: #ceil ili floor
                            if VERBOSE:
                                print("PoSeBAn")
                            # dodaj math.ceil(n_poss) (zadnja bita)
                            n_curr = math.ceil(n_poss)
                            q = 2 ** n_curr
                            bits_curr = full_add_data[stream_pointer:(stream_pointer+n_curr)]
                            # IF THERE IS NO ENOUGH LEFT -> add zero to end, BUT encode exact expected length
                            stream_pointer += n_curr
                            original_image[h][w][cc] = original_image[h][w][cc] - (original_image[h][w][cc]  % q)
                            data_mask_img[h][w][cc] = bits_to_integer(bits_curr) #
                            ##
                            if VERBOSE:
                                print("bits_curr:", bits_curr)
                                print("quan_img[h][w][cc]:", original_image[h][w][cc])
                                print("bits_to_integer(bits_curr):", bits_to_integer(bits_curr))
                                print("after[h][w][cc]:", data_mask_img[h][w][cc])
                                print("PPPPPP")
                                print("PPPPPP")
                                print("PPPPPP")

                        #obican piksokanal, obicno upisivanje
                        else: 
                            if VERBOSE:
                                print("OBIcaN")
                            # dodaj math.floor(n_poss) (zadnja bita)
                            n_curr = math.floor(n_poss)
                            q = 2 ** n_curr
                            bits_curr = full_add_data[stream_pointer:(stream_pointer+n_curr)]
                            # IF THERE IS NO ENOUGH LEFT -> add zero to end, BUT encode exact expected length
                            stream_pointer += n_curr
                            original_image[h][w][cc] = original_image[h][w][cc] - (original_image[h][w][cc]  % q)
                            if n_curr > 0:
                                data_mask_img[h][w][cc] = bits_to_integer(bits_curr) #
                                if VERBOSE:
                                    print("bits_curr:", bits_curr)
                                    print("quan_img[h][w][cc]:", original_image[h][w][cc])
                                    print("bits_to_integer(bits_curr):", bits_to_integer(bits_curr))
                                    print("after[h][w][cc]:", data_mask_img[h][w][cc])
                                    print("OOOOOO")
                                    print("OOOOOO")
                                    print("OOOOOO")

                    #n je prirodan broj
                    else: 
                        if VERBOSE:
                            print("ALOHA", n_poss, "----", n_decimal)
                        # svugdje isto dodaj (possible_n) zadnja bita
                        n_curr = n_poss
                        q = 2 ** n_curr
                        bits_curr = full_add_data[stream_pointer:(stream_pointer+n_curr)]
                        stream_pointer += n_curr
                        original_image[h][w][cc] = original_image[h][w][cc] - (original_image[h][w][cc]  % q)
                        if n_curr > 0:
                            data_mask_img[h][w][cc] = bits_to_integer(bits_curr) #

    
                if VERBOSE:
                    print("stream_idx:", stream_pointer)
                    print("--pozicija:", h, ",", w, ",", cc)
                    print()
                    
                pix_ch_i += 1


    return original_image + data_mask_img, data_mask_img

def decode(processed_img, VERBOSE = False):

    metadata_bitstream = ''
    add_data_bitstream = ''
    ext_type = ''
    stream_len = -1
    n_opt_poss = -1
    n_real = -1
    n_decimal = -1
    stream_cnt = 0

    pix_ch_i = -12
    

    for h in range(0, in_img.shape[0]):
        for w in range(0, in_img.shape[1]):
            for cc in range(0, in_img.shape[2]):

                # if h>3 or w>4: ###OBRISI!!!
                #     break

                # print("--pozicija:", h, ",", w, ",", cc)
                # print("before[h][w][cc]:", processed_img[h][w][cc])
                # print()

                if (stream_len != -1) and (stream_cnt>=stream_len):
                    break

                if pix_ch_i < 0: # extract metadata
                    n_curr = 4
                    q = 2 ** n_curr
                    excess = processed_img[h][w][cc]  % q
                    metadata_bitstream += integer_to_bits(excess, num_bits=n_curr)
                    processed_img[h][w][cc] = processed_img[h][w][cc] - excess # in-place


                
                else: # extract additional data
                    if pix_ch_i == 0:
                        n_metadata = metadata_bitstream[0:12]
                        len_metadata = metadata_bitstream[12:44]
                        type_metadata = metadata_bitstream[44:]
                        n_opt_poss = bits_to_n(n_metadata, bits = 12)
                        n_real = n_opt_poss // 1
                        n_decimal = n_opt_poss % 1
                        stream_len = bits_to_integer(len_metadata)
                        ext_type = decode_extension(type_metadata)

                        if VERBOSE:
                            print("n_real:", n_real)
                            print("n_decimal:", n_decimal)
                            print("metadata_len:", len(metadata_bitstream))
                            print("stream_len:", stream_len)

                    
                    
                    # n je decimalan
                    if n_decimal > 0:

                        #poseban pixo-kanal di je zapisano više info
                        if pix_ch_i % ( math.floor(1 / n_decimal) ) == 0:
                            n_curr = math.ceil(n_opt_poss)
                            q = 2 ** n_curr
                            excess = processed_img[h][w][cc] % q
                            add_data_bitstream += integer_to_bits(excess, num_bits=n_curr)
                            processed_img[h][w][cc] = processed_img[h][w][cc] - excess # in-place
                            stream_cnt += n_curr


                        #obican piksokanal, obicno upisivanje
                        else: 
                            n_curr = math.floor(n_opt_poss)
                            if n_curr > 0:
                                q = 2 ** n_curr
                                excess = processed_img[h][w][cc] % q
                                add_data_bitstream += integer_to_bits(excess, num_bits=n_curr)
                                processed_img[h][w][cc] = processed_img[h][w][cc] - excess # in-place
                                stream_cnt += n_curr


                    # n je prirodan broj
                    else: 
                        n_curr = n_opt_poss
                        if n_curr > 0:
                            q = 2 ** n_curr
                            excess = processed_img[h][w][cc] % q
                            add_data_bitstream += integer_to_bits(excess, num_bits=n_curr)
                            processed_img[h][w][cc] = processed_img[h][w][cc] - excess # in-place
                            stream_cnt += n_curr


                pix_ch_i += 1

    # if (len(add_data_bitstream) > stream_len), crop excess from the end
    if len(add_data_bitstream) > stream_len:
        add_data_bitstream = add_data_bitstream[:stream_len]
    
    return processed_img, (add_data_bitstream, ext_type)


# ============= Streamlit app ==============


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

                        # Convert bitstream to file
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