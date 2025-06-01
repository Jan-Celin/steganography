import cv2
import os
import numpy as np
import math
from utils import file_to_bits, optimal_n_finder, possible_n, n_to_bits, bits_to_integer, bits_to_n, integer_to_bits, is_supported_extension, encode_extension, decode_extension

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


def encode(original_image, additional_data_path, n=-1, verbose=False): # -> n=-1 znaci da treba naci optimalan n, inace koristi zadani n
    """
    Function for encoding additional data into an image.
    Args:
        original_image (numpy.ndarray): The original image to encode data into.
        additional_data_path (str): Path to the file containing additional data to encode.
        n (int, optional): The number of bits per pixel to use for encoding. If -1, finds optimal n.
        VERBOSE (bool, optional): If True, prints detailed information during encoding.
    """
    
    in_img = original_image.copy()

    # provjera je li additional data file type podrzan!
    ext = os.path.splitext(additional_data_path)[1] 
    if not is_supported_extension(ext):
        print(f"[!] Unsupported extension: {ext}")
        print("Supported extensions are:", ', '.join(sorted(EXTENSION_MAP.keys())))
        return None

    data_mask_img = np.zeros(in_img.shape, dtype=np.uint8)  # 24 bits per pixel # first 4 pixels are to store metadata (n_opt)
    bit_stream = file_to_bits(additional_data_path)

    if n == -1:  # nađi optimalni n_opt_poss

        orig_img_shape = original_image.shape
        if len(orig_img_shape) == 3:
            original_img_num_bits = orig_img_shape[0] * orig_img_shape[1] * orig_img_shape[2] * 8
        else:  # for b/w photos
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
    
    full_add_data = n_to_bits(n=n_poss, bits=12) + integer_to_bits(len_add_data, num_bits=32) + encode_extension(ext) + bit_stream  # metadata + add_data # MAPIRAJ file type

    n_real = n_poss // 1
    n_decimal = n_poss % 1
    pix_ch_i = -12
    stream_pointer = 0

    if verbose:
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
                if stream_pointer >= len(full_add_data): # very important, to know when whole data is loaded
                    break

                # zapiši metapodatak (n)
                if pix_ch_i < 0:
                    n_curr = 4
                    q = 2 ** n_curr
                    bits_curr = full_add_data[stream_pointer:(stream_pointer+n_curr)]
                    stream_pointer += n_curr

                    original_image[h][w][cc] = original_image[h][w][cc] - (original_image[h][w][cc]  % q)
                    data_mask_img[h][w][cc] = bits_to_integer(bits_curr) #
                    if verbose:
                        print("bits_curr:", bits_curr)
                        print("quan_img[h][w][cc]:", original_image[h][w][cc])
                        print("bits_to_integer(bits_curr):", bits_to_integer(bits_curr))
                        print("MMMMMM")
                        print("MMMMMM")
                        print("MMMMMM")
                
                    
                    
                else:
                    # n je decimalan
                    if n_decimal > 0:
                        if verbose:
                            print("HEYYY", n_poss, "----", n_decimal)
                        
                        # poseban pixo-kanal di zapisujemo više info
                        if pix_ch_i % ( math.floor(1 / n_decimal) ) == 0: # ceil ili floor
                            if verbose:
                                print("PoSeBAn")
                            # dodaj math.ceil(n_poss) (zadnja bita)
                            n_curr = math.ceil(n_poss)
                            q = 2 ** n_curr
                            bits_curr = full_add_data[stream_pointer:(stream_pointer+n_curr)]
                            # IF THERE IS NO ENOUGH LEFT -> add zero to end, BUT encode exact expected length
                            stream_pointer += n_curr
                            original_image[h][w][cc] = original_image[h][w][cc] - (original_image[h][w][cc]  % q)
                            data_mask_img[h][w][cc] = bits_to_integer(bits_curr)
                            
                            if verbose:
                                print("bits_curr:", bits_curr)
                                print("quan_img[h][w][cc]:", original_image[h][w][cc])
                                print("bits_to_integer(bits_curr):", bits_to_integer(bits_curr))
                                print("after[h][w][cc]:", data_mask_img[h][w][cc])
                                print("PPPPPP")
                                print("PPPPPP")
                                print("PPPPPP")

                        # obican piksokanal, obicno upisivanje
                        else: 
                            if verbose:
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
                                if verbose:
                                    print("bits_curr:", bits_curr)
                                    print("quan_img[h][w][cc]:", original_image[h][w][cc])
                                    print("bits_to_integer(bits_curr):", bits_to_integer(bits_curr))
                                    print("after[h][w][cc]:", data_mask_img[h][w][cc])
                                    print("OOOOOO")
                                    print("OOOOOO")
                                    print("OOOOOO")

                    # n je prirodan broj
                    else: 
                        if verbose:
                            print("ALOHA", n_poss, "----", n_decimal)
                        # svugdje isto dodaj (possible_n) zadnja bita
                        n_curr = n_poss
                        q = 2 ** n_curr
                        bits_curr = full_add_data[stream_pointer:(stream_pointer+n_curr)]
                        stream_pointer += n_curr
                        original_image[h][w][cc] = original_image[h][w][cc] - (original_image[h][w][cc]  % q)
                        if n_curr > 0:
                            data_mask_img[h][w][cc] = bits_to_integer(bits_curr) #

    
                if verbose:
                    print("stream_idx:", stream_pointer)
                    print("--pozicija:", h, ",", w, ",", cc)
                    print()
                    
                pix_ch_i += 1


    return original_image + data_mask_img, data_mask_img
