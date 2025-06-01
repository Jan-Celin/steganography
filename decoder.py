import math
from utils import integer_to_bits, bits_to_integer, bits_to_n, decode_extension

def decode(processed_img, verbose=False):
    """
    Function for decoding additional data from an image.
    Args:
        processed_img (numpy.ndarray): The image from which to decode data.
        verbose (bool, optional): If True, prints detailed information during decoding.
    """

    in_img = processed_img.copy()

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
                if (stream_len != -1) and (stream_cnt>=stream_len):
                    break

                if pix_ch_i < 0:  # extract metadata
                    n_curr = 4
                    q = 2 ** n_curr
                    excess = processed_img[h][w][cc]  % q
                    metadata_bitstream += integer_to_bits(excess, num_bits=n_curr)
                    processed_img[h][w][cc] = processed_img[h][w][cc] - excess


                
                else:  # extract additional data
                    if pix_ch_i == 0:
                        n_metadata = metadata_bitstream[0:12]
                        len_metadata = metadata_bitstream[12:44]
                        type_metadata = metadata_bitstream[44:]
                        n_opt_poss = bits_to_n(n_metadata, bits = 12)
                        n_real = n_opt_poss // 1
                        n_decimal = n_opt_poss % 1
                        stream_len = bits_to_integer(len_metadata)
                        ext_type = decode_extension(type_metadata)

                        if verbose:
                            print("n_real:", n_real)
                            print("n_decimal:", n_decimal)
                            print("metadata_len:", len(metadata_bitstream))
                            print("stream_len:", stream_len)

                    
                    
                    # n je decimalan
                    if n_decimal > 0:

                        # poseban pixo-kanal di je zapisano više info
                        if pix_ch_i % ( math.floor(1 / n_decimal) ) == 0:
                            n_curr = math.ceil(n_opt_poss)
                            q = 2 ** n_curr
                            excess = processed_img[h][w][cc] % q
                            add_data_bitstream += integer_to_bits(excess, num_bits=n_curr)
                            processed_img[h][w][cc] = processed_img[h][w][cc] - excess
                            stream_cnt += n_curr


                        # obican piksokanal, obicno upisivanje
                        else: 
                            n_curr = math.floor(n_opt_poss)
                            if n_curr > 0:
                                q = 2 ** n_curr
                                excess = processed_img[h][w][cc] % q
                                add_data_bitstream += integer_to_bits(excess, num_bits=n_curr)
                                processed_img[h][w][cc] = processed_img[h][w][cc] - excess
                                stream_cnt += n_curr


                    # n je prirodan broj
                    else: 
                        n_curr = n_opt_poss
                        if n_curr > 0:
                            q = 2 ** n_curr
                            excess = processed_img[h][w][cc] % q
                            add_data_bitstream += integer_to_bits(excess, num_bits=n_curr)
                            processed_img[h][w][cc] = processed_img[h][w][cc] - excess
                            stream_cnt += n_curr


                pix_ch_i += 1

    if len(add_data_bitstream) > stream_len:
        add_data_bitstream = add_data_bitstream[:stream_len]
    
    return processed_img, (add_data_bitstream, ext_type)