import math


def convert_bytes(size_in_bits):
    # Converts size in bits to a human-readable format (Bytes, KB, MB, GB)
    size_in_bytes = size_in_bits / 8
    if size_in_bytes < 1024:
        return f"{size_in_bytes} Bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_in_bytes / 1024**3:.2f} GB"

def file_to_bits(file_path):
    # Reads a file and converts its content to a bit stream
    with open(file_path, "rb") as file:
        byte_data = file.read()  
    bit_stream = ''.join(format(byte, '08b') for byte in byte_data)  # Convert to bits
    return bit_stream

def optimal_n_finder(original_img_num_bits, data_add_num_bits, metadata_bits=48):
    # Calculates the optimal n (q) based on the size of the original image and additional data
    # Takes into account that not the whole image is available, but that a part is reserved for metadata
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
    # Not all n values are posible to save exactly, so to ensure we encode/decode with exactly same n, we introduce this wrapper function
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
    max_scaled = 2 ** bits
    if n>=8:
        print(f"Error: n has to be lower than 8!")
        print(f"Will autocast n to 7.9")
    n = min(n, 7.9)
    n_scale_real = (n * (max_scaled/8) )  
    n_scale_real_round = math.ceil(n_scale_real)
    
    return format(n_scale_real_round, f'0{bits}b') 

def bits_to_n(stream_b, bits = 12):
    max_scaled = 2 ** bits
    if len(stream_b) != bits:
        print(f"Error: stream of bits has size {len(stream_b)} but {bits} is expected!")
        return None
    n_scale_real = int(stream_b, 2)
    
    n = (n_scale_real * (8/max_scaled) )
    return n


def bits_to_integer(bit_stream):
    if not all(bit in '01' for bit in bit_stream):
        raise ValueError("Input must be a string of bits (only '0' and '1') -> recieved:", bit_stream)
    
    return int(bit_stream, 2)


def integer_to_bits(number, num_bits):
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
