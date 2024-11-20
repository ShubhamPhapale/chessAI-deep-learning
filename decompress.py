# Decompress the file
import zstandard as zstd

def decompress_file(input_file, output_file):
    with open(input_file, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with open(output_file, 'wb') as destination:
            dctx.copy_stream(compressed, destination)

decompress_file('data.pgn.zst', 'data.pgn')