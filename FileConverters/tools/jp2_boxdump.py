#!/usr/bin/env python3
"""
jp2_boxdump.py

Print the box structure of a JPEG2000 file (.jp2 / .jpx / raw .j2k): every
box with its offset and length, the text of XML/XMP boxes, and the main
codestream header (SIZ, COD, QCD, COM, ...). Used to compare files written
by tifs_to_jp2.py with the ones NeuroInfo / Microfile+ writes.

Only headers are read: the codestream body is skipped, so this is fast even
on large files. Note that opening a OneDrive online-only file still makes
OneDrive download all of it.

    python jp2_boxdump.py "section 33-rmtg.jp2"
    python jp2_boxdump.py --full-xmp "section 33-rmtg.jp2"
"""

import argparse
import os
import struct
import sys

# Boxes that contain other boxes
SUPERBOXES = {b"jp2h", b"res ", b"uinf", b"jpch", b"jplh", b"cgrp", b"ftbl",
              b"comp", b"asoc", b"drep", b"page"}
TEXT_BOXES = {b"xml ", b"lbl ", b"uuid"}
MARKER_NAMES = {
    0xFF50: "CAP", 0xFF51: "SIZ", 0xFF52: "COD", 0xFF53: "COC", 0xFF55: "TLM",
    0xFF5C: "QCD", 0xFF5D: "QCC", 0xFF5F: "POC", 0xFF60: "PPM", 0xFF63: "CRG",
    0xFF64: "COM", 0xFF74: "MCT", 0xFF75: "MCC", 0xFF77: "MCO", 0xFF78: "CPF",
}
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")
MAX_BOX_READ = 1 << 20  # don't load boxes bigger than this (other than to skip them)


def dump_codestream_header(f, indent, out):
    """Print main-header marker segments, stopping at the first tile (SOT)."""
    data = f.read(65536)
    if data[:2] != b"\xff\x4f":
        out(f"{indent}   (no SOC marker - not a codestream)")
        return
    pos = 2
    while pos + 4 <= len(data):
        marker = struct.unpack(">H", data[pos:pos + 2])[0]
        if marker == 0xFF90:  # SOT
            break
        seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        seg = data[pos + 4:pos + 2 + seg_len]
        name = MARKER_NAMES.get(marker, hex(marker))
        if name == "SIZ":
            rsiz, xs, ys, xo, yo, xt, yt, xto, yto, n = struct.unpack(">HIIIIIIIIH", seg[:36])
            comps = [tuple(seg[36 + 3 * k:39 + 3 * k]) for k in range(n)]
            bits = sorted({(c[0] & 0x7F) + 1 for c in comps})
            out(f"{indent}   SIZ  {xs - xo}x{ys - yo}, tile {xt}x{yt}, {n} components, "
                f"{'/'.join(map(str, bits))}-bit (Ssiz,XRsiz,YRsiz = {comps})")
        elif name == "COD":
            scod, prog, layers, mct, levels, cbw, cbh, style, xform = struct.unpack(">BBHBBBBBB", seg[:10])
            progs = {0: "LRCP", 1: "RLCP", 2: "RPCL", 3: "PCRL", 4: "CPRL"}
            out(f"{indent}   COD  {progs.get(prog, prog)}, {layers} layer(s), MCT={mct}, "
                f"{levels} levels, code-block {2 ** (cbw + 2)}x{2 ** (cbh + 2)}, "
                f"{'5/3 reversible' if xform else '9/7 irreversible'}, "
                f"precincts={'yes' if scod & 1 else 'no'}")
        elif name == "COM":
            out(f"{indent}   COM  {seg[2:].decode('latin-1').rstrip()}")
        else:
            out(f"{indent}   {name}  len={seg_len}: {seg[:24].hex(' ')}")
        pos += 2 + seg_len


def walk(f, start, end, depth, args, out):
    indent = "  " * depth
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        length, box_type = struct.unpack(">I4s", f.read(8))
        header = 8
        if length == 1:
            length = struct.unpack(">Q", f.read(8))[0]
            header = 16
        elif length == 0:
            length = end - pos
        if length < header:
            out(f"{indent}** bad box length {length} at offset {pos} **")
            return
        name = box_type.decode("latin-1")
        out(f"{indent}'{name}' at {pos}, length {length}")
        body = length - header
        if box_type in SUPERBOXES:
            walk(f, pos + header, pos + length, depth + 1, args, out)
        elif box_type == b"jp2c":
            dump_codestream_header(f, indent, out)
        elif body <= MAX_BOX_READ:
            data = f.read(body)
            if box_type in TEXT_BOXES:
                if box_type == b"uuid":
                    kind = "XMP" if data[:16] == XMP_UUID else "uuid " + data[:16].hex()
                    data = data[16:]
                else:
                    kind = name.strip()
                # Collapse whitespace: XMP packets carry ~128 KB of blank padding
                text = " ".join(data.decode("utf-8", "replace").split())
                if not args.full_xmp and len(text) > 4000:
                    text = text[:4000] + f" ... [{len(text) - 4000} more chars; use --full-xmp]"
                out(f"{indent}   {kind}: {text}")
            else:
                out(f"{indent}   {data[:64].hex(' ')}")
        pos += length


def main():
    ap = argparse.ArgumentParser(description="Dump the box structure of a JPEG2000 file.")
    ap.add_argument("path")
    ap.add_argument("--full-xmp", action="store_true",
                    help="Print XML/XMP boxes in full (default: first 4000 chars)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.path.isfile(args.path):
        sys.exit(f"File not found: {args.path}")
    with open(args.path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
        if f.read(2) == b"\xff\x4f":  # raw codestream, no boxes
            f.seek(0)
            dump_codestream_header(f, "", print)
        else:
            walk(f, 0, size, 0, args, print)


if __name__ == "__main__":
    main()
