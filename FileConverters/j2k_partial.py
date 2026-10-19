"""Reading only the needed parts of a large JPEG 2000 frame.

NeuroInfo's JPX frames are single-tile codestreams written in resolution order (one tile-part per resolution), with 128 x 128
precincts, one quality layer, and the length of every packet recorded (PLT markers). So the pictures at 1/8, 1/16... are a prefix
of the data, and a window at full resolution needs only the packets of the precincts around it. build() makes a codestream that
holds just those packets (every other packet replaced by the one-byte empty packet, which is valid), reading only their byte
ranges from the file: about 7 MB of a 59 MB frame for the 1/8 picture and a 500 px window, and the decoders treat the result like
any other JPEG 2000 stream (the pixels they return are the same as from the whole frame).

parse_layout() raises Unsupported for codestreams that are not laid out like this (the caller then reads the whole frame).
"""
import math
import struct


class Unsupported(Exception):
    pass


def _u16(d, i):
    return struct.unpack(">H", d[i:i + 2])[0]


def _u32(d, i):
    return struct.unpack(">I", d[i:i + 4])[0]


class Layout:
    """The structure of one codestream: its main header, and for each tile-part (resolution) its header and packet lengths."""


def parse_layout(f, off, length):
    """Read the main header and the tile-part headers (which hold the packet lengths) of the codestream at f[off:off + length]."""
    def rd(o, n):
        f.seek(off + o)
        return f.read(n)

    try:
        L = Layout()
        head = rd(0, 4096)
        if _u16(head, 0) != 0xFF4F:
            raise Unsupported("not a codestream")
        i, segs = 2, []
        while True:
            if i + 4 > len(head):
                head += rd(len(head), 65536)
            m = _u16(head, i)
            if m == 0xFF90:
                break
            ln = _u16(head, i + 2)
            if i + 2 + ln > len(head):
                head += rd(len(head), 65536)
            segs.append((m, i, 2 + ln))
            i += 2 + ln
        L.main_end, L.main_segments, L.head = i, segs, head[:i]
        for m, a, n in segs:
            if m == 0xFF51:
                L.xsiz, L.ysiz, L.xo, L.yo = _u32(head, a + 6), _u32(head, a + 10), _u32(head, a + 14), _u32(head, a + 18)
                L.xt, L.yt, L.ncomp = _u32(head, a + 22), _u32(head, a + 26), _u16(head, a + 38)
            elif m == 0xFF52:
                scod, L.prog, L.layers, L.levels = head[a + 4], head[a + 5], _u16(head, a + 6), head[a + 9]
                if not scod & 1 or scod & 6:
                    raise Unsupported("needs precincts, and no SOP/EPH markers")
                L.pp = [(b & 15, b >> 4) for b in head[a + 14:a + n]]
            elif m in (0xFF53, 0xFF5E, 0xFF5F):
                raise Unsupported("COC/RGN/POC markers")
        if L.xt < L.xsiz or L.yt < L.ysiz or L.xo or L.yo:
            raise Unsupported("more than one tile, or an image offset")
        if L.layers != 1 or L.prog not in (1, 2) or len(L.pp) != L.levels + 1:
            raise Unsupported("needs one layer and resolution progression")
        L.tps, pos = [], L.main_end
        while pos < length - 2:
            sot = rd(pos, 12)
            if _u16(sot, 0) != 0xFF90:
                raise Unsupported("tile-part not found")
            psot = _u32(sot, 6)
            if psot == 0:
                raise Unsupported("tile-part of unknown length")
            chunk, j, plt, other = rd(pos, min(psot, 1 << 18)), 12, {}, []
            while True:
                if j + 4 > len(chunk):
                    chunk += rd(pos + len(chunk), min(psot - len(chunk), 1 << 18))
                m = _u16(chunk, j)
                if m == 0xFF93:
                    break
                ln = _u16(chunk, j + 2)
                if j + 2 + ln > len(chunk):
                    chunk += rd(pos + len(chunk), min(psot - len(chunk), 1 << 18))
                if m == 0xFF58:
                    plt[chunk[j + 4]] = chunk[j + 5:j + 2 + ln]
                else:
                    other.append(chunk[j:j + 2 + ln])
                j += 2 + ln
            data_start = pos + j + 2
            lens, cur = [], 0
            for z in sorted(plt):
                for b in plt[z]:
                    cur = (cur << 7) | (b & 0x7F)
                    if not b & 0x80:
                        lens.append(cur)
                        cur = 0
            if not lens or sum(lens) != pos + psot - data_start:
                raise Unsupported("packet lengths missing or not adding up")
            L.tps.append(dict(sot=sot, header_other=b"".join(other), data_start=data_start, lens=lens))
            pos += psot
        if len(L.tps) != L.levels + 1:
            raise Unsupported("not one tile-part per resolution")
        for r, tp in enumerate(L.tps):
            nx, ny = _grid(L, r)
            if len(tp["lens"]) != nx * ny * L.ncomp:
                raise Unsupported("packets do not match the precincts")
        return L
    except (struct.error, IndexError, AttributeError) as e:
        raise Unsupported(f"unexpected structure ({e})")


def _grid(L, r):
    """Number of precincts across and down at resolution r."""
    s = 2 ** (L.levels - r)
    wr, hr = math.ceil(L.xsiz / s), math.ceil(L.ysiz / s)
    return math.ceil(wr / 2 ** L.pp[r][0]), math.ceil(hr / 2 ** L.pp[r][1])


def build(f, off, layout, keep_res, window=None, pad=1, gap=1 << 18):
    """A codestream with all the packets of the resolutions up to keep_res (-1: none), and for the higher ones, those of the
    precincts that cover window = (u0, v0, w, h) at full resolution, plus `pad` precincts around them (the wavelet filters
    reach over the window's border). Ranges of the file less than `gap` bytes apart are read together."""
    L = layout
    plan, ranges = [], []
    for r, tp in enumerate(L.tps):
        px, py = 2 ** L.pp[r][0], 2 ** L.pp[r][1]
        nx, ny = _grid(L, r)
        if r <= keep_res:
            xs, ys = range(nx), range(ny)
        elif window is None:
            xs, ys = (), ()
        else:
            u0, v0, w, h = window
            s = 2 ** (L.levels - r)
            xs = range(max(0, (u0 // s) // px - pad), min(nx, ((u0 + w - 1) // s) // px + pad + 1))
            ys = range(max(0, (v0 // s) // py - pad), min(ny, ((v0 + h - 1) // s) // py + pad + 1))
        offs, o = [], tp["data_start"]
        for ln in tp["lens"]:
            offs.append(o)
            o += ln
        want = {(x, y) for y in ys for x in xs}
        items = []
        for y in range(ny):
            for x in range(nx):
                for c in range(L.ncomp):
                    k = (y * nx + x) * L.ncomp + c
                    it = (offs[k], tp["lens"][k]) if (x, y) in want else None
                    items.append(it)
                    if it:
                        ranges.append(it)
        plan.append(items)
    merged = []
    for o, n in sorted(ranges):
        if merged and o - (merged[-1][0] + merged[-1][1]) <= gap:
            merged[-1][1] = o + n - merged[-1][0]
        else:
            merged.append([o, n])
    blocks = []
    for o, n in merged:
        f.seek(off + o)
        blocks.append((o, f.read(n)))

    def get(o, n):
        for bo, bd in blocks:
            if bo <= o and o + n <= bo + len(bd):
                return bd[o - bo:o - bo + n]
        raise KeyError(o)

    out = bytearray(L.head[:2])
    for m, a, n in L.main_segments:
        if m != 0xFF55:      # (the tile-part lengths of the original no longer hold)
            out += L.head[a:a + n]
    for r, tp in enumerate(L.tps):
        body = bytearray()
        for it in plan[r]:
            body += get(*it) if it else b"\x00"
        hdr = tp["header_other"]
        sot = bytearray(tp["sot"])
        sot[6:10] = struct.pack(">I", 12 + len(hdr) + 2 + len(body))
        out += sot + hdr + b"\xff\x93" + body
    out += b"\xff\xd9"
    return bytes(out)
