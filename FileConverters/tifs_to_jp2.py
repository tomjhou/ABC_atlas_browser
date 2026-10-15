#!/usr/bin/env python3
"""
tifs_to_jp2.py

Merge single-channel greyscale TIFFs (one file per channel, e.g. the Leica
export "section 33-rmtg--C00.tif" ... "--C03.tif") into one multi-channel
JPEG2000 (.jp2) per section, laid out the same way NeuroInfo /
Microfile+ writes its own .jp2 files, so NeuroInfo picks up the channel
colors, channel (layer) names and pixel size.

Run with no arguments to open a GUI. Otherwise use the command line:

    python tifs_to_jp2.py "D:/.../10x" "D:/.../10x/jp2" ^
        --channel C02 Sox6 red --channel C01 FoxP1 green ^
        --channel C03 Synaptophysin blue

    Only process some sections:   --sections 28-35 40
    Lossless (much larger files): --lossless

File layout written (mirrors the files Microfile+ 2024.1.1 produces):
    'jP  ' signature
    'ftyp' brand jp2
    'jp2h' -> 'ihdr' (N components, 8- or 16-bit unsigned), 'colr' (sRGB
              for 3 channels, greyscale otherwise, as Microfile+ does)
    'uuid' XMP packet (MBF schema): pixel size, one ColorMap per channel
           (black -> channel color), and one Group per channel whose
           GroupName is the layer name
    'jp2c' codestream: single tile, no multi-component transform, 6 wavelet
           levels, RPCL progression with precincts, plus the COM markers
           "MBF Bioscience", "MBF_DIMT:1", "MBF_DIMZ:1"

File names: every '<name>--C0N.tif' is one channel of section '<name>'.
Leica pyramid exports ('<name>_pmd_0--C00.tif' ... '_pmd_7--C00.tif') are
detected; only the full-resolution level _pmd_0 is used, and '_pmd_0' is
dropped from the output name. Section numbers (for choosing which sections
to convert) come from "section N" anywhere in the name, otherwise from the
last number in the name.

Pixel size comes from the TIFF resolution tags unless one is entered. A
section whose TIFF has no calibration, or one outside 0.05-50 um/pixel
(e.g. Photoshop's default 72 dpi), is not converted until a pixel size is
entered.

8-bit TIFFs are written as 8-bit. 12-bit data: the Leica TIFFs hold 12-bit
values (0-4095) in 16-bit files, which is why they look dim. By default
(input bits "auto") a section whose selected channels never exceed 4095 is
multiplied by 16 (the same as the ImageJ Convert16 macro) so it fills the
16-bit range.

Requires: numpy, tifffile, imagecodecs. Any that are missing are installed
automatically with pip into the Python environment running this script.
"""

import argparse
import datetime
import glob
import importlib
import importlib.util
import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


# Import name -> pip package name
REQUIRED_PACKAGES = {"numpy": "numpy", "tifffile": "tifffile", "imagecodecs": "imagecodecs"}


def _missing_packages():
    return [pip_name for module, pip_name in REQUIRED_PACKAGES.items()
            if importlib.util.find_spec(module) is None]


def _report_install_failure(message):
    print(message, file=sys.stderr)
    if len(sys.argv) == 1:
        # GUI launch, possibly with no console window to show the message
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing libraries", message)
            root.destroy()
        except Exception:
            pass
    sys.exit(1)


def ensure_packages():
    """Install any missing REQUIRED_PACKAGES into this interpreter's environment."""
    missing = _missing_packages()
    if not missing:
        return
    if getattr(sys, "frozen", False):
        _report_install_failure(f"This build is missing {', '.join(missing)} and "
                                f"can't install packages itself.")
    print(f"Installing missing libraries ({', '.join(missing)}) into the Python at\n"
          f"  {sys.executable}\nThis only happens once and may take a minute...", flush=True)
    if importlib.util.find_spec("pip") is None:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"])
    # Prebuilt wheels only: building numpy/imagecodecs from source needs a C
    # compiler and native libraries, and fails after a long wait without them
    result = subprocess.run([sys.executable, "-m", "pip", "install",
                             "--only-binary", ":all:", *missing])
    importlib.invalidate_caches()
    # pip falls back to a per-user install when site-packages isn't writable;
    # that folder may not have existed (so isn't on sys.path) when Python started
    import site
    user_site = site.getusersitepackages()
    if site.ENABLE_USER_SITE and os.path.isdir(user_site) and user_site not in sys.path:
        site.addsitedir(user_site)
    still_missing = _missing_packages()
    if still_missing:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _report_install_failure(
            f"Couldn't install {', '.join(still_missing)} for Python {version} "
            f"(pip exit code {result.returncode}).\n\n"
            f"If this Python version is very new, prebuilt packages may not exist "
            f"for it yet. Try running the script with an older Python, e.g.\n"
            f"  py -3.12 tifs_to_jp2.py")
    print("Libraries installed.\n")


ensure_packages()

import numpy as np  # noqa: E402  (imported after ensure_packages)
import tifffile  # noqa: E402
import imagecodecs  # noqa: E402


NAMED_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
    "gray": (255, 255, 255),
    "grey": (255, 255, 255),
}

MAX_CHANNELS = 4

# Default lossy quality (target PSNR, dB) by output bit depth
DEFAULT_QUALITY_DB = {8: 45.0, 16: 50.0}

# UUID that marks an XMP packet inside a JPEG2000 'uuid' box
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")
# Microfile+ pads its XMP packet (the uuid box payload after the UUID) to
# exactly this many bytes
XMP_PACKET_SIZE = 131072


class UserCanceled(Exception):
    pass


def parse_color(text):
    t = text.strip().lower()
    if t in NAMED_COLORS:
        return NAMED_COLORS[t]
    m = re.fullmatch(r"#?([0-9a-f]{6})", t)
    if m:
        h = m.group(1)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    m = re.fullmatch(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", t)
    if m and all(int(v) <= 255 for v in m.groups()):
        return tuple(int(v) for v in m.groups())
    raise ValueError(
        f"Unknown color '{text}' (use a name like red/green/blue/cyan/"
        f"magenta/yellow/white, a hex value like #FF8000, or R,G,B)")


def parse_sections(text):
    """'28-35, 40 42' -> {28, ..., 35, 40, 42}; blank -> None (= all)."""
    text = text.strip()
    if not text:
        return None
    wanted = set()
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            lo, hi = sorted((int(m.group(1)), int(m.group(2))))
            wanted.update(range(lo, hi + 1))
        elif part.isdigit():
            wanted.add(int(part))
        else:
            raise ValueError(f"Can't understand section '{part}' "
                             f"(use numbers and ranges, e.g. 28-35, 40)")
    return wanted


def box(box_type, payload):
    return struct.pack(">I4s", 8 + len(payload), box_type) + payload


def xml_attr(s):
    return (str(s).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def build_xmp(channels, pixel_um, bits=16):
    """MBF XMP packet, structured exactly like Microfile+'s."""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
    colormaps = []
    chans = []
    groups = []
    white_point = 2 ** bits - 1
    for i, ch in enumerate(channels):
        r, g, b = ch["color"]
        colormaps.append(
            f'<rdf:li> <rdf:Description xmp:Map="{i}" xmp:BitsUsed="{bits}" '
            f'xmp:MapType="0" xmp:BlackPoint="0" xmp:WhitePoint="{white_point}" '
            f'xmp:Gamma="1.000000" xmp:GrayScale="False" xmp:Gain="1.000000" '
            f'xmp:WhiteBalance="1.000000" xmp:Offset="0"> '
            f'<xmp:BlackColor xmp:Red="0" xmp:Green="0" xmp:Blue="0" xmp:Alpha="0"/> '
            f'<xmp:WhiteColor xmp:Red="{r}" xmp:Green="{g}" xmp:Blue="{b}" xmp:Alpha="0"/> '
            f'</rdf:Description> </rdf:li>')
        chans.append(f'<rdf:li xmp:Channel="{i}" xmp:ColorMapID="{i}" xmp:Enabled="True"/>')
        groups.append(
            f'<rdf:li xmp:GroupName="{xml_attr(ch["name"])}" xmp:Channel="{i}" '
            f'xmp:ExposureSeconds="-1.000000" xmp:LineIntervalSeconds="0.000000" '
            f'xmp:Gain="-1.000000"/>')

    xmp = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?> '
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.6.0"> '
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"> '
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/"> '
        '<xmp:Image_Headers> <rdf:Bag> '
        '<rdf:li xmp:MbfXmpVersion="2" xmp:Application="Microfile+" '
        'xmp:Version="2024.1.1" xmp:CameraName="Unspecified" '
        'xmp:MicroscopeName="Unspecified" xmp:LensName="Unspecified" '
        'xmp:Make="Unspecified" xmp:Model="Unspecified" '
        'xmp:CorrectionFactor="0.000000" xmp:CalculatedIllumination="fluorescence" '
        'xmp:CombinationMethod="-1" '
        f'xmp:PixelWidth="{pixel_um:.6f}" xmp:PixelHeight="{pixel_um:.6f}" '
        'xmp:SpacingZ="1.000000" xmp:OrigPixelWidth="-1.000000" '
        'xmp:OrigPixelHeight="-1.000000" xmp:OrigSpacingZ="-1.000000" '
        'xmp:OriginX="0.000000" xmp:OriginY="0.000000" xmp:OriginZ="0.000000" '
        f'xmp:CreateDate="{now}"/> '
        '</rdf:Bag> </xmp:Image_Headers> '
        '<xmp:ColorMaps> <rdf:Bag> ' + " ".join(colormaps) + ' </rdf:Bag> </xmp:ColorMaps> '
        '<xmp:Images> <rdf:Bag> <rdf:li> <rdf:Description xmp:Image="Default"> '
        '<xmp:Channels> <rdf:Bag> ' + " ".join(chans) + ' </rdf:Bag> </xmp:Channels> '
        '<xmp:Groups> <rdf:Bag> ' + " ".join(groups) + ' </rdf:Bag> </xmp:Groups> '
        '</rdf:Description> </rdf:li> </rdf:Bag> </xmp:Images> '
        '</rdf:Description> </rdf:RDF> </x:xmpmeta> ')
    head = xmp.encode("utf-8")
    tail = b'<?xpacket end="w"?>'
    pad = max(0, XMP_PACKET_SIZE - len(head) - len(tail))
    # XMP convention: writable padding is whitespace, newline every ~100 chars
    padding = (b" " * 99 + b"\n") * (pad // 100) + b" " * (pad % 100)
    return XMP_UUID + head + padding + tail


def add_com_markers(codestream, comments):
    """Insert COM marker segments just before the first tile-part (SOT)."""
    if codestream[:2] != b"\xff\x4f":
        raise ValueError("Not a JPEG2000 codestream (missing SOC)")
    pos = 2
    while True:
        marker = struct.unpack(">H", codestream[pos:pos + 2])[0]
        if marker == 0xFF90:  # SOT: end of main header
            break
        seg_len = struct.unpack(">H", codestream[pos + 2:pos + 4])[0]
        pos += 2 + seg_len
    coms = b""
    for c in comments:
        text = c.encode("latin-1")
        # Rcom = 1: Latin-1 text
        coms += struct.pack(">HHH", 0xFF64, 4 + len(text), 1) + text
    return codestream[:pos] + coms + codestream[pos:]


def build_jp2(codestream, height, width, n_comp, bits, xmp_payload):
    ihdr = struct.pack(">IIHBBBB", height, width, n_comp, bits - 1, 7, 1, 0)
    # Enumerated colorspace: 16 = sRGB, 17 = greyscale. Microfile+ uses sRGB
    # only for exactly 3 channels and greyscale (with no cdef box) otherwise.
    enum_cs = 16 if n_comp == 3 else 17
    colr = struct.pack(">BBBI", 1, 0, 0, enum_cs)
    jp2h_payload = box(b"ihdr", ihdr) + box(b"colr", colr)
    out = box(b"jP  ", b"\r\n\x87\n")
    out += box(b"ftyp", b"jp2 " + struct.pack(">I", 0) + b"jp2 ")
    out += box(b"jp2h", jp2h_payload)
    out += box(b"uuid", xmp_payload)
    if 8 + len(codestream) < 2 ** 32:
        out += box(b"jp2c", codestream)
    else:
        out += struct.pack(">I4sQ", 1, b"jp2c", 16 + len(codestream)) + codestream
    return out


def scan_folder(input_dir):
    """
    Map section name -> {tag: path} for every '<name>--C0N.tif'. For Leica
    pyramid exports ('<name>_pmd_<level>--C0N.tif') only level 0 (full
    resolution) is used, and the section name excludes '_pmd_0'.
    """
    sections = {}
    for path in glob.glob(os.path.join(input_dir, "*.tif")):
        m = re.fullmatch(r"(.*?)(_pmd_(\d+))?--(C\d+)\.tif",
                         os.path.basename(path), re.IGNORECASE)
        if not m:
            continue
        name, level, tag = m.group(1), m.group(3), m.group(4).upper()
        if level is not None and int(level) != 0:
            continue
        sections.setdefault(name, {})[tag] = path
    return sections


def section_number(name):
    """'Sequence 001_section 10-vta' -> 10; else the last number in the name."""
    m = re.search(r"section\s*(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def section_sort_key(prefix):
    n = section_number(prefix)
    return (n is None, n or 0, prefix)


def tiff_pixel_size_um(path):
    """Pixel width in microns from the TIFF resolution tags, or None."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags
        if "XResolution" not in tags:
            return None
        num, den = tags["XResolution"].value
        if num == 0 or den == 0:
            return None
        px_per_unit = num / den
        unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
        if tif.is_imagej:
            # ImageJ writes pixels per its own unit (normally microns)
            desc = tif.imagej_metadata or {}
            if str(desc.get("unit", "")).lower() in ("micron", "um", "\u00b5m", "\\u00b5m"):
                return round(1.0 / px_per_unit, 5)
            return None
        if unit == 3:  # centimeter
            return round(1e4 / px_per_unit, 5)
        if unit == 2:  # inch
            return round(25400.0 / px_per_unit, 5)
        return None


class PixelSizeError(ValueError):
    pass


# Pixel sizes outside this range (microns) are almost certainly not a
# microscope calibration, e.g. 72/96/300 dpi written by default by Photoshop
PLAUSIBLE_PIXEL_UM = (0.05, 50.0)

PIXEL_SIZE_HINT = "enter the pixel size in the Pixel size box (or --pixel-size)"


def resolve_pixel_size(path, opts):
    """
    Pixel size to write, as (microns, description). Uses opts.pixel_size if
    given, otherwise the TIFF's resolution tags. Raises PixelSizeError if the
    TIFF has no calibration or one that looks like a default print resolution.
    """
    if opts.pixel_size is not None:
        lo, hi = PLAUSIBLE_PIXEL_UM
        note = "" if lo <= opts.pixel_size <= hi else "  ** unusual value - please double-check **"
        return opts.pixel_size, f"{opts.pixel_size:g} um (entered){note}"
    um = tiff_pixel_size_um(path)
    if um is None:
        raise PixelSizeError(f"{os.path.basename(path)} has no pixel size in its "
                             f"metadata; {PIXEL_SIZE_HINT}")
    lo, hi = PLAUSIBLE_PIXEL_UM
    if not lo <= um <= hi:
        dpi = 25400.0 / um
        raise PixelSizeError(
            f"{os.path.basename(path)} says {um:g} um/pixel ({dpi:.0f} dpi), which "
            f"looks like a default print resolution, not a microscope calibration; "
            f"{PIXEL_SIZE_HINT}")
    return um, f"{um:g} um (from TIFF)"


def convert_section(files, channels, opts, output_path, log=print):
    t0 = time.time()
    # Checked first, so a bad calibration fails before the slow image read
    pixel_um, _ = resolve_pixel_size(files[channels[0]["tag"]], opts)
    planes = []
    for ch in channels:
        img = tifffile.imread(files[ch["tag"]])
        if img.ndim != 2:
            raise ValueError(f"{files[ch['tag']]} is not a single greyscale plane "
                             f"(shape {img.shape})")
        if img.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"{files[ch['tag']]} has unsupported dtype {img.dtype}")
        planes.append(img)
    shapes = {p.shape for p in planes}
    if len(shapes) != 1:
        raise ValueError(f"Channel images differ in size: {sorted(shapes)}")

    if all(p.dtype == np.uint8 for p in planes):
        # Kept as 8-bit, as Microfile+ does
        out_bits = 8
        bits = "8"
    else:
        out_bits = 16
        planes = [p.astype(np.uint16) * 257 if p.dtype == np.uint8 else p
                  for p in planes]
        bits = opts.input_bits
        if bits == "auto":
            bits = "12" if max(int(p.max()) for p in planes) <= 4095 else "16"
        if bits == "12":
            for p in planes:
                np.left_shift(p, 4, out=p)  # x16, same as ImageJ "Multiply 16"
    data = np.stack(planes, axis=-1)
    del planes
    height, width, n_comp = data.shape

    # PSNR is measured against the full-scale value, so the same dB is much
    # gentler on 8-bit data; these defaults give roughly 20-25:1 on our scans
    quality = opts.quality
    if quality is None:
        quality = DEFAULT_QUALITY_DB[out_bits]

    codestream = imagecodecs.jpeg2k_encode(
        data,
        None if opts.lossless else quality,
        codecformat="j2k",
        reversible=opts.lossless,
        mct=False,  # keep fluorescence channels independent
        resolutions=7,  # 6 decomposition levels, as Microfile+ uses
        numthreads=opts.threads,
    )
    del data
    codestream = add_com_markers(
        codestream, ["MBF Bioscience", "MBF_DIMT:1", "MBF_DIMZ:1"])
    jp2 = build_jp2(codestream, height, width, n_comp, out_bits,
                    build_xmp(channels, pixel_um, out_bits))

    tmp = output_path + ".partial"
    with open(tmp, "wb") as f:
        f.write(jp2)
    os.replace(tmp, output_path)
    raw_bytes = height * width * n_comp * out_bits // 8
    compression = "lossless" if opts.lossless else f"{quality:g} dB"
    log(f"  {width}x{height}, {n_comp} ch, input {bits}-bit, "
        f"{pixel_um:.4f} um/px, {compression} -> {len(jp2) / 1e6:.1f} MB "
        f"(ratio {raw_bytes / len(jp2):.1f}:1) in {time.time() - t0:.0f} s")


def plan_conversion(input_dir, output_dir, channels, opts, sections=None):
    """
    Work out what convert_folder would do, without reading any image data.
    Returns (jobs, incomplete): jobs is a list of dicts with keys name, files,
    out and skip (output exists and overwrite is off), in conversion order;
    incomplete is a list of (name, missing tags) for sections lacking files.
    """
    tags = [c["tag"] for c in channels]
    found = scan_folder(input_dir)
    jobs = []
    incomplete = []
    for name in sorted(found, key=section_sort_key):
        if sections is not None and section_number(name) not in sections:
            continue
        files = found[name]
        missing = [t for t in tags if t not in files]
        if missing:
            incomplete.append((name, missing))
            continue
        out = os.path.join(output_dir, name + opts.suffix + ".jp2")
        jobs.append({"name": name, "files": files, "out": out,
                     "skip": os.path.exists(out) and not opts.overwrite})
    return jobs, incomplete


def log_channels(channels, log):
    if len(channels) not in (3, 4):
        log(f"NOTE: {len(channels)} channel(s) requested. NeuroInfo's own files "
            f"have 3 or 4; other counts have not been tested in NeuroInfo.")
    log("Channels: " + ", ".join(
        f"{c['tag']} -> '{c['name']}' {c['color']}" for c in channels))


# Preview estimates the data range from this many evenly spaced rows. A fixed
# count (not every Nth row) keeps the read to a few MB however large the
# file is: Windows read-ahead means every-Nth-row reads most of the file.
PREVIEW_SAMPLE_ROWS = 128


# Windows file attributes of OneDrive/SharePoint "online-only" placeholders.
# Opening such a file makes OneDrive download all of it.
_CLOUD_ONLY_ATTRS = 0x00400000 | 0x00040000 | 0x00001000  # RECALL_ON_DATA_ACCESS | RECALL_ON_OPEN | OFFLINE


def is_online_only(path):
    """True for a cloud placeholder that isn't stored on this computer.
    Checking the attributes doesn't trigger a download."""
    try:
        return bool(getattr(os.stat(path), "st_file_attributes", 0) & _CLOUD_ONLY_ATTRS)
    except OSError:
        return False


def sampled_max(path, n_rows=PREVIEW_SAMPLE_ROWS):
    """Max pixel value over n_rows evenly spaced rows. Uncompressed TIFFs are
    memory-mapped, so only those rows are read; others are read in full."""
    try:
        data = tifffile.memmap(path, mode="r")
    except ValueError:
        return int(tifffile.imread(path).max())
    rows = np.unique(np.linspace(0, data.shape[0] - 1, n_rows).astype(int))
    return int(max(int(data[r].max()) for r in rows))


def describe_scaling(dtypes, max_value, opts):
    """Plain-language summary of how convert_section will treat the data."""
    if all(d == np.uint8 for d in dtypes):
        return "8-bit data, written as 8-bit"
    if opts.input_bits == "12":
        return "treated as 12-bit (forced): values x16 to fill 16 bits"
    if opts.input_bits == "16":
        return "treated as 16-bit (forced): values unchanged"
    if max_value is None:
        return "data range not checked"
    if max_value <= 4095:
        return (f"max {max_value} in sampled rows -> 12-bit data, values x16 "
                f"to fill 16 bits")
    return f"max {max_value} in sampled rows -> 16-bit data, values unchanged"


def preview_folder(input_dir, output_dir, channels, opts, sections=None,
                   log=print, stop_event=None):
    """
    List the sections, input files and output files a conversion would use,
    with each section's pixel size and how its values will be scaled. Reads
    TIFF headers plus PREVIEW_SAMPLE_ROWS rows of pixels per file, except for
    OneDrive online-only files, which are never opened (opening one would
    download all of it). Returns (n_to_convert, n_to_skip).
    """
    jobs, incomplete = plan_conversion(input_dir, output_dir, channels, opts, sections)
    n_skip = sum(j["skip"] for j in jobs)
    log(f"Preview: {len(jobs)} section(s) match; output folder '{output_dir}'")
    log_channels(channels, log)
    name_width = max([len(j["name"]) for j in jobs] + [10])
    online_count = 0
    online_bytes = 0
    for job in jobs:
        if stop_event is not None and stop_event.is_set():
            raise UserCanceled
        note = "   [exists - will be skipped]" if job["skip"] else ""
        log(f"{job['name']:<{name_width}} -> {os.path.basename(job['out'])}{note}")
        shapes = set()
        dtypes = []
        max_value = None
        any_online = False
        for ch in channels:
            path = job["files"][ch["tag"]]
            if is_online_only(path):
                any_online = True
                if not job["skip"]:
                    online_count += 1
                    online_bytes += os.path.getsize(path)
                log(f"    {ch['tag']}: {os.path.basename(path)}   "
                    f"(online-only in OneDrive - not downloaded for preview)")
                continue
            try:
                with tifffile.TiffFile(path) as tif:
                    page = tif.pages[0]
                    shapes.add(page.shape)
                    dtypes.append(page.dtype)
                    info = f"{page.shape[1]}x{page.shape[0]}, {page.dtype.itemsize * 8}-bit file"
                    if len(page.shape) != 2:
                        info += "  ** not a single greyscale plane **"
                if opts.input_bits == "auto" and page.dtype == np.uint16:
                    m = sampled_max(path)
                    info += f", max {m}"
                    max_value = m if max_value is None else max(max_value, m)
            except Exception as e:
                info = f"** can't read: {e} **"
            log(f"    {ch['tag']}: {os.path.basename(path)}   ({info})")
        if len(shapes) > 1:
            log("    ** WARNING: channel images differ in size; this section will fail **")
        first_path = job["files"][channels[0]["tag"]]
        if opts.pixel_size is None and is_online_only(first_path):
            pixel = "not checked (file is online-only)"
        else:
            try:
                _, pixel = resolve_pixel_size(first_path, opts)
            except PixelSizeError as e:
                pixel = f"** {e} - this section will fail **"
            except Exception as e:
                pixel = f"** can't read: {e} **"
        log(f"    Pixel size: {pixel}")
        if any_online:
            log("    Values: not checked (some files are online-only); decided during conversion")
        elif dtypes:
            log(f"    Values: {describe_scaling(dtypes, max_value, opts)}")
    for name, missing in incomplete:
        log(f"Not converted: '{name}' - missing {', '.join(missing)}")
    if not jobs:
        log("No matching sections found.")
    if online_count:
        log(f"NOTE: {online_count} input file(s) ({online_bytes / 1e9:.1f} GB) are online-only "
            f"in OneDrive; converting will download them to this computer.")
    log(f"Will convert {len(jobs) - n_skip}, skip {n_skip} (already exist).")
    return len(jobs) - n_skip, n_skip


def convert_folder(input_dir, output_dir, channels, opts, sections=None,
                   log=print, progress=None, stop_event=None):
    """
    Convert every section in input_dir that has all requested channel files.
    sections: set of section numbers to restrict to, or None for all.
    progress(i, n, name) is called before each section.
    Returns (n_converted, n_skipped, n_failed).
    """
    jobs, incomplete = plan_conversion(input_dir, output_dir, channels, opts, sections)
    for name, missing in incomplete:
        log(f"Skipping '{name}': missing {', '.join(missing)}")
    if not jobs:
        log("No matching sections found.")
        return 0, 0, 0
    os.makedirs(output_dir, exist_ok=True)
    log_channels(channels, log)

    n_done = n_skipped = n_failed = 0
    for i, job in enumerate(jobs, 1):
        if stop_event is not None and stop_event.is_set():
            raise UserCanceled
        if progress is not None:
            progress(i, len(jobs), job["name"])
        log(f"[{i}/{len(jobs)}] {job['name']}")
        if job["skip"]:
            log("  exists, skipping")
            n_skipped += 1
            continue
        try:
            convert_section(job["files"], channels, opts, job["out"], log)
            n_done += 1
        except Exception as e:  # keep going with the other sections
            n_failed += 1
            log(f"  FAILED: {e}")
    log(f"Done: {n_done} converted, {n_skipped} skipped (already exist), "
        f"{n_failed} failed.")
    return n_done, n_skipped, n_failed


##################################################
#
#    Command line
#
##################################################

def main_cli():
    ap = argparse.ArgumentParser(
        description="Merge per-channel greyscale TIFFs into NeuroInfo-style "
                    "multi-channel 16-bit JPEG2000 files. Run with no "
                    "arguments to open the GUI instead.")
    ap.add_argument("input_dir", help="Folder containing '<section>--C0N.tif' files")
    ap.add_argument("output_dir", help="Folder to write .jp2 files into")
    ap.add_argument("--channel", nargs=3, action="append", required=True,
                    metavar=("TAG", "NAME", "COLOR"),
                    help="Channel file suffix (e.g. C02), layer name, and display "
                         "color. Repeat once per channel, in the order the "
                         "channels should appear in the file.")
    ap.add_argument("--sections", nargs="+",
                    help="Only convert these section numbers, e.g. 28-35 40")
    ap.add_argument("--input-bits", choices=["auto", "12", "16"], default="auto",
                    help="12 = scale values x16 to fill 16 bits; auto (default) "
                         "picks 12 when no selected channel exceeds 4095")
    ap.add_argument("--quality", type=float,
                    help="Lossy quality as target PSNR in dB (default 50 for "
                         "16-bit, 45 for 8-bit; higher = better quality and "
                         "bigger files)")
    ap.add_argument("--lossless", action="store_true",
                    help="Reversible lossless compression (files ~10x+ larger)")
    ap.add_argument("--pixel-size", type=float,
                    help="Pixel size in microns (default: read from the TIFF)")
    ap.add_argument("--suffix", default="",
                    help="Text appended to the section name in the output file "
                         "name, e.g. '--C02' to match NeuroInfo's naming")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files (default: skip them)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the files that would be converted, then stop")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                    help="Encoder threads (default: all CPUs)")
    args = ap.parse_args()

    if len(args.channel) > MAX_CHANNELS:
        ap.error(f"At most {MAX_CHANNELS} channels are supported")
    channels = []
    for tag, name, color in args.channel:
        tag = tag.upper()
        if not re.fullmatch(r"C\d+", tag):
            ap.error(f"Channel tag '{tag}' should look like C00, C01, ...")
        try:
            rgb = parse_color(color)
        except ValueError as e:
            ap.error(str(e))
        channels.append({"tag": tag, "name": name, "color": rgb})
    if len({c["tag"] for c in channels}) != len(channels):
        ap.error("The same channel tag was given twice")
    try:
        sections = parse_sections(" ".join(args.sections or []))
    except ValueError as e:
        ap.error(str(e))

    opts = SimpleNamespace(input_bits=args.input_bits, quality=args.quality,
                           lossless=args.lossless, pixel_size=args.pixel_size,
                           suffix=args.suffix, overwrite=args.overwrite,
                           threads=args.threads)
    if args.dry_run:
        preview_folder(args.input_dir, args.output_dir, channels, opts, sections)
        return
    n_done, n_skipped, n_failed = convert_folder(
        args.input_dir, args.output_dir, channels, opts, sections)
    if n_failed:
        sys.exit(1)


##################################################
#
#    GUI
#
##################################################

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".tifs_to_jp2_settings.json")

# Leica's own LUT colors for C00..C03 in these exports
DEFAULT_CHANNELS = [
    {"tag": "C00", "name": "", "color": "blue"},
    {"tag": "C01", "name": "", "color": "green"},
    {"tag": "C02", "name": "", "color": "yellow"},
    {"tag": "C03", "name": "", "color": "red"},
]


class Jp2Gui:
    PAD = 5

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk

        self.root = tk.Tk()
        self.root.title("TIFF to NeuroInfo JP2 converter")

        settings = self.load_settings()

        self.var_input = tk.StringVar(self.root, settings.get("input_dir", ""))
        self.var_output = tk.StringVar(self.root, settings.get("output_dir", ""))
        self.var_sections = tk.StringVar(self.root, settings.get("sections", ""))
        self.var_quality = tk.StringVar(self.root, str(settings.get("quality", "")))
        self.var_lossless = tk.BooleanVar(self.root, settings.get("lossless", False))
        self.var_overwrite = tk.BooleanVar(self.root, settings.get("overwrite", False))
        self.var_bits = tk.StringVar(self.root, settings.get("input_bits", "auto"))
        self.var_pixel = tk.StringVar(self.root, settings.get("pixel_size", ""))
        self.var_found = tk.StringVar(self.root, "")
        self.var_status = tk.StringVar(self.root, "Ready")

        saved_channels = settings.get("channels", DEFAULT_CHANNELS)
        self.channel_vars = []
        for i in range(MAX_CHANNELS):
            ch = saved_channels[i] if i < len(saved_channels) else DEFAULT_CHANNELS[i]
            self.channel_vars.append({
                "tag": tk.StringVar(self.root, ch.get("tag", DEFAULT_CHANNELS[i]["tag"])),
                "name": tk.StringVar(self.root, ch.get("name", "")),
                "color": tk.StringVar(self.root, ch.get("color", DEFAULT_CHANNELS[i]["color"])),
            })

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        self.build()
        self.update_found()
        self.root.after(100, self.poll_log)

    @staticmethod
    def load_settings():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save_settings(self):
        settings = {
            "input_dir": self.var_input.get(),
            "output_dir": self.var_output.get(),
            "sections": self.var_sections.get(),
            "quality": self.var_quality.get(),
            "lossless": self.var_lossless.get(),
            "overwrite": self.var_overwrite.get(),
            "input_bits": self.var_bits.get(),
            "pixel_size": self.var_pixel.get(),
            "channels": [{k: v.get() for k, v in cv.items()} for cv in self.channel_vars],
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except OSError:
            pass

    def build(self):
        tk, ttk = self.tk, self.ttk
        pad = self.PAD

        # ****** Folders
        f = tk.LabelFrame(self.root, text="Folders")
        f.pack(side=tk.TOP, fill=tk.X, padx=15, pady=(10, 5), ipadx=5, ipady=5)
        f.columnconfigure(1, weight=1)

        tk.Label(f, text="TIFF folder:").grid(row=0, column=0, sticky=tk.E, padx=pad, pady=pad)
        e = ttk.Entry(f, textvariable=self.var_input, width=70)
        e.grid(row=0, column=1, sticky=tk.EW, pady=pad)
        e.bind("<FocusOut>", lambda _e: self.update_found())
        e.bind("<Return>", lambda _e: self.update_found())
        ttk.Button(f, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=pad)

        tk.Label(f, textvariable=self.var_found, fg="gray25", anchor="w", justify=tk.LEFT).grid(
            row=1, column=1, columnspan=2, sticky=tk.W)

        tk.Label(f, text="Output folder:").grid(row=2, column=0, sticky=tk.E, padx=pad, pady=pad)
        ttk.Entry(f, textvariable=self.var_output, width=70).grid(row=2, column=1, sticky=tk.EW, pady=pad)
        ttk.Button(f, text="Browse...", command=self.browse_output).grid(row=2, column=2, padx=pad)
        tk.Label(f, text="(blank = 'jp2' subfolder of the TIFF folder)", fg="gray25").grid(
            row=3, column=1, sticky=tk.W)

        tk.Label(f, text="Sections:").grid(row=4, column=0, sticky=tk.E, padx=pad, pady=pad)
        ttk.Entry(f, textvariable=self.var_sections, width=30).grid(row=4, column=1, sticky=tk.W, pady=pad)
        tk.Label(f, text="(blank = all; e.g. 28-35, 40)", fg="gray25").grid(
            row=5, column=1, sticky=tk.W)

        # ****** Channels
        f = tk.LabelFrame(self.root, text="Channels (in file order; leave label blank to omit a channel)")
        f.pack(side=tk.TOP, fill=tk.X, padx=15, pady=5, ipadx=5, ipady=5)
        for col, text in enumerate(["", "File", "Layer label", "Color", ""]):
            tk.Label(f, text=text).grid(row=0, column=col, padx=pad, sticky=tk.W)
        self.swatches = []
        for i, cv in enumerate(self.channel_vars):
            r = i + 1
            tk.Label(f, text=f"Layer {i + 1}:").grid(row=r, column=0, sticky=tk.E, padx=pad, pady=2)
            ttk.Combobox(f, textvariable=cv["tag"], width=6,
                         values=[f"C{k:02d}" for k in range(MAX_CHANNELS)]).grid(
                row=r, column=1, padx=pad, pady=2)
            ttk.Entry(f, textvariable=cv["name"], width=30).grid(row=r, column=2, padx=pad, pady=2)
            ttk.Combobox(f, textvariable=cv["color"], width=12,
                         values=["red", "green", "blue", "cyan", "magenta", "yellow", "white"]).grid(
                row=r, column=3, padx=pad, pady=2)
            sw = tk.Label(f, width=3, relief="sunken", cursor="hand2")
            sw.grid(row=r, column=4, padx=pad, pady=2)
            sw.bind("<Button-1>", lambda _e, k=i: self.pick_color(k))
            self.swatches.append(sw)
            cv["color"].trace_add("write", lambda *_a, k=i: self.update_swatch(k))
            self.update_swatch(i)

        # ****** Options
        f = tk.LabelFrame(self.root, text="Options")
        f.pack(side=tk.TOP, fill=tk.X, padx=15, pady=5, ipadx=5, ipady=5)
        tk.Label(f, text="Quality (dB):").grid(row=0, column=0, sticky=tk.E, padx=pad, pady=pad)
        self.quality_box = ttk.Spinbox(f, textvariable=self.var_quality, from_=30, to=80,
                                       increment=1, width=6)
        self.quality_box.grid(row=0, column=1, sticky=tk.W, pady=pad)
        tk.Label(f, text="blank = auto (50 for 16-bit, 45 for 8-bit, about 20-25:1); "
                         "higher = better quality, bigger files",
                 fg="gray25").grid(row=0, column=2, sticky=tk.W, padx=pad)
        ttk.Checkbutton(f, text="Lossless (files ~15x larger)", variable=self.var_lossless,
                        command=self.update_quality_state).grid(
            row=1, column=1, columnspan=2, sticky=tk.W, pady=pad)

        tk.Label(f, text="Input bits:").grid(row=2, column=0, sticky=tk.E, padx=pad, pady=pad)
        ttk.Combobox(f, textvariable=self.var_bits, values=["auto", "12", "16"],
                     state="readonly", width=6).grid(row=2, column=1, sticky=tk.W, pady=pad)
        tk.Label(f, text="12-bit data is multiplied by 16; auto detects it",
                 fg="gray25").grid(row=2, column=2, sticky=tk.W, padx=pad)

        tk.Label(f, text="Pixel size (\u00b5m):").grid(row=3, column=0, sticky=tk.E, padx=pad, pady=pad)
        ttk.Entry(f, textvariable=self.var_pixel, width=10).grid(row=3, column=1, sticky=tk.W, pady=pad)
        tk.Label(f, text="blank = read from each TIFF",
                 fg="gray25").grid(row=3, column=2, sticky=tk.W, padx=pad)

        ttk.Checkbutton(f, text="Overwrite existing .jp2 files", variable=self.var_overwrite).grid(
            row=4, column=1, columnspan=2, sticky=tk.W, pady=pad)
        self.update_quality_state()

        # ****** Buttons
        f = tk.Frame(self.root)
        f.pack(side=tk.TOP, pady=5)
        self.button_preview = ttk.Button(f, text="Preview", command=self.preview)
        self.button_preview.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=8)
        self.button_convert = ttk.Button(f, text="Convert", command=self.start)
        self.button_convert.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=8)
        self.button_cancel = ttk.Button(f, text="Cancel", command=self.cancel, state="disabled")
        self.button_cancel.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=8)
        ttk.Button(f, text="Open output folder", command=self.open_output).pack(
            side=tk.LEFT, padx=10, ipadx=15, ipady=8)
        ttk.Button(f, text="Exit", command=self.exit).pack(side=tk.LEFT, padx=10, ipadx=15, ipady=8)

        # ****** Status and log
        f = tk.Frame(self.root)
        f.pack(side=tk.TOP, fill=tk.X, padx=15)
        tk.Label(f, text="Status:").pack(side=tk.LEFT)
        tk.Label(f, textvariable=self.var_status, relief="sunken", anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        f = tk.Frame(self.root)
        f.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=(5, 10))
        self.log_text = tk.Text(f, height=16, wrap="none", state="disabled", font=("Consolas", 9))
        sb = ttk.Scrollbar(f, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.exit)

    # ---------- helpers

    def update_quality_state(self):
        self.quality_box["state"] = "disabled" if self.var_lossless.get() else "normal"

    def update_swatch(self, k):
        try:
            r, g, b = parse_color(self.channel_vars[k]["color"].get())
            self.swatches[k]["bg"] = f"#{r:02x}{g:02x}{b:02x}"
            self.swatches[k]["text"] = ""
        except ValueError:
            self.swatches[k]["bg"] = self.root.cget("bg")
            self.swatches[k]["text"] = "?"

    def pick_color(self, k):
        from tkinter import colorchooser
        try:
            r, g, b = parse_color(self.channel_vars[k]["color"].get())
            initial = f"#{r:02x}{g:02x}{b:02x}"
        except ValueError:
            initial = None
        _rgb, hex_color = colorchooser.askcolor(initial, parent=self.root,
                                                title=f"Color for layer {k + 1}")
        if hex_color:
            self.channel_vars[k]["color"].set(hex_color.upper())

    def browse_input(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.var_input.get() or None,
                                    title="Folder containing the per-channel TIFFs")
        if d:
            self.var_input.set(os.path.normpath(d))
            self.update_found()

    def browse_output(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.output_dir() or None,
                                    title="Folder to write .jp2 files into")
        if d:
            self.var_output.set(os.path.normpath(d))

    def output_dir(self):
        out = self.var_output.get().strip()
        if out:
            return out
        inp = self.var_input.get().strip()
        return os.path.join(inp, "jp2") if inp else ""

    def open_output(self):
        d = self.output_dir()
        if d and os.path.isdir(d):
            os.startfile(d)
        else:
            self.var_status.set("Output folder doesn't exist yet")

    def update_found(self):
        d = self.var_input.get().strip()
        if not d:
            self.var_found.set("")
            return
        if not os.path.isdir(d):
            self.var_found.set("Folder not found")
            return
        found = scan_folder(d)
        if not found:
            self.var_found.set("No '<section>--C0N.tif' files found in this folder")
            return
        nums = sorted(n for n in (section_number(p) for p in found) if n is not None)
        tags = sorted({t for files in found.values() for t in files})
        rng = f" (numbered {nums[0]}-{nums[-1]})" if nums else ""
        example = sorted(found, key=section_sort_key)[0]
        self.var_found.set(f"Found {len(found)} sections{rng}; channel files: {', '.join(tags)}\n"
                           f"Output names like: '{example}.jp2'  (Preview lists every file)")

    def log(self, text):
        # Called from the worker thread; the GUI thread drains the queue
        self.log_queue.put(("log", text))

    def poll_log(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    print(payload)
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", payload + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                elif kind == "status":
                    self.var_status.set(payload)
                elif kind == "done":
                    self.worker = None
                    self.button_preview["state"] = "normal"
                    self.button_convert["state"] = "normal"
                    self.button_cancel["state"] = "disabled"
                    self.var_status.set(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_log)

    def show_error(self, msg):
        from tkinter import messagebox
        messagebox.showerror("Can't start conversion", msg, parent=self.root)

    # ---------- actions

    def gather_inputs(self):
        """Validate the form. Returns (input_dir, output_dir, channels, opts,
        sections), or None after showing an error."""
        input_dir = self.var_input.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self.show_error("Please choose an existing TIFF folder.")
            return None

        channels = []
        for i, cv in enumerate(self.channel_vars):
            name = cv["name"].get().strip()
            if not name:
                continue
            tag = cv["tag"].get().strip().upper()
            if not re.fullmatch(r"C\d+", tag):
                self.show_error(f"Layer {i + 1}: file '{tag}' should look like C00, C01, ...")
                return None
            try:
                rgb = parse_color(cv["color"].get())
            except ValueError as e:
                self.show_error(f"Layer {i + 1}: {e}")
                return None
            channels.append({"tag": tag, "name": name, "color": rgb})
        if not channels:
            self.show_error("Enter a label for at least one channel.")
            return None
        if len({c["tag"] for c in channels}) != len(channels):
            self.show_error("The same channel file (C0N) is used for two layers.")
            return None

        try:
            sections = parse_sections(self.var_sections.get())
        except ValueError as e:
            self.show_error(str(e))
            return None

        quality = None
        if not self.var_lossless.get() and self.var_quality.get().strip():
            try:
                quality = float(self.var_quality.get())
            except ValueError:
                self.show_error("Quality must be a number (dB), e.g. 50, or blank for automatic.")
                return None

        pixel_size = None
        if self.var_pixel.get().strip():
            try:
                pixel_size = float(self.var_pixel.get())
                if pixel_size <= 0:
                    raise ValueError
            except ValueError:
                self.show_error("Pixel size must be a positive number in microns, or blank.")
                return None

        opts = SimpleNamespace(input_bits=self.var_bits.get(), quality=quality,
                               lossless=self.var_lossless.get(), pixel_size=pixel_size,
                               suffix="", overwrite=self.var_overwrite.get(),
                               threads=os.cpu_count() or 1)
        self.save_settings()
        return input_dir, self.output_dir(), channels, opts, sections

    def run_in_background(self, work, status):
        """Run work() on a worker thread with the buttons disabled; work
        returns the final status message."""
        self.stop_event.clear()
        self.button_preview["state"] = "disabled"
        self.button_convert["state"] = "disabled"
        self.button_cancel["state"] = "normal"
        self.var_status.set(status)

        def wrapper():
            msg = "Completed"
            try:
                msg = work()
            except UserCanceled:
                self.log("Canceled.")
                msg = "Canceled"
            except Exception as e:
                self.log(f"Error: {e}")
                msg = f"Error: {e}"
            finally:
                self.log_queue.put(("done", msg))

        self.worker = threading.Thread(target=wrapper, daemon=True)
        self.worker.start()

    def preview(self):
        if self.worker is not None:
            return
        inputs = self.gather_inputs()
        if inputs is None:
            return
        input_dir, output_dir, channels, opts, sections = inputs
        self.log(f"\n===== Preview of '{input_dir}' =====")

        def work():
            n_convert, n_skip = preview_folder(input_dir, output_dir, channels, opts, sections,
                                               log=self.log, stop_event=self.stop_event)
            return f"Preview: {n_convert} to convert, {n_skip} to skip"

        self.run_in_background(work, "Reading file headers...")

    def start(self):
        if self.worker is not None:
            return
        inputs = self.gather_inputs()
        if inputs is None:
            return
        input_dir, output_dir, channels, opts, sections = inputs
        self.log(f"\n===== Converting '{input_dir}' -> '{output_dir}' =====")

        def progress(i, n, name):
            self.log_queue.put(("status", f"({i} of {n}) {name}"))

        def work():
            n_done, n_skipped, n_failed = convert_folder(
                input_dir, output_dir, channels, opts, sections,
                log=self.log, progress=progress, stop_event=self.stop_event)
            return f"Completed: {n_done} converted, {n_skipped} skipped, {n_failed} failed"

        self.run_in_background(work, "Starting...")

    def cancel(self):
        if self.worker is not None:
            self.stop_event.set()
            self.button_cancel["state"] = "disabled"
            self.var_status.set("Canceling after the current section...")

    def exit(self):
        self.save_settings()
        if self.worker is not None:
            # A section may be half-written; only its .partial file is left behind
            self.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())
        self.root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        Jp2Gui().run()
