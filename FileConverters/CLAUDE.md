# FileConverters

- `LifClass.py`, `LifConverterGUI.py`, `reader.py`, `basic_gui.py`, `LifConverterConsole.py`:
  the Leica .LIF to TIFF/JPG converter (copied from github.com/tomjhou/LIFconverter).
- `tifs_to_jp2.py`: per-channel Leica TIFFs to NeuroInfo JP2 (notes below).
- `tools/jp2_boxdump.py`: prints a JPEG2000 file's boxes, XMP and codestream header.
  Use it to compare our output with NeuroInfo's own files.

# tifs_to_jp2 — NeuroInfo JP2 converter

Merges per-channel Leica TIFFs (`<name>--C0N.tif`) into one multi-channel JPEG2000
per section that NeuroInfo (MBF Bioscience) can open with channel colors, layer
names and pixel size. Replaces NeuroInfo's own converter, which is buggy.

## NeuroInfo JP2 format (worked out from Microfile+ 2024.1.1 output)
- Boxes: `jP  `, `ftyp` (jp2), `jp2h` (ihdr + colr), `uuid` (XMP), `jp2c`.
- colr: sRGB (16) only for exactly 3 channels; greyscale (17) otherwise, with NO cdef box.
- XMP lives in a `uuid` box (UUID be7acfcb97a942e89c71999491e3afac), padded to exactly
  131072 bytes after the UUID. Layer name = `xmp:GroupName`; color = ColorMap WhiteColor;
  BitsUsed/WhitePoint match bit depth (8 -> 255, 16 -> 65535).
- Codestream (theirs: Kakadu 6.4): single tile, no MCT, 6 decomposition levels, RPCL,
  precincts, COM markers "MBF Bioscience", "MBF_DIMT:1", "MBF_DIMZ:1".
- NeuroInfo's own output is NOT ground truth: it duplicated channels ("Vgat_copy") and
  wrote 0.575 um for half-resolution pyramid data that is really 1.15 um.
- OPEN QUESTION: not yet confirmed that NeuroInfo displays our layer names/colors.
  Ask before assuming it works.

## Source data facts
- Leica TIFFs are 12-bit values (max 4095) in 16-bit files, so they look dark;
  the converter scales x16. Metadata/*.xlif confirms (`Resolution="12"`, `LUTName`).
- Pixel size comes from standard TIFF XResolution/ResolutionUnit (Leica writes
  numerator 4294967295, unit cm -> round to 5 decimals). ImageJ uses unit=1 plus
  `unit=micron` in ImageDescription. Values outside 0.05-50 um are rejected.
- "Pyramid" exports (`_pmd_0`...`_pmd_7`) are reduced-resolution 8-bit copies; even
  `_pmd_0` is 2x downsampled. Use the full-resolution folder instead.

## Encoder (imagecodecs / OpenJPEG)
- Single tile, single quality layer only. Quality is target PSNR in dB relative to
  full scale, so the same dB compresses 8-bit far less: defaults 50 dB (16-bit) and
  45 dB (8-bit) give ~20-25:1. No compression-ratio control; if tiling, multiple
  layers or a target ratio is needed, switch to glymur or Kakadu.

## Working with the data — IMPORTANT
- Data folders are OneDrive/SharePoint and most files are online-only placeholders.
  Opening ANY byte (even a TIFF header) downloads the whole file (200 MB-1.2 GB).
  Check the attribute first (`os.stat().st_file_attributes & 0x400000`, see
  `is_online_only`); never scan, preview or convert real folders casually.
  Free space again with `attrib -P +U <file>`. A careless test once pulled ~40 GB.
- Prefer small synthetic TIFFs (tifffile.imwrite) for tests.
- Don't open GUI windows during automated tests: construct `Jp2Gui`, call
  `root.withdraw()`, and drive it with `root.update()`.
- `gather_inputs()` saves `~/.tifs_to_jp2_settings.json`, the user's remembered GUI
  settings. Tests must not overwrite or delete it.

## Environment
- Windows; several Pythons installed, and `python` may be 3.15. The script
  auto-installs numpy/tifffile/imagecodecs (wheels only) into whichever Python runs it.
