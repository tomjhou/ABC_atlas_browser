# FileConverters

- `LifClass.py`, `LifConverterGUI.py`, `reader.py`, `basic_gui.py`, `LifConverterConsole.py`:
  the Leica .LIF to TIFF/JPG converter (copied from github.com/tomjhou/LIFconverter).
- `tifs_to_jp2.py`: per-channel Leica TIFFs to NeuroInfo JP2 (notes below).
- `tools/jp2_boxdump.py`: prints a JPEG2000 file's boxes, XMP and codestream header.
  Use it to compare our output with NeuroInfo's own files.
- `neuroinfo_stack.py` + `jpx_aligner.py` (GUI): re-align NeuroInfo serial section assembler
  stacks (stage 1 only; stage 2 atlas registration is edited in NeuroInfo). See below.

- `SerialAssembler.py` (start window) -> `section_loader.py` + `section_project.py` (step 1: find sections on .czi scans:
  threshold, outlines, order; project file `*_project.json` (older `*.sections.json` still open)) -> `jpx_aligner.py` with an in-memory Stack
  (`jpx_path == ""`, `Stack.unsaved`, sections rendered from the scans with `czi_origin = (0, 0)`; first export uses
  `write_jpx_new`). `j2k_partial.py` reads only the needed packets of NeuroInfo's JPEG 2000 frames. `section_order.py` is the thumbnail
  grid for ordering sections by dragging, shared by step 1's third tab and the aligner's "Thumbnails..." window.
  A `.jpx` can be added in step 1 on its own: a single-row "JPX project" (`Project.is_jpx`; `jpx_order` = frames in section order,
  `jpx_sections`), outlines tab skipped, order-tab thumbnails read from the JPX's own frames (`jpx_crop`, lazily, cached as `jt_*.npz`),
  Done -> `ns.open_stack(..., alignment=project.alignment)` + `sp.apply_jpx_order`. `stack_json` records each section's `frame`. The
  aligner's menu "Back to step 1" (`go_back_to_ordering`) saves the project and `SerialAssembler.detect_and_align` reopens the loader at
  the order tab (`SectionLoader.resume`); the alignment is kept. `neuroinfo_dialogs.py` holds the "which NeuroInfo file?" dialog.

# jpx_aligner / neuroinfo_stack

- NeuroInfo only reads the .jpx (confirmed by the user); .xml/.h5 are used only to recover
  NeuroInfo's alignment from a fresh NeuroInfo stack.
- Alignment recovery: fit case-file outline (source um) -> reconstruction.xml outline (stack um),
  matched by contour GUID and point index; exact (~0.004 um). Do NOT decode the CaseFile ITK
  transform chain: its Reflection2D steps don't match the real mirroring.
- Pixel convention, verified by rendering (corr ~0.997, offset < 0.4 px vs NeuroInfo frames):
  source (col,row) = (px*col, -px*row) um; canvas (u,v) = (px*u, -px*v) um; px = 0.575.
- Edit = rotation/shift/flip in stack um about the outline's AREA centroid (the pivot, stored per section
  in the JSON; files written earlier keep the pivot they stored, the mean of the outline's vertices), on top of the base
  transform. Exports carry a JSON description in a private 'uuid' box (ALIGNER_UUID) appended at
  the end of the file; all other boxes are copied verbatim from the opened file. Only sections whose
  edit changed are re-rendered (default 60 dB); others' codestreams are copied byte for byte.
- Canvas is fixed (NeuroInfo's size); export warns if a section extends past it.
- Previews are cached in %LOCALAPPDATA%\JhouLab\jpx_aligner_cache (keyed by path, size, mtime).
- Test GUI changes headless: build `Aligner()`, `root.withdraw()`, patch `canvas.winfo_width/height`
  and the dialogs, then grab `ImageTk.getimage(app.photo)`. Never show windows unannounced.
- CONFIRMED (2026-10-16): NeuroInfo 2026.1.1 opens JPX files written this way, including the
  aligner's uuid box at the end and OpenJPEG codestreams in place of Kakadu's.

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
- CONFIRMED (2026-10-15): NeuroInfo 2026.1.1 reads our layer names and colors. It built a
  Reconstruction.jpx from our 2-channel JP2s (Tyler, Sequence 001) and kept "gad2"/"foxp1"
  and green/red.

## NeuroInfo Reconstruction.jpx / .xml (aligned section stack)
- JPX (`ftyp` jpx, `rreq`), one `uuid` XMP for the whole stack, then per section a
  `jpch` + `jplh` (both empty) and a `jp2c` codestream. Every frame has the same canvas size
  (sections padded with zeros), 16-bit, Kakadu 20:1. Extract by walking the top-level boxes
  and passing each `jp2c` payload to `imagecodecs.jpeg2k_decode`.
- XMP `xmp:Images` has one entry per frame (Rotation/Translation 0: alignment is baked into
  the pixels; PixelWidth; SizeZ = section spacing in um). Application="NeuroInfo".
- Reconstruction.xml (MBF XML, appname NeuroInfo): `<images>` points at the .jpx (path may use
  a different drive letter); one `SectionOutline` `<contour>` per section, points in um with
  `sid="S<n>"`; `<filefacts>` lists sections with `top` (z, um) and cut thickness. Sections are
  renumbered 1..N, so they don't map by name to the source files.

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
