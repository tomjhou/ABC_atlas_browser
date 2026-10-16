"""Region surface meshes for an Allen-style atlas, in the format of the meshes_coronal folder of MBF Bioscience's
Allen2017 atlas (NeuroInfo): one legacy-VTK file per region, '<acronym>_<id>.vtk' (a '/' in the acronym becomes '_'),
ASCII, version 4.2, POINTS then TRIANGLE_STRIPS, coordinates in um.

How the existing meshes relate to the annotation volume (checked on Allen2017):
  * a point (x, y, z) um is voxel index (x, y, z) * voxel size, with the NRRD's fastest-varying axis as x (left-right),
    then y (dorsal-ventral), then z (anterior-posterior); surfaces made this way from the volume lie within 7-19 um of
    Allen's own meshes (under one 25 um voxel), closer than with the voxel-centre convention;
  * a region's mesh covers its voxels and those of all its descendants (ACAd has no voxels of its own; its mesh wraps
    its layers);
  * not every region has one: Allen's set leaves out e.g. the layer 6b regions and some small tracts. New regions (ids
    absent from the reference ontology) get meshes here; the selection of the others is kept.

Used by the CCF browser's "Generate meshes" button; also usable from the command line:
    python atlas_meshes.py <annotation.nrrd> <ontology.csv> <reference atlas folder> <output meshes folder>
"""
import csv
import gzip
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

MESH_FOLDER_NAME = 'meshes_coronal'
# Surface settings, chosen to match Allen's meshes (see the module docstring)
SMOOTH_ITERATIONS = 20
SMOOTH_PASS_BAND = 0.05
DECIMATE_REDUCTION = 0.85   # Allen's meshes have about 1/7 of the marching-cubes vertices
# Pieces of a region smaller than this (face-connected voxels) are left out of its surface: a lone voxel, e.g. a stray
# brush click, would become a tiny cube that smoothing and simplification turn into a scrap with holes.
MIN_PIECE_VOXELS = 8


class Canceled(Exception):
    pass


# ---- reading -------------------------------------------------------------------------------------------------------
def read_nrrd_labels(path):
    """(labels as stored: [z, y, x] with x fastest, voxel size in um) of an annotation .nrrd with an axis-aligned,
    isotropic grid."""
    raw = Path(path).read_bytes()
    end = raw.find(b'\n\n')
    if not raw.startswith(b'NRRD') or end < 0:
        raise ValueError(f"{Path(path).name} is not an NRRD file with an attached header")
    fields = {}
    for line in raw[:end].decode('latin-1').splitlines()[1:]:
        if line.startswith('#') or ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        fields[key.strip().lower()] = value.strip()
    kinds = {'unsigned int': 'u4', 'uint': 'u4', 'uint32': 'u4', 'uint32_t': 'u4', 'int': 'i4', 'int32': 'i4',
             'unsigned short': 'u2', 'ushort': 'u2', 'uint16': 'u2', 'short': 'i2', 'int16': 'i2',
             'unsigned char': 'u1', 'uchar': 'u1', 'uint8': 'u1'}
    kind = kinds.get(fields.get('type', '').lower())
    if kind is None:
        raise ValueError(f"unsupported NRRD voxel type {fields.get('type')!r}")
    sizes = [int(v) for v in fields['sizes'].split()]
    if len(sizes) != 3:
        raise ValueError("expected a 3-D volume")
    order = '>' if fields.get('endian', 'little') == 'big' else '<'
    body = raw[end + 2:]
    encoding = fields.get('encoding', 'raw').lower()
    if encoding in ('gzip', 'gz'):
        body = gzip.decompress(body)
    elif encoding != 'raw':
        raise ValueError(f"unsupported NRRD encoding {encoding!r}")
    labels = np.frombuffer(body, order + kind).reshape(sizes[::-1])
    voxel = 25.0
    directions = re.findall(r'\(([^)]*)\)', fields.get('space directions', ''))
    if directions:
        matrix = np.array([[float(v) for v in d.split(',')] for d in directions])
        diag = np.diag(matrix)
        if np.abs(matrix - np.diag(diag)).max() > 1e-6 or np.ptp(np.abs(diag)) > 1e-6:
            raise ValueError("the volume's grid is rotated or not isotropic; meshes need an axis-aligned grid")
        voxel = float(abs(diag[0]))
    return labels, voxel


def read_ontology(path):
    """{id: {'acronym', 'name', 'path': [ids from the root]}} from an Allen-style ontology CSV."""
    regions = {}
    with open(path, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                rid = int(row['id'])
            except (KeyError, ValueError):
                continue
            ids = [int(v) for v in (row.get('structure_id_path') or '').strip('/').split('/') if v.strip().isdigit()]
            regions[rid] = {'acronym': row.get('acronym') or str(rid), 'name': row.get('name') or '',
                            'path': ids or [rid]}
    if not regions:
        raise ValueError(f"no regions found in {Path(path).name}")
    return regions


def find_ontology(folder):
    found = sorted(Path(folder).glob('*ontology*.csv'))
    return found[0] if found else None


def find_annotation(folder):
    """The annotation volume of an MBF atlas folder (e.g. AllenCoronal_25um_anatomy.nrrd), or None."""
    found = sorted(Path(folder).glob('*anatomy*.nrrd'))
    return found[0] if found else None


def mesh_file_name(acronym, rid):
    return re.sub(r'[\\/:*?"<>|]', '_', acronym) + f'_{rid}.vtk'


def mesh_id(file_name):
    m = re.search(r'_(\d+)\.vtk$', file_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ---- planning ------------------------------------------------------------------------------------------------------
def descendants(ontology):
    """{id: set of the id and all its descendants}."""
    out = {rid: {rid} for rid in ontology}
    for rid, region in ontology.items():
        for ancestor in region['path']:
            if ancestor in out:
                out[ancestor].add(rid)
    return out


def plan(annotation_path, ontology_path, reference_folder, output_folder, copy_existing=True):
    """What generate() will do, as a dict: 'new' [(id, acronym, name, voxels)], 'no_voxels' [(id, acronym)] for new
    regions with nothing to mesh, 'copy' (number of reference meshes to copy), 'stale' [(acronym, id, changed voxels)]
    for existing meshes whose region's voxels changed (left as they are), plus the inputs."""
    labels, voxel = read_nrrd_labels(annotation_path)
    ontology = read_ontology(ontology_path)
    reference_folder = Path(reference_folder)
    ref_meshes = reference_folder / MESH_FOLDER_NAME if (reference_folder / MESH_FOLDER_NAME).is_dir() \
        else reference_folder
    mesh_files = {mesh_id(f.name): f for f in ref_meshes.glob('*.vtk') if mesh_id(f.name) is not None}
    if not mesh_files:
        raise ValueError(f"no reference meshes (<acronym>_<id>.vtk) found in {ref_meshes}")
    ref_ontology_path = find_ontology(ref_meshes.parent if ref_meshes.name == MESH_FOLDER_NAME else reference_folder)
    if ref_ontology_path is None:
        raise ValueError(f"no ontology*.csv in the reference atlas folder {ref_meshes.parent}")
    ref_ontology = read_ontology(ref_ontology_path)

    ids, counts = np.unique(labels, return_counts=True)
    voxels = dict(zip(ids.tolist(), counts.tolist()))
    unknown = sorted(set(voxels) - set(ontology) - {0})
    tree = descendants(ontology)
    new, no_voxels, specks = [], [], []
    for rid in sorted(set(ontology) - set(ref_ontology)):
        n = sum(voxels.get(d, 0) for d in tree[rid])
        if n:
            new.append((rid, ontology[rid]['acronym'], ontology[rid]['name'], n))
            dropped = small_piece_voxels(labels, tree[rid])
            if dropped:
                specks.append((ontology[rid]['acronym'], rid, dropped))
        else:
            no_voxels.append((rid, ontology[rid]['acronym']))

    stale = []
    ref_annotation = find_annotation(ref_meshes.parent)
    if ref_annotation is not None:
        try:
            old, _ = read_nrrd_labels(ref_annotation)
        except Exception:
            old = None
        if old is not None and old.shape == labels.shape:
            changed = old != labels
            if changed.any():
                touched = set(np.unique(old[changed]).tolist()) | set(np.unique(labels[changed]).tolist())
                for rid, f in sorted(mesh_files.items()):
                    if rid not in tree or not (tree[rid] & touched):
                        continue
                    members = np.array(sorted(tree[rid]))
                    was = np.isin(old[changed], members)
                    now = np.isin(labels[changed], members)
                    diff = int((was != now).sum())            # voxels entering or leaving the mesh's region
                    own = int(((old[changed] == rid) != (labels[changed] == rid)).sum())   # the id's own voxels
                    if diff or own:
                        stale.append((ontology[rid]['acronym'], rid, diff, own))
    output_folder = Path(output_folder)
    to_copy = 0
    if copy_existing and output_folder.resolve() != ref_meshes.resolve():
        to_copy = sum(1 for f in mesh_files.values() if not (output_folder / f.name).exists())
    return {'labels': labels, 'voxel_um': voxel, 'ontology': ontology, 'tree': tree, 'new': new,
            'no_voxels': no_voxels, 'specks': specks, 'unknown_ids': unknown, 'stale': stale, 'copy': to_copy,
            'reference_meshes': ref_meshes, 'mesh_files': mesh_files, 'output': output_folder,
            'copy_existing': copy_existing}


def describe_plan(p):
    lines = [f"New regions to mesh: {len(p['new'])}"]
    lines += [f"   {acr} ({rid}) {name}: {n:,} voxels -> {mesh_file_name(acr, rid)}" for rid, acr, name, n in p['new']]
    if p['specks']:
        lines.append(f"Isolated bits under {MIN_PIECE_VOXELS} voxels, left out of the surface (still in the volume; "
                     "e.g. stray paint clicks):")
        lines += [f"   {acr} ({rid}): {n} voxel{'s' if n != 1 else ''}" for acr, rid, n in p['specks']]
    if p['no_voxels']:
        lines.append("New regions with no voxels in the volume (no mesh): "
                     + ", ".join(f"{a} ({r})" for r, a in p['no_voxels']))
    if p['copy_existing']:
        lines.append(f"Existing meshes copied from {p['reference_meshes']}: {p['copy']}"
                     + ("" if p['copy'] else " (all already in the output folder)"))
    if p['stale']:
        lines.append("Existing regions that changed (their meshes are kept as they are). A mesh covers the region and "
                     "all its sub-regions, so only voxels entering or leaving that whole family move its surface:")
        for acr, rid, mesh_diff, own in p['stale']:
            line = f"   {acr} ({rid}): surface affected by {mesh_diff:,} voxel{'s' if mesh_diff != 1 else ''}"
            if own and own != mesh_diff:
                line += f"; {own:,} of its own voxels relabeled"
                if own > mesh_diff:
                    line += " (mostly to or from its sub-regions, still inside its mesh)"
            lines.append(line)
    if p['unknown_ids']:
        lines.append(f"Region values in the volume missing from the ontology: {p['unknown_ids'][:12]}")
    lines.append(f"Output folder: {p['output']}")
    return "\n".join(lines)


# ---- meshing -------------------------------------------------------------------------------------------------------
def region_mask(labels, ids):
    """(mask cropped with a margin, its [z, y, x] offset, voxels left out as small pieces) for the voxels in `ids`."""
    from scipy import ndimage
    mask = np.isin(labels, np.fromiter(ids, dtype=np.int64))
    z, y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("no voxels")
    lo = np.maximum(np.array([z.min(), y.min(), x.min()]) - 2, 0)
    hi = np.array([z.max(), y.max(), x.max()]) + 3
    sub = np.zeros(hi - lo, np.uint8)   # an empty margin all round, so the surface closes
    part = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub[:part.shape[0], :part.shape[1], :part.shape[2]] = part
    pieces, n = ndimage.label(sub)       # face-connected
    sizes = np.bincount(pieces.ravel())
    small = [k for k in range(1, n + 1) if sizes[k] < MIN_PIECE_VOXELS]
    dropped = int(sizes[small].sum()) if small else 0
    if small and dropped < sub.sum():
        sub[np.isin(pieces, small)] = 0
    else:
        dropped = 0                      # nothing but small pieces: mesh them anyway
    return sub, lo, dropped


def small_piece_voxels(labels, ids):
    return region_mask(labels, ids)[2]


def make_mesh(labels, ids, voxel_um):
    """vtkPolyData surface of the voxels whose value is in `ids`, smoothed and simplified like Allen's meshes."""
    import vtk
    from vtk.util import numpy_support
    sub, lo, _dropped = region_mask(labels, ids)
    image = vtk.vtkImageData()
    image.SetDimensions(sub.shape[2], sub.shape[1], sub.shape[0])
    image.SetSpacing(voxel_um, voxel_um, voxel_um)
    image.SetOrigin(lo[2] * voxel_um, lo[1] * voxel_um, lo[0] * voxel_um)
    image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(sub.ravel(), deep=True,
                                                               array_type=vtk.VTK_UNSIGNED_CHAR))
    surface = vtk.vtkDiscreteFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetValue(0, 1)
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(surface.GetOutputPort())
    smooth.SetNumberOfIterations(SMOOTH_ITERATIONS)
    smooth.SetPassBand(SMOOTH_PASS_BAND)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    # Topology-preserving simplification: keeps the surface closed (quadric decimation opened small holes in
    # ACB_core), removing vertices only where that is safe, so it may stop short of the target in places.
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(smooth.GetOutputPort())
    decimate.SetTargetReduction(DECIMATE_REDUCTION)
    decimate.PreserveTopologyOn()
    decimate.SplittingOff()
    decimate.BoundaryVertexDeletionOff()
    strips = vtk.vtkStripper()
    strips.SetInputConnection(decimate.GetOutputPort())
    clean = vtk.vtkPolyData()
    strips.Update()
    clean.SetPoints(strips.GetOutput().GetPoints())
    clean.SetStrips(strips.GetOutput().GetStrips())   # points and strips only, as in Allen's files
    return clean


def write_mesh(polydata, path):
    import vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetFileTypeToASCII()
    writer.SetFileVersion(42)
    if not writer.Write():
        raise OSError(f"could not write {path}")


def generate(p, progress=None, stop=None):
    """Carry out a plan(): copy the reference meshes (if asked) and write the new regions' meshes. Returns the list of
    files written for new regions."""
    out = p['output']
    out.mkdir(parents=True, exist_ok=True)
    if p['copy_existing'] and out.resolve() != p['reference_meshes'].resolve():
        files = [f for f in p['mesh_files'].values() if not (out / f.name).exists()]
        for k, f in enumerate(files, 1):
            if stop is not None and stop():
                raise Canceled
            shutil.copy2(f, out / f.name)
            if progress and (k % 20 == 0 or k == len(files)):
                progress(f"copying existing meshes {k}/{len(files)}")
    written = []
    for k, (rid, acr, _name, _n) in enumerate(p['new'], 1):
        if stop is not None and stop():
            raise Canceled
        if progress:
            progress(f"meshing {acr} ({k}/{len(p['new'])})")
        mesh = make_mesh(p['labels'], p['tree'][rid], p['voxel_um'])
        target = out / mesh_file_name(acr, rid)
        tmp = target.with_suffix('.partial')
        write_mesh(mesh, tmp)
        os.replace(tmp, target)
        written.append(target)
    return written


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    p = plan(*argv[1:5])
    print(describe_plan(p))
    for f in generate(p, progress=lambda t: print("  " + t, flush=True)):
        print("wrote", f)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
