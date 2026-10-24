"""The Allen CCF (2020) atlas drawn over the UMAP viewer's section maps, as the outlines of its regions.

The cells carry their position in the CCF (the registration the atlas ships), so each section's own (x, y) can be fitted to CCF
coordinates: a low-order polynomial for each section, which follows the section's tilt and warp (a section spans a few hundred
microns of the AP axis). Every pixel of a section map is then looked up in the 25 um annotation volume, and the boundary of each
region is traced and kept as a polygon in the section's own mm coordinates, so it can be drawn crisply at any zoom."""
import cv2
import numpy as np

VOXEL_MM = 0.025              # the annotation volume: [AP, DV, ML], 25 um voxels, labels are the cells' parcellation_index
FIT_DEGREE = 2
RASTER_MM = 0.0125            # the grid the regions are traced on (half a voxel)
SMOOTH_SIGMA_PX = 1.6         # blurs the voxel steps out of a region's mask before its boundary is traced
MIN_CONTOUR_PX = 6            # a traced boundary shorter than this is dropped


def poly_terms(x, y, center, scale):
    u, v = (x - center[0]) / scale, (y - center[1]) / scale
    return np.column_stack([u ** i * v ** j for i in range(FIT_DEGREE + 1) for j in range(FIT_DEGREE + 1 - i)])


def fit_section(x, y, ccf):
    """The fit of a section's CCF coordinates (n, 3: AP, DV, ML in mm) on its own (x, y): (center, scale, coefficients (terms, 3)), or None
    if too few cells have a CCF position."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ccf).all(axis=1)
    if ok.sum() < 200:
        return None
    x, y, ccf = x[ok], y[ok], ccf[ok]
    center = (float(x.mean()), float(y.mean()))
    scale = float(max(np.ptp(x), np.ptp(y), 1e-6))
    coef, *_ = np.linalg.lstsq(poly_terms(x, y, center, scale), ccf, rcond=None)
    return center, scale, coef


def trace_regions(volume, fit, home):
    """The boundaries of the CCF regions across the section map whose view is `home` (xmin, xmax, ymin, ymax; mm): a list of (n, 2)
    float arrays of (x, y) in mm, one closed outline each."""
    center, scale, coef = fit
    x0, x1, y0, y1 = home
    xs = np.arange(x0, x1, RASTER_MM) + RASTER_MM / 2
    ys = np.arange(y0, y1, RASTER_MM) + RASTER_MM / 2
    gx, gy = np.meshgrid(xs, ys)
    ccf = poly_terms(gx.ravel(), gy.ravel(), center, scale) @ coef
    idx = np.floor(ccf / VOXEL_MM).astype(np.int64)
    inside = np.ones(len(idx), bool)
    for axis, n in enumerate(volume.shape):
        inside &= (idx[:, axis] >= 0) & (idx[:, axis] < n)
    labels = np.zeros(len(idx), np.int64)
    labels[inside] = volume[idx[inside, 0], idx[inside, 1], idx[inside, 2]]
    labels = labels.reshape(gy.shape)
    polygons = []
    for region in np.unique(labels):
        if region == 0:
            continue
        mask = (labels == region).astype(np.uint8) * 255
        if SMOOTH_SIGMA_PX:
            mask = cv2.GaussianBlur(mask, (0, 0), SMOOTH_SIGMA_PX)
        mask = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 3 or cv2.arcLength(c, True) < MIN_CONTOUR_PX:
                continue
            p = c[:, 0, :].astype(np.float64)
            polygons.append(np.column_stack([x0 + (p[:, 0] + 0.5) * RASTER_MM, y0 + (p[:, 1] + 0.5) * RASTER_MM]))
    return polygons


def section_outlines(volume, cells, homes):
    """{section: polygons} for every section in `homes` ({section: its map's view}); `cells`: {section: (x, y, ccf (n, 3))}. A section with
    too few registered cells gets an empty list."""
    out = {}
    for section, home in homes.items():
        data = cells.get(section)
        fit = fit_section(*data) if data is not None else None
        out[section] = trace_regions(volume, fit, home) if fit is not None else []
    return out
