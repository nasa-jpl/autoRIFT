# Generating Geocoded Master and Slave TIFFs for AutoRIFT from Sentinel-1 (ISCE2)

This guide addresses [nasa-jpl/autoRIFT#119](https://github.com/nasa-jpl/autoRIFT/issues/119)
and the related onboarding pain in [#96](https://github.com/nasa-jpl/autoRIFT/issues/96) and
[#110](https://github.com/nasa-jpl/autoRIFT/issues/110).

## Goal

Produce **geocoded master and slave GeoTIFFs** (Cartesian meters, X/Y) that `testGeogrid.py` and
`testautoRIFT.py` expect when running the radar-coordinate workflow for Sentinel-1 TOPS SLC pairs.

## Prerequisites

- ISCE2 installed with `topsApp.py` support
- A DEM in **Cartesian meters** (not lat/lon), same convention as Geogrid instructions
- Coregistered SLC pair processed through ISCE2 **mergeBursts** (topsApp)

## Recommended ISCE2 stopping point

For Sentinel-1 IW SLCs, run `topsApp.py` through the step that produces coregistered bursts, e.g.:

```bash
topsApp.py topsApp.xml --end=mergebursts
```

Typical output directories (names vary by ISCE2 version/config):

| Role | ISCE2 path (example) | Used for |
|------|----------------------|----------|
| Master geometry | `merged/reference/` or `merged/fine_coreg/` | Geogrid `-m` |
| Secondary metadata | `merged/secondary/` | Geogrid `-s` |
| DEM | `topo/dem_*.tif` (XY meters) | Geogrid `-d` |

> **Tip from maintainers:** For Geogrid, `-m` should point at the directory containing orbit
> geometry for the **coregistered overlap** (often `fine_coreg`), and `-s` at the secondary
> folder (date/time metadata only). See
> [autoRIFT#42](https://github.com/nasa-jpl/autoRIFT/issues/42#issuecomment-962075885).

## Step 1 — Run Geogrid (ISCE2)

```bash
python "$(python -c 'import isce2; import os; print(os.path.dirname(isce2.__file__))')/contrib/geo_autoRIFT/geogrid/testGeogrid_ISCE.py" \
  -m /path/to/merged/fine_coreg \
  -s /path/to/merged/secondary \
  -d /path/to/dem_xy_meters.tif
```

This writes `window_location.tif` and related Geogrid products in the working directory.

## Step 2 — Export master/slave GeoTIFFs

Geogrid produces radar-grid intermediates. For AutoRIFT CLI tests that take **two geocoded TIFF
inputs**, export the geocoded amplitude rasters from the Geogrid/ISCE2 tree. A common pattern
after a successful Geogrid run:

```bash
# Example names — adjust to your ISCE2 output tree
MASTER_TIF=/path/to/geogrid/master.tif
SLAVE_TIF=/path/to/geogrid/slave.tif
```

If your tree only has SLC stacks, use GDAL to materialize the geocoded rasters referenced by the
Geogrid metadata (or re-run Geogrid with explicit GeoTIFF outputs per your local ISCE2 recipe).

## Step 3 — Run AutoRIFT

```bash
python testGeogrid.py -m "$MASTER_TIF" -s "$SLAVE_TIF" -d /path/to/dem_xy_meters.tif
python testautoRIFT.py -m "$MASTER_TIF" -s "$SLAVE_TIF" -d /path/to/dem_xy_meters.tif
```

`testautoRIFT.py` should emit `offset.tif` in the working directory for image-grid runs (see #110).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `min() arg is an empty sequence` in testGeogrid_ISCE | Wrong `-m`/`-s` folders (GRD vs SLC, wrong merge step) | Point `-m` at `fine_coreg`, `-s` at `secondary` |
| Geogrid accepts GRD but AutoRIFT fails | GRD not supported for full geocoded radar workflow | Use SLC through mergeBursts |
| No `offset.tif` after testautoRIFT | Image-grid path not writing output (regression) | See fix in #110 / PR for offset.tif write |

## References

- [Geogrid instructions](https://github.com/leiyangleon/Geogrid/blob/master/docs/instruction.md)
- [AutoRIFT instructions](https://github.com/leiyangleon/Geogrid/blob/master/docs/instruction.md) (paired workflow)
- Maintainer guidance on S1 SLC vs GRD: [autoRIFT#96](https://github.com/nasa-jpl/autoRIFT/issues/96#issuecomment-1960952306)
