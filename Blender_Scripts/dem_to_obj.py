import numpy as np
import rasterio
from rasterio.windows import from_bounds

DEM_PATH = "/vsicurl/https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif"
OUT_OBJ = r"C:\Users\xavec\EGH490-2\lola_patch.obj"
SEARCH_M = 5000      # search within this distance of the pole (m)
PATCH_M = 400        # size of the terrain patch (m)

with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(DEM_PATH) as src:
    win = from_bounds(-SEARCH_M, -SEARCH_M, SEARCH_M, SEARCH_M, src.transform)
    z = src.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
    print(f"nodata={src.nodata}  NaN fraction={np.isnan(z).mean():.3f}  "
          f"z range {np.nanmin(z):.1f} to {np.nanmax(z):.1f} m")
    res = src.res[0]
    win_transform = src.window_transform(win)

# Slope (deg) per pixel, then mean slope over every PATCH-sized window (box filter via cumulative sums)
gy, gx = np.gradient(z, res)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
n = int(PATCH_M / res)

def box_sum(a):
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]

bad = np.isnan(slope)
mean_slope = box_sum(np.where(bad, 0, slope)) / n**2
mean_slope[box_sum(bad.astype(float)) > 0] = np.inf    # skip any window containing missing data
i, j = np.unravel_index(np.argmin(mean_slope), mean_slope.shape)
patch = z[i:i + n, j:j + n]
patch -= patch.mean()
cx, cy = win_transform * (j + n / 2, i + n / 2)
print(f"Patch mean slope {mean_slope[i, j]:.2f} deg, relief {patch.max() - patch.min():.1f} m, "
      f"centre (x, y) = ({cx:.0f}, {cy:.0f}) m polar stereographic, {np.hypot(cx, cy) / 1000:.2f} km from pole")

# Write a grid mesh centred on the origin (Z up, metres)
xs = (np.arange(n) - (n - 1) / 2) * res
with open(OUT_OBJ, "w") as f:
    for r in range(n):
        for col in range(n):
            f.write(f"v {xs[col]:.3f} {-xs[r]:.3f} {patch[r, col]:.3f}\n")
    for r in range(n - 1):
        for col in range(n - 1):
            a = r * n + col + 1
            f.write(f"f {a} {a + n} {a + n + 1} {a + 1}\n")
print(f"Saved {OUT_OBJ}")