from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


def approximate_sky_view_factor(
    nodes: pd.DataFrame,
    height_raster,
    *,
    node_crs: str = "EPSG:25830",
    max_radius_m: float = 200.0,
    radial_step_m: float = 5.0,
    azimuth_count: int = 36,
    observer_height_m: float = 1.5,
) -> pd.Series:
    """Aproxima SVF mediante el ángulo máximo de horizonte en radios equiespaciados.

    El raster debe contener alturas de edificios relativas al suelo. La aproximación
    usa ``mean(cos(horizon_angle)**2)`` y devuelve valores limitados a [0, 1].
    """
    distances = np.arange(radial_step_m, max_radius_m + radial_step_m, radial_step_m)
    azimuths = np.linspace(0, 2 * np.pi, azimuth_count, endpoint=False)
    values: list[float] = []
    with rasterio.open(height_raster) as src:
        transformer = Transformer.from_crs(node_crs, src.crs, always_xy=True)
        nodata = src.nodata
        for row in nodes.itertuples():
            horizons = []
            for angle in azimuths:
                xs = row.x + distances * np.sin(angle)
                ys = row.y + distances * np.cos(angle)
                if str(src.crs) != node_crs:
                    xs, ys = transformer.transform(xs, ys)
                samples = np.asarray([sample[0] for sample in src.sample(zip(xs, ys))], dtype=float)
                if nodata is not None:
                    samples[np.isclose(samples, nodata)] = 0.0
                samples[~np.isfinite(samples)] = 0.0
                heights = np.maximum(samples - observer_height_m, 0.0)
                horizons.append(float(np.max(np.arctan2(heights, distances))))
            svf = float(np.mean(np.cos(np.asarray(horizons)) ** 2))
            values.append(float(np.clip(svf, 0.0, 1.0)))
    return pd.Series(values, index=nodes.index, name="sky_view_factor")
