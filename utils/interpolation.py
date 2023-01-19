import numpy as np
import scipy.interpolate

INTERP_KIND = {2: "linear", 3: "quadratic", 4: "cubic"}


def interpolate_paths(z, x, y, rep_id):
    consecutive_year_blocks = np.where(np.diff(z) != 1)[0] + 1
    z_blocks = np.split(z, consecutive_year_blocks)
    x_blocks = np.split(x, consecutive_year_blocks)
    y_blocks = np.split(y, consecutive_year_blocks)
    paths = []
    for block_idx, zs in enumerate(z_blocks):
        if len(zs) > 1:
            kind = INTERP_KIND.get(len(zs), "cubic")
        else:
            rep_id_list = np.array([rep_id] * len(zs))
            paths.append(
                (zs, x_blocks[block_idx], y_blocks[block_idx], rep_id_list)
            )
            continue
        z = np.round(np.linspace(np.min(zs), np.max(zs), 20), 2)
        x = scipy.interpolate.interp1d(zs, x_blocks[block_idx], kind=kind)(z)
        y = scipy.interpolate.interp1d(zs, y_blocks[block_idx], kind=kind)(z)
        rep_id_list = np.array([rep_id] * len(z))
        paths.append((z, x, y, rep_id_list))
    return paths
