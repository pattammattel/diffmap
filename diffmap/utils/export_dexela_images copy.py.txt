import os
import h5py
import numpy as np


def _as_numpy(data):
    """Convert iterable data to a numpy array when possible."""
    try:
        arr = np.asarray(data)
        if arr.dtype == object:
            arr = np.stack(list(data))
        return arr
    except Exception:
        return np.array(list(data), dtype=object)


def _write_dataset(group, name, data, compression="gzip", compression_opts=4):
    """Write numeric arrays efficiently; fall back to UTF-8 strings if needed."""
    if not isinstance(data, np.ndarray):
        data = _as_numpy(data)

    if isinstance(data, np.ndarray) and data.dtype != object:
        group.create_dataset(
            name,
            data=data,
            compression=compression,
            compression_opts=compression_opts,
            chunks=True,
        )
        return

    # Fallback for strings / mixed objects
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = [str(x) for x in data.tolist()]
    else:
        data = [str(x) for x in data]

    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.array(data, dtype=dt))


def export_scan_to_hdf5(scan_id, working_dir, fields=None, xrf_func=None, stream_name="primary"):
    """
    Export databroker scan data to HDF5.

    Parameters
    ----------
    scan_id : int or str
        Scan number or UID.
    working_dir : str
        Output directory.
    fields : dict
        Example:
        {
            "images": ["dexela1_image"],
            "motors": ["zpsth", "zpssx"],
            "scalars": ["I0", "It"]
        }
    xrf_func : callable or None
        Function like: xrf_stack, elem_list = get_all_xrf_roi_data(hdr)
    stream_name : str
        Databroker stream name, usually "primary".


    usage
    fields = {
    "images": ["dexela1_image"],
    "motors": ["zpsth", "zpssx", "zpssy"],
    }

    export_scan_to_hdf5(
        scan_id=395854,
        working_dir="/nsls2/data2/hxn/proposals/2026-1/pass-319117/diffraction/",
        fields=fields,
        xrf_func=get_all_xrf_roi_data,
    )
    """
    global db

    if fields is None:
        fields = {}

    h = db[scan_id]

    # Pick a detector name for the filename
    det_name = "data"
    if "images" in fields and fields["images"]:
        det_name = fields["images"][0]
    else:
        # fall back to the first requested field name
        for group_name, keys in fields.items():
            if keys:
                det_name = keys[0]
                break

    det_name = det_name.replace("_image", "")
    filename = f"scan_{scan_id}_{det_name}.h5"
    filepath = os.path.join(working_dir, filename)

    with h5py.File(filepath, "w") as f:
        # ------------------------
        # Metadata
        # ------------------------
        meta_grp = f.create_group("metadata")
        for k, v in h.start.items():
            try:
                meta_grp.attrs[k] = v
            except Exception:
                meta_grp.attrs[k] = str(v)

        # ------------------------
        # Regular databroker fields
        # ------------------------
        for group_name, keys in fields.items():
            grp = f.create_group(group_name)

            for key in keys:
                try:
                    data = list(h.data(key, stream_name=stream_name))
                    data = _as_numpy(data)
                    _write_dataset(grp, key, data)
                except Exception as e:
                    print(f"[SKIP] {key}: {e}")

        # ------------------------
        # XRF ROI data
        # ------------------------
        if xrf_func is not None:
            try:
                xrf_stack, elem_list = xrf_func(h)

                xrf_grp = f.create_group("xrf")
                _write_dataset(xrf_grp, "roi_stack", xrf_stack)

                dt = h5py.string_dtype(encoding="utf-8")
                xrf_grp.create_dataset(
                    "element_names",
                    data=np.array(elem_list, dtype=dt),
                )
            except Exception as e:
                print(f"[SKIP] XRF export failed: {e}")

    print(f"[EXPORT] Saved: {filepath}")
    return filepath