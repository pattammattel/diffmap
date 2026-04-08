"""
Standalone module for exporting scan metadata/details to CSV.

This module provides functionality to export scan metadata from the databroker
without exporting detector data. It's useful for creating beamline logbooks.

Author: HXN Beamline
Date: 2026-04-08
"""

import os
import csv
import datetime
import getpass
import warnings
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import hxntools with fallback
try:
    from hxntools.CompositeBroker import db
    from hxntools.scan_info import get_scan_positions
except ImportError:
    print("Trying overlay environment...")
    sys.path.insert(0, '/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
    try:
        from hxntools.CompositeBroker import db
        from hxntools.scan_info import get_scan_positions
    except ImportError:
        db = None
        get_scan_positions = None
        print("Offline analysis; hxntools not found")

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def get_scan_details(hdr):
    """
    Extract scan metadata/parameters from a scan header.
    
    Parameters
    ----------
    hdr : Header
        Scan header from databroker
        
    Returns
    -------
    dict
        Dictionary containing scan metadata fields
    """
    start_doc = hdr.start
    param_dict = {"scan_id": start_doc.get("scan_id")}
    
    # 2D_FLY_PANDA logic
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        df = db.get_table(hdr, stream_name="baseline")
        mots = start_doc['motors']
        
        # Create a datetime object from the Unix time
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc["motors"]
        
        if "detectors" in start_doc.keys():
            param_dict["detectors"] = start_doc["detectors"]
            param_dict["scan_start1"] = start_doc["scan_start1"]
            param_dict["num1"] = start_doc["num1"]
            param_dict["scan_end1"] = start_doc["scan_end1"]
            if len(mots) == 2:
                param_dict["scan_start2"] = start_doc["scan_start2"]
                param_dict["scan_end2"] = start_doc["scan_end2"]
                param_dict["num2"] = start_doc["num2"]
            param_dict["exposure_time"] = start_doc["exposure_time"]
        elif "scan" in start_doc.keys():
            param_dict["scan"] = start_doc["scan"]
        
        param_dict["zp_theta"] = np.round(df.zpsth.iloc[0], 3)
        param_dict["mll_theta"] = np.round(df.dsth.iloc[0], 3)
        param_dict["energy"] = np.round(df.energy.iloc[0], 3)
        return param_dict
    
    # rel_scan logic
    elif start_doc.get('plan_name') == 'rel_scan':
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc.get("motors", [])
        param_dict["detectors"] = start_doc.get("detectors", [])
        param_dict["num_points"] = start_doc.get("num_points", None)
        param_dict["num_intervals"] = start_doc.get("num_intervals", None)
        param_dict["plan_args"] = start_doc.get("plan_args", {})
        param_dict["plan_type"] = start_doc.get("plan_type", None)
        param_dict["plan_name"] = start_doc.get("plan_name", None)
        param_dict["scan_name"] = start_doc.get("scan_name", None)
        param_dict["sample"] = start_doc.get("sample", None)
        param_dict["PI"] = start_doc.get("PI", None)
        param_dict["experimenters"] = start_doc.get("experimenters", None)
        param_dict["shape"] = start_doc.get("shape", None)
        return param_dict
    
    # Fallback for other scan types
    else:
        print("[SCAN META] Not all metadata was exported; fallback option used")
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc.get("motors", [])
        param_dict["detectors"] = start_doc.get("detectors", [])
        return param_dict


def get_scan_metadata(hdr):
    """
    Get full scan metadata including baseline table.
    
    Parameters
    ----------
    hdr : Header
        Scan header from databroker
        
    Returns
    -------
    pd.DataFrame
        Concatenated baseline data and scan details
    """
    output = db.get_table(hdr, stream_name="baseline")
    df_dictionary = pd.DataFrame([get_scan_details(hdr)])
    output = pd.concat([output, df_dictionary], ignore_index=True)
    return output


def export_scan_details_batch(
    sid_list,
    wd=".",
    return_dataframe=False
):
    """
    Export scan metadata/details for a list of scan IDs to CSV.
    
    This function extracts metadata from scans without loading detector data,
    making it fast and efficient for creating beamline logbooks.
    
    Parameters
    ----------
    sid_list : list of int or int
        List of scan IDs to export, or a single scan ID
    wd : str, optional
        Working directory where CSV will be saved (default: ".")
    return_dataframe : bool, optional
        If True, returns a pandas DataFrame of the results (default: False)
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame of scan details if return_dataframe=True, else None
    
    Notes
    -----
    - Writes a CSV file named "scan_details_<first>_to_<last>_<timestamp>.csv"
    - Handles errors gracefully and logs them in the CSV
    - All fields from get_scan_details() are exported
    - Nested dictionaries and lists are converted to strings for CSV compatibility
    
    Example
    -------
    >>> # Export metadata for multiple scans
    >>> export_scan_details_batch([200001, 200002, 200003], wd="/data/exports")
    
    >>> # Get DataFrame of results
    >>> df = export_scan_details_batch([200001, 200002], return_dataframe=True)
    
    >>> # Export a single scan
    >>> export_scan_details_batch(200001)
    """
    # Normalize to list
    if isinstance(sid_list, (int, float)):
        sid_list = [int(sid_list)]
    
    first_sid = sid_list[0]
    last_sid = sid_list[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"scan_details_{first_sid}_to_{last_sid}_{timestamp}.csv"
    log_path = os.path.join(wd, log_filename)
    
    all_rows = []
    all_fields = set(['scan_id'])
    os_user = os.getlogin() if hasattr(os, 'getlogin') else getpass.getuser()
    
    for sid in tqdm(sid_list, desc="Exporting scan details"):
        try:
            hdr = db[int(sid)]
            # Get full metadata including baseline table
            metadata_df = get_scan_metadata(hdr)
            
            # Convert DataFrame to dictionary (take first row if multiple)
            if len(metadata_df) > 0:
                row = metadata_df.iloc[0].to_dict()
                # Ensure scan_id is present
                if "scan_id" not in row:
                    row["scan_id"] = sid
            else:
                row = {"scan_id": sid}
            
            # Handle nested dicts and lists by converting to strings
            for key, value in list(row.items()):
                if isinstance(value, dict):
                    row[key] = str(value)
                elif isinstance(value, (list, tuple)):
                    row[key] = str(value)
                elif pd.isna(value):
                    row[key] = None
            
            all_rows.append(row)
            all_fields.update(row.keys())
            
        except Exception as e:
            error_msg = f"Error processing scan {sid}: {e}"
            print(error_msg)
            all_rows.append({
                "scan_id": sid,
                "error": str(e),
                "os_user": os_user
            })
            all_fields.update(["error", "os_user"])
    
    # Ensure all rows have all fields
    all_fields = sorted(list(all_fields))
    for row in all_rows:
        for field in all_fields:
            if field not in row:
                row[field] = None
    
    # Write to CSV
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    
    print(f"✅ Scan details exported to: {log_path}")
    
    if return_dataframe:
        return pd.DataFrame(all_rows)
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("Scan Metadata Export Tool")
    print("=" * 70)
    print("\n✅ Module loaded successfully!")
    print("\n📘 Usage:")
    print("   >>> from export_scan_details import export_scan_details_batch")
    print("   >>> export_scan_details_batch([200001, 200002, 200003])")
    print("\n📘 Functions available:")
    print("   • export_scan_details_batch(sid_list, wd='.', return_dataframe=False)")
    print("       → Export scan metadata to CSV (no detector data)")
    print("   • get_scan_details(hdr)")
    print("       → Extract metadata dictionary from a scan header")
    print("   • get_scan_metadata(hdr)")
    print("       → Get full metadata DataFrame including baseline")
    print("\n" + "=" * 70)
