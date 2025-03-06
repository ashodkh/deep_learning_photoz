import numpy as np
import pandas as pd
from astropy.io import fits

# Source catalog file paths
catalog_files = {
    "cosmos": "./candels-catalogs/hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1-1photom_cat.fits",
    "egs": "./candels-catalogs/hlsp_candels_hst_wfc3_egs-tot-multiband_f160w_v1-1photom_cat.fits",
    "goodsn": "./candels-catalogs/hlsp_candels_hst_wfc3_goodsn-barro19_multi_v1-1_photometry-cat.fits",
    "goodss": "./candels-catalogs/hlsp_candels_hst_wfc3_goodss-tot-multiband_f160w_v1_cat.fits",
    "uds": "./candels-catalogs/hlsp_candels_hst_wfc3_uds-tot-multiband_f160w_v1_cat.fits",
}


# Load source catalogs
def load_fits_data(file_path):
    with fits.open(file_path) as hdul:
        return hdul[1].data


photometric_data = {field: load_fits_data(path) for field, path in catalog_files.items()}

# Print dataset sizes
print({field: len(data["ID"]) for field, data in photometric_data.items()})

# Load photometric redshift catalogs
cols_cat = ['file', 'ID', 'RA', 'DEC', 'z_best', 'z_best_type', 'z_spec', 'z_spec_ref', 'z_grism', 'mFDa4_z_peak', 'mFDa4_z_weight', 'mFDa4_z683_low', 'mFDa4_z683_high', 'mFDa4_z954_low', 'mFDa4_z954_high', 'HB4_z_peak', 'HB4_z_weight', 'HB4_z683_low', 'HB4_z683_high', 'HB4_z954_low', 'HB4_z954_high', 'Finkelstein_z_peak', 'Finkelstein_z_weight', 'Finkelstein_z683_low', 'Finkelstein_z683_high', 'Finkelstein_z954_low', 'Finkelstein_z954_high', 'Fontana_z_peak', 'Fontana_z_weight', 'Fontana_z683_low', 'Fontana_z683_high', 'Fontana_z954_low', 'Fontana_z954_high', 'Pforr_z_peak', 'Pforr_z_weight', 'Pforr_z683_low', 'Pforr_z683_high', 'Pforr_z954_low', 'Pforr_z954_high', 'Salvato_z_peak', 'Salvato_z_weight', 'Salvato_z683_low', 'Salvato_z683_high', 'Salvato_z954_low', 'Salvato_z954_high', 'Wiklind_z_peak', 'Wiklind_z_weight', 'Wiklind_z683_low', 'Wiklind_z683_high', 'Wiklind_z954_low', 'Wiklind_z954_high', 'Wuyts_z_peak', 'Wuyts_z_weight', 'Wuyts_z683_low', 'Wuyts_z683_high', 'Wuyts_z954_low', 'Wuyts_z954_high',]
photoz_catalogs = {
    field: pd.read_csv(f"./candels-catalogs/hlsp_candels_hst_wfc3_{field}_multi_v2_redshift-cat.txt",
                       names=cols_cat, skiprows=61, sep='\s+')
    for field in catalog_files.keys()
}

# Compute total spec-z and grism-z counts
total_spec_z = sum((df["z_spec"] > 0).sum() for df in photoz_catalogs.values())
total_grism_z = sum((df["z_grism"] > 0).sum() for df in photoz_catalogs.values())
print(f"Total Spec-z: {total_spec_z}, Total Grism-z: {total_grism_z}")

# Goodsn photoz catalog has 6 more galaxies so i have to match the catalogs
goodsn_sorter = np.argsort(photoz_catalogs["goodsn"]["ID"].values)
goodsn_inds = np.searchsorted(photoz_catalogs["goodsn"]["ID"].values, photometric_data["goodsn"]["ID"], sorter=goodsn_sorter)

# Define common columns
common_columns = ["ID", "RA", "DEC", "ACS_F606W_FLUX", "ACS_F814W_FLUX", "WFC3_F125W_FLUX", "WFC3_F160W_FLUX"]
common_columns_uds = ["ID", "RA", "DEC", "Flux_F606W_hst", "Flux_F814W_hst", "Flux_F125W_hst", "Flux_F160W_hst"]


# initialize dictionaries for all fields with spec-z's and grism-z's
# all catalogs are already matched except for goodsn, which has 5 more galaxies
# so need to use goodsn_inds
def create_dataframe(field, common_cols, data, photoz):
    field_dict = {
        "field": [field] * len(data),
        "z_spec": photoz["z_spec"].values,
        "z_grism": photoz["z_grism"].values,
    }
    if field == "goodsn":
        field_dict["z_spec"] = photoz["z_spec"].values[goodsn_inds]
        field_dict["z_grism"] = photoz["z_grism"].values[goodsn_inds]
    
    for i, col in enumerate(common_cols):
        field_dict[col] = data[common_columns_uds[i]] if field == "uds" else data[col]
    
    return pd.DataFrame(field_dict)


# Construct merged dataframe
merged_df = pd.concat([
    create_dataframe(field, common_columns, photometric_data[field], photoz_catalogs[field])
    for field in catalog_files.keys()
], ignore_index=True)

# Save merged catalog
output_path = "./candels-catalogs/merged-candels-catalog-for-cutouts.csv"
merged_df.to_csv(output_path, index=False)
print(f"Merged catalog saved to {output_path}")
