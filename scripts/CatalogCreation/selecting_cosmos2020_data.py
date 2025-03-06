from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.sfd import SFDQuery
import extinction


# Define file paths
classic_file_path = "./additional-catalogs/cosmos2020/COSMOS2020_CLASSIC_R1_v2.2_p3.fits"
specz_file_path = "./additional-catalogs/cosmos2020/specz_compilation_COSMOS_January24_v1.04_unique.fits"

classic_data = Table.read(classic_file_path, hdu=1)

# combine with spec-z's
cos2020_speczs = fits.open(specz_file_path)
s_pos_ids = np.where(cos2020_speczs[1].data['Id Classic'] > 0)
sorter = np.argsort(classic_data['ID'])
speczs_inds = sorter[np.searchsorted(classic_data['ID'], cos2020_speczs[1].data['Id Classic'][s_pos_ids], sorter=sorter)]

speczs = -1 * np.ones(len(classic_data))
speczs[speczs_inds] = cos2020_speczs[1].data['spec-z'][s_pos_ids]
specz_flags = -1 * np.ones(len(classic_data))
specz_flags[speczs_inds] = cos2020_speczs[1].data['flag'][s_pos_ids]

cos2020_speczs.close()

classic_data['spec_zs'] = speczs
classic_data['spec_zs_flags'] = specz_flags

# Define filters and columns to keep
filters = ['CFHT_ustar', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y',
           'UVISTA_Y', 'UVISTA_J', 'UVISTA_H', 'UVISTA_Ks', 'IRAC_CH1', 'IRAC_CH2', 'ACS_F814W']

central_wavelengths = np.array([3858, 4847, 6219, 7699, 8894, 9761, 10216, 
                                12525, 16466, 21557, 35686, 45067, 8333])

columns_to_keep = ['ID', 'ALPHA_J2000', 'DELTA_J2000', 'lp_zBEST', 'lp_zPDF_l68', 'lp_zPDF_u68',
                   'ACS_A_WORLD', 'ACS_B_WORLD', 'ACS_THETA_WORLD', 'ACS_FWHM_WORLD', 
                   'ACS_MU_MAX', 'ACS_MU_CLASS', 'spec_zs', 'spec_zs_flags']

# Include flux and magnitude columns dynamically
for f in filters:
    flux_col = f + '_FLUX_AUTO' if f + '_FLUX_AUTO' in classic_data.columns else f + '_FLUX'
    mag_col = f + '_MAG_AUTO' if f + '_MAG_AUTO' in classic_data.columns else f + '_MAG'
    err_flux_col = flux_col.replace('FLUX', 'FLUXERR')
    err_mag_col = mag_col.replace('MAG', 'MAGERR')

    columns_to_keep.extend([flux_col, err_flux_col, mag_col, err_mag_col])


df_selected = classic_data[columns_to_keep].to_pandas()

coords = SkyCoord(classic_data['ALPHA_J2000'], classic_data['DELTA_J2000'], unit='deg')
ebvs = SFDQuery()(coords)
extinction_in_mags = np.zeros((len(ebvs), len(central_wavelengths)))
for i in range(len(ebvs)):
    extinction_in_mags[i,:] = extinction.fm07(central_wavelengths, 3.1*ebvs[i])

corrected_mags = {
    f"{f}_ext_corrected_mags": (classic_data[f"{f}_MAG_AUTO"] if f"{f}_MAG_AUTO" in classic_data.columns else classic_data[f"{f}_MAG"]) + extinction_in_mags[:, i]
    for i, f in enumerate(filters)
}

df_ext_corrected = pd.DataFrame(corrected_mags)
df_ebvs = pd.DataFrame({'ebvs': ebvs})
df_selected = pd.concat([df_selected, df_ext_corrected, df_ebvs], axis=1)

df_selected.to_pickle('cosmos2020-selected-data.pkl')