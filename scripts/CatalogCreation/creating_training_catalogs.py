import numpy as np
import pandas as pd
import h5py
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sklearn.model_selection import train_test_split
from dustmaps.sfd import SFDQuery


# Load merged catalog
data_path = '/pscratch/sd/a/ashodkh/image_photo_z/'
merged_df = pd.read_csv(f'{data_path}merged-candels-catalog-for-cutouts.csv')

# Filter objects with valid flux values in all bands
valid_flux_mask = (
    (merged_df['ACS_F606W_FLUX'] > 0) &
    (merged_df['ACS_F814W_FLUX'] > 0) &
    (merged_df['WFC3_F125W_FLUX'] > 0) &
    (merged_df['WFC3_F160W_FLUX'] > 0)
)
merged_df = merged_df[valid_flux_mask]

# Extract COSMOS field data
merged_df_cos = merged_df[merged_df['field'] == 'cos']
cos2020_data = pd.read_pickle(f'{data_path}cosmos2020-selected-data.pkl')

# Match CANDELS COSMOS with COSMOS2020 catalog
candels_coords = SkyCoord(ra=merged_df_cos['RA'], dec=merged_df_cos['DEC'], unit='deg')
cos2020_coords = SkyCoord(ra=cos2020_data['ALPHA_J2000'], dec=cos2020_data['DELTA_J2000'], unit='deg')
idx, d2d, _ = match_coordinates_sky(candels_coords, cos2020_coords)

# Apply selection criteria
h_mag = 23.9 - 2.5 * np.log10(merged_df_cos['WFC3_F160W_FLUX'].values)
selection_mask = (
    (h_mag < 25) &
    (d2d.arcsec < 0.4) &
    (cos2020_data['lp_zBEST'][idx] > 0) & np.isfinite(cos2020_data['lp_zBEST'][idx])
)

# Define fields and filters
filters = ['f606w', 'f814w', 'f125w', 'f160w']
instruments = ['acs', 'acs', 'wfc3', 'wfc3']
fields = ['cos', 'uds', 'egs', 'gn', 'gs']
px, py = 108, 108  # Image dimensions
load_images = True  # Set to True if images should be loaded


photoz_filters = ['ACS_F606W_FLUX', 'ACS_F814W_FLUX', 'WFC3_F125W_FLUX', 'WFC3_F160W_FLUX']
for field_index, field in enumerate(fields):
    field_df = merged_df[merged_df['field'].values == field]
    
    field_names = field_df['field']
    field_fluxes = field_df[[filter_name for filter_name in photoz_filters]].values
    field_colors = np.log10(field_fluxes[:,1:]) - np.log10(field_fluxes[:,:-1])
    field_color_features = np.concatenate((field_colors, np.log10(field_fluxes[:,1]).reshape(-1,1)), axis=1)
    coords = SkyCoord(field_df['RA'], field_df['DEC'], unit='deg')
    sfd = SFDQuery()
    ebvs = sfd(coords)

    batch = 128
    n = len(field_df)

    n_batches = n//batch

    # Load images if enabled
    if load_images:
        images_all, ids_all = [], []
        for i in range(len(field_df) // 128 + 1):
            images = []
            for j, filter_name in enumerate(filters):
                file_path = f'{data_path}candels_cutouts/selected_positive_fluxes/{field}/{instruments[j]}_{filter_name}_cutouts{i}.hdf5'
                with h5py.File(file_path, 'r') as f:
                    images.append(f['cutouts'][:])
                    if j == 0:
                        ids_all.append(f['ids'][:].squeeze())
            images_all.append(np.stack(images, axis=1))
        images_all = np.concatenate(images_all)
        ids_all = np.concatenate(ids_all, dtype='int')
        assert (field_df['ID'].values == ids_all).all()

    redshifts = -1 * np.ones(len(field_df))
    redshift_errs = -1 * np.ones(len(field_df))
    
    z_spec_s = field_df['z_spec'].values>0
    z_grism_s = field_df['z_grism'].values>0
    z_grism_s[z_spec_s] = False
    print(f'number of spec_zs in field {field} is {np.sum(z_spec_s)}')
    print(f'number of grism_zs in field {field} is {np.sum(z_grism_s)}')
    
    redshifts[z_grism_s] = field_df['z_grism'].values[z_grism_s]
    redshift_errs[z_grism_s] = 0.05
    
    redshifts[z_spec_s] = field_df['z_spec'].values[z_spec_s]
    redshift_errs[z_spec_s] = 0.01
    
    redshift_weights = 1/redshift_errs**2
    redshift_weights[np.logical_not(z_spec_s)*(field_df['z_grism'].values<0)] = 0

    
    if field == 'cos':
        cos_extra_spec_z_inds = (cos2020_data['spec_zs'][idx] > 0) * (cos2020_data['spec_zs_flags'][idx] >= 3)
        overlap_inds = cos_extra_spec_z_inds * z_spec_s
        cos_extra_spec_z_inds[overlap_inds] = False
        
        redshifts[cos_extra_spec_z_inds] = cos2020_data['spec_zs'][idx][cos_extra_spec_z_inds]
        redshift_errs[cos_extra_spec_z_inds] = 0.01
        redshift_weights[cos_extra_spec_z_inds] = 1/0.01**2

        print('number of extra spec_zs in field', field, np.sum(cos_extra_spec_z_inds))
        
        selection_mask[redshifts >0] = False
        redshifts[selection_mask] = cos2020_data['lp_zBEST'].values[idx][selection_mask]
        redshift_errs[selection_mask] = cos2020_data['lp_zPDF_u68'].values[idx][selection_mask] - cos2020_data['lp_zPDF_l68'].values[idx][selection_mask]
        redshift_weights[selection_mask] = 1/redshift_errs[selection_mask]**2
    
    if field_index == 0:
        if load_images:
            images_all_fields = images_all
            ids_all_fields = ids_all
        f160w_mags_all_fields = 23.9 - 2.5*np.log10(field_df['WFC3_F160W_FLUX'].values)
        
        redshifts_all_fields = redshifts
        redshift_errs_all_fields = redshift_errs
        redshift_weights_all_fields = redshift_weights
        color_features_all_fields = field_color_features
        ebvs_all_fields = ebvs
        fluxes_all_fields = field_fluxes
        field_names_all_fields = field_names
    else:
        if load_images:
            images_all_fields = np.concatenate((images_all_fields, images_all), axis=0, dtype='float16')
            ids_all_fields = np.concatenate((ids_all_fields, ids_all), dtype='int')
        f160w_mags_all_fields = np.concatenate((f160w_mags_all_fields,23.9 - 2.5*np.log10(field_df['WFC3_F160W_FLUX'].values)), axis=0)
        
        redshifts_all_fields = np.concatenate((redshifts_all_fields, redshifts), axis=0, dtype='float16')
        redshift_errs_all_fields = np.concatenate((redshift_errs_all_fields, redshift_errs), axis=0, dtype='float16')
        redshift_weights_all_fields = np.concatenate((redshift_weights_all_fields, redshift_weights), axis=0, dtype='float16')
        
        color_features_all_fields = np.concatenate((color_features_all_fields, field_color_features), axis=0, dtype='float16')
        ebvs_all_fields = np.concatenate((ebvs_all_fields, ebvs), dtype='float16')
        fluxes_all_fields = np.concatenate((fluxes_all_fields, field_fluxes), dtype='float16')
        field_names_all_fields = np.concatenate((field_names_all_fields, field_names))

if load_images:
    s_zeros = np.zeros(images_all_fields.shape[0], dtype='bool')
    zeros_true = np.zeros(images_all_fields.shape[0], dtype='bool')
    for j in range(len(filters)):
        for i in range(images_all_fields.shape[0]):
            if np.all(images_all_fields[i,j,:,:] == 0):
                zeros_true[i] = 1
        s_zeros = np.logical_or(s_zeros, zeros_true)
    print(f'number of zero images: {int(zeros_true.sum())}/{len(zeros_true)}')
    s_mag = f160w_mags_all_fields < 26
    print(f'number of images after removing zeros and mag cut: {np.sum(s_mag*np.logical_not(s_zeros))}/{images_all_fields.shape[0]}')

    s_combined = s_mag * np.logical_not(s_zeros)
    inds = np.argwhere(s_combined).squeeze()
else:
    s_mag = f160w_mags_all_fields < 26
    print(f'number of images after mag cut: {np.sum(s_mag)}')

    s_combined = s_mag
    inds = np.argwhere(s_combined).squeeze()

if load_images:
    image_quantiles = np.nanquantile(images_all_fields[inds,:,:,:], q=(0.05,0.5,0.95), axis=(0,2,3))

if load_images:
    for j in range(4):
        inf_inds = np.where(np.isinf(images_all_fields[:,j,:,:]))
        images_all_fields[:,j,:,:][inf_inds] = image_quantiles[2,j]

z_inds = np.argwhere((redshifts_all_fields > 0) * np.isfinite(redshift_weights_all_fields)).squeeze()
train_inds, not_train_inds = train_test_split(z_inds, train_size=0.7, random_state=42)
val_inds, test_inds = train_test_split(not_train_inds, train_size=0.5, random_state=42)

use_redshift = np.zeros(len(redshifts_all_fields), dtype=int)
use_redshift[train_inds] = 1

s_mag[train_inds] = 1

custom_ids_all_fields = np.arange(len(redshifts_all_fields))

if load_images:
    f_bright = h5py.File(f'{data_path}candels_combined_catalog_bright_with_train.hdf5', 'w')
    dset = f_bright.create_dataset('images', data=images_all_fields[s_mag,:,:,:])
    dset = f_bright.create_dataset('redshifts', data=redshifts_all_fields[s_mag])
    dset = f_bright.create_dataset('redshift_weights', data=redshift_weights_all_fields[s_mag])
    dset = f_bright.create_dataset('redshift_errs', data=redshift_errs_all_fields[s_mag])
    dset = f_bright.create_dataset('use_redshift', data=use_redshift[s_mag])
    dset = f_bright.create_dataset('f160w_mags', data=f160w_mags_all_fields[s_mag])
    dset = f_bright.create_dataset('color_features', data=color_features_all_fields[s_mag])
    dset = f_bright.create_dataset('ebvs', data=ebvs_all_fields[s_mag])
    dset = f_bright.create_dataset('fluxes', data=fluxes_all_fields[s_mag])
    dset = f_bright.create_dataset('field_names', data=field_names_all_fields[s_mag])
    f_bright.close()
else:
    f_bright = h5py.File(f'{data_path}candels_combined_catalog_bright.hdf5', 'r+')
    # data = f_bright['use_redshift']
    # data[:] = use_redshift[s_mag]
    # dset = f_bright.create_dataset('field_names', data=field_names_all_fields[s_mag])
    #del f_bright['custom_ids']
    dset = f_bright.create_dataset('custom_ids', data=custom_ids_all_fields[s_mag])
    f_bright.close()

    inds_list = [train_inds, val_inds, test_inds]
inds_names = ['train', 'val', 'test']

if load_images:
    for i in range(len(inds_list)):
        f = h5py.File(f'{data_path}candels_{inds_names[i]}_catalog.hdf5', 'w')
        inds = inds_list[i]
        dset = f.create_dataset('images', data=images_all_fields[inds,:,:,:])
        dset = f.create_dataset('redshifts', data=redshifts_all_fields[inds])
        dset = f.create_dataset('redshift_weights', data=redshift_weights_all_fields[inds])
        dset = f.create_dataset('redshift_errs', data=redshift_errs_all_fields[inds])
        dset = f.create_dataset('f160w_mags', data=f160w_mags_all_fields[inds])
        dset = f.create_dataset('color_features', data=color_features_all_fields[inds])
        dset = f.create_dataset('ebvs', data=ebvs_all_fields[inds])
        dset = f.create_dataset('fluxes', data=fluxes_all_fields[inds])
        f.close()
else:
    for i in range(len(inds_list)):
        f = h5py.File(f'{data_path}candels_{inds_names[i]}_catalog.hdf5', 'r+')
        inds = inds_list[i]
        #dset = f.create_dataset('use_redshift', data=np.ones(len(inds), dtype=int))
        #dset = f.create_dataset('field_names', data=field_names_all_fields[inds])
        dset = f.create_dataset('custom_ids', data=custom_ids_all_fields[inds])
        f.close()
    