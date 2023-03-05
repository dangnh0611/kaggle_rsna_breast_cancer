from src.utils.dicom import J2K_SYNTAX_UID, convert_with_dali_parallel, convert_with_dicomsdl_parallel, make_uid_transfer_dict
import time
import pandas as pd
import os


CSV_PATH = '/home/dangnh36/datasets/.comp/rsna/train.csv'
DCM_ROOT_DIR = '/home/dangnh36/datasets/.comp/rsna/train_images'
SAVE_ROOT_DIR = '/home/dangnh36/datasets/.comp/rsna/exports/uint8_voilut_png'
SAVE_BACKEND = 'cv2'
SAVE_DTYPE = 'uint8'
SAVE_EXT = 'png'


df = pd.read_csv(CSV_PATH)
print('Total samples:', len(df))

os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

machine_id_to_syntax_uid = make_uid_transfer_dict(df, DCM_ROOT_DIR)
dali_dcm_paths = []
dali_save_paths = []
dicomsdl_dcm_paths = []
dicomsdl_save_paths = []
for i, row in df.iterrows():
    dcm_path = os.path.join(DCM_ROOT_DIR, str(row.patient_id),
                            f'{row.image_id}.dcm')
    save_path = os.path.join(SAVE_ROOT_DIR,
                                f'{row.patient_id}@{row.image_id}.{SAVE_EXT}')
    syntax_uid = machine_id_to_syntax_uid[row.machine_id]
    if syntax_uid == J2K_SYNTAX_UID:
        dali_dcm_paths.append(dcm_path)
        dali_save_paths.append(save_path)
    else:
        dicomsdl_dcm_paths.append(dcm_path)
        dicomsdl_save_paths.append(save_path)


# process with dali
print('Convert with DALI:', len(dali_dcm_paths))
start = time.time()
j2k_temp_dir = os.path.join(SAVE_ROOT_DIR, 'temp')
convert_with_dali_parallel(
    dali_dcm_paths,
    dali_save_paths,
    save_backend = SAVE_BACKEND,
    save_dtype = SAVE_DTYPE,
    chunk=64,  # disk
    batch_size=1,  # disk, ram_v3
    num_threads=2,
    py_num_workers=1,  # ram_v3
    device_id=[1, 1, 1, 1, 6, 6, 6, 6, 7, 7, 7, 7],
    cache='ram_v3',
    j2k_temp_dir=j2k_temp_dir,  # disk
    parallel_n_jobs=12,
    parallel_backend='loky')
end = time.time()
print(f'\n---DALI done in {end - start} sec.\n')


############ PROCESS WITH DICOMSDL
print('Convert with dicomsdl:', len(dicomsdl_dcm_paths))
start = time.time()
convert_with_dicomsdl_parallel(dicomsdl_dcm_paths,
                                dicomsdl_save_paths,
                                save_backend = SAVE_BACKEND,
                                save_dtype = SAVE_DTYPE,
                                parallel_n_jobs=16,
                                parallel_backend='loky')
end = time.time()
print(f'\n---Dicomsdl done in {end - start} sec.\n')