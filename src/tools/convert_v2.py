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
dicomsdl_dcm_paths = []
dicomsdl_save_paths = []
for i, row in df.iterrows():
    dcm_path = os.path.join(DCM_ROOT_DIR, str(row.patient_id),
                            f'{row.image_id}.dcm')
    save_path = os.path.join(SAVE_ROOT_DIR,
                                f'{row.patient_id}@{row.image_id}.{SAVE_EXT}')
    syntax_uid = machine_id_to_syntax_uid[row.machine_id]
    if syntax_uid == J2K_SYNTAX_UID:
        dicomsdl_dcm_paths.append(dcm_path)
        dicomsdl_save_paths.append(save_path)
    else:
        dicomsdl_dcm_paths.append(dcm_path)
        dicomsdl_save_paths.append(save_path)


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