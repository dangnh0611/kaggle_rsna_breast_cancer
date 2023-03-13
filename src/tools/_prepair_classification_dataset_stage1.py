import os
import shutil
import subprocess as sp
import sys

import numpy as np
import pandas as pd
from pydicom import dcmread
from tqdm import tqdm

from settings import SETTINGS
from src.utils.misc import make_symlink, rm_and_mkdir

__all__ = [
    'stage1_process_rsna', 'stage1_process_vindr', 'stage1_process_miniddsm',
    'stage1_process_cmmd', 'stage1_process_cddcesm', 'stage1_process_bmcd'
]


def get_dicom_meta_from_file(dicom_root_dir,
                             patient_id,
                             img_id,
                             extension='dcm'):
    dicom_path = os.path.join(dicom_root_dir,
                              f'{patient_id}/{img_id}.{extension}')
    return dcmread(dicom_path, stop_before_pixels=True)


def get_dicom_meta(dicom_root_dir, dicom_df, extension='dcm', verbose=False):
    dicom_data = dict()
    keywords = set()
    dicom_df = dicom_df[["patient_id",
                         "image_id"]].copy().reset_index(drop=True)

    for i in tqdm(range(len(dicom_df))):
        row = dicom_df.loc[i]
        patient_id = row["patient_id"]
        image_id = row["image_id"]
        dicom = get_dicom_meta_from_file(dicom_root_dir, patient_id, image_id,
                                         extension)
        assert dicom.get('ModalityLUTSequence', None) is None
        if patient_id not in dicom_data:
            dicom_data[patient_id] = dict()
        if image_id not in dicom_data[patient_id]:
            dicom_data[patient_id][image_id] = dict()
        for feature in dicom.iterall():
            dicom_data[patient_id][image_id][feature.keyword] = feature.value
            keywords.add(feature.keyword)
    for keyword in keywords:
        dicom_df[keyword] = dicom_df[["patient_id", "image_id"]].apply(
            lambda x: np.nan if keyword not in dicom_data[x.patient_id][
                x.image_id] else dicom_data[x.patient_id][x.image_id][keyword],
            axis=1)
    if verbose:
        print(": Keywords extracted from dicom files:")
        for keyword in keywords:
            print("--> {}".format(keyword))
    return dicom_df


########################### COMPETITION DATA ###########################
def stage1_process_rsna(raw_root_dir,
                        stage1_images_dir,
                        cleaned_label_path,
                        force_copy=False):
    del force_copy
    dcm_root_dir = os.path.join(raw_root_dir, 'train_images')
    df = pd.read_csv(os.path.join(raw_root_dir, 'train.csv'))
    dicom_df = get_dicom_meta(dcm_root_dir, df, extension='dcm')
    dicom_df.drop(columns='patient_id', inplace=True)
    dicom_df.rename(columns={
        name: '__' + name
        for name in dicom_df.columns if name != 'image_id'
    },
                    inplace=True)
    dicom_df['__WindowCenterList'] = dicom_df['__WindowCenter'].apply(
        lambda x: [float(x)]
        if isinstance(x, float) else [float(e) for e in x])
    dicom_df['__WindowCenterListLength'] = dicom_df[
        '__WindowCenterList'].apply(len)
    dicom_df['__WindowWidthList'] = dicom_df['__WindowWidth'].apply(
        lambda x: [float(x)]
        if isinstance(x, float) else [float(e) for e in x])
    dicom_df['__WindowWidthListLength'] = dicom_df['__WindowWidthList'].apply(
        len)
    merged = pd.merge(
        df,
        dicom_df,
        how="inner",
        on='image_id',
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("", "__"),
        copy=True,
        indicator=False,
        validate='1:1',
    ).reset_index(drop=True)
    merged.to_csv(cleaned_label_path, index=False)
    make_symlink(dcm_root_dir, stage1_images_dir)


########################### VINDR ###########################
def stage1_process_vindr(raw_root_dir,
                         stage1_images_dir,
                         cleaned_label_path,
                         force_copy=False):
    del force_copy
    dcm_root_dir = os.path.join(raw_root_dir, 'images')
    df = pd.read_csv(os.path.join(raw_root_dir,
                                  'breast-level_annotations.csv'))
    df.rename(columns={
        'study_id': 'patient_id',
        'view_position': 'view',
        'breast_birads': 'BIRADS',
        'breast_density': 'density'
    },
              inplace=True)
    df.BIRADS = df.BIRADS.apply(lambda x: int(x.split()[-1]))
    df.density = df.density.apply(lambda x: x.split()[-1]
                                  if type(x) == str else x)
    # read Dicom metadata
    dicom_df = get_dicom_meta(dcm_root_dir, df, extension='dicom')
    # meta_df.to_csv(os.path.join(dataset_root_dir, 'dicom_meta_only.csv'), index = False)
    dicom_df.rename(columns={
        name: '__' + name
        for name in dicom_df.columns if name != 'image_id'
    },
                    inplace=True)
    # merge label df with dicom metadata
    merged = pd.merge(
        df,
        dicom_df,
        how="inner",
        on='image_id',
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("??", "__"),
        copy=True,
        indicator=False,
        validate='1:1',
    )
    merged['__hasVOILUTSequence'] = merged['__VOILUTSequence'].isna()
    merged['__hasLUTDescriptor'] = merged['__LUTDescriptor'].isna()
    merged['__hasLUTData'] = merged['__LUTData'].isna()
    merged.rename(columns={'__PatientAge': 'age'}, inplace=True)
    # ignore some unused columns
    merged = merged[[
        'patient_id', 'series_id', 'image_id', 'laterality', 'view', 'age',
        'BIRADS', 'density', 'split', '__SmallestImagePixelValue',
        '__PixelRepresentation', '__SOPInstanceUID', '__PatientWeight',
        '__WindowWidth', '__PixelPaddingValue', '__BodyPartExamined',
        '__StudyInstanceUID', '__ViewPosition', '__LargestImagePixelValue',
        '__PixelSpacing', '__PixelPaddingRangeLimit', '__SeriesInstanceUID',
        '__SamplesPerPixel', '__PresentationLUTShape',
        '__WindowCenterWidthExplanation', '__WindowCenter',
        '__ImagerPixelSpacing', '__VOILUTFunction', '__RescaleIntercept',
        '__RescaleSlope', '__PhotometricInterpretation', '__Rows', '__Columns',
        '__hasVOILUTSequence', '__hasLUTDescriptor', '__hasLUTData'
    ]]

    merged['cancer'] = merged.BIRADS.apply(lambda x: 1 if x == 5 else 0)
    merged.to_csv(cleaned_label_path, index=False)
    make_symlink(dcm_root_dir, stage1_images_dir)


########################### MINI-DDSM ###########################
def stage1_process_miniddsm(raw_root_dir,
                            stage1_images_dir,
                            cleaned_label_path,
                            force_copy=False):
    df = pd.read_excel(
        os.path.join(raw_root_dir, 'Data-MoreThanTwoMasks',
                     'Data-MoreThanTwoMasks.xlsx'))
    df['patient_id'] = df['fileName'].apply(lambda x: x.split('.')[0])
    df['image_id'] = df['patient_id'] + '|' + df['Side'] + '|' + df['View']

    # make cancer dict at breast-level (laterality-level)
    cancer_dict = {}
    for group_name, sub_df in tqdm(df.groupby(['patient_id', 'Side'])):
        breast = '|'.join(group_name)
        sub_df = sub_df.reset_index(drop=True)
        assert sub_df.Status.nunique() == 1
        status = sub_df.at[0, 'Status']
        no_annotation = True
        for j in range(len(sub_df)):
            if type(sub_df.at[j, 'Tumour_Contour']) == str:
                if len(sub_df.at[j, 'Tumour_Contour']) > 5:
                    no_annotation = False
        if status == 'Normal':
            if not no_annotation:
                raise AssertionError('status=normal but has annotation.')
            else:
                cancer_dict[breast] = 0
        elif status == 'Benign':
            cancer_dict[breast] = 0
        elif status == 'Cancer':
            # cancer is labeled at patient-level, not laterality-level
            # cancer patient + no annotation on that side --> normal
            # cancer patient + has annotation on that side --> cancer
            if no_annotation:
                cancer_dict[breast] = 0
            else:
                cancer_dict[breast] = 1
        else:
            raise AssertionError(f'Invalid status=`{status}`')

    df['cancer'] = -1
    for i in tqdm(range(len(df))):
        breast = df.at[i, 'patient_id'] + '|' + df.at[i, 'Side']
        df.at[i, 'cancer'] = cancer_dict[breast]

    # create symlink to re-structure the dataset
    src_dir = os.path.join(raw_root_dir, 'MINI-DDSM-Complete-PNG-16')
    dst_dir = stage1_images_dir
    for i in tqdm(range(len(df))):
        status = df.at[i, 'Status']
        patient_id = df.at[i, 'patient_id']
        patient_idx = patient_id.split('_')[1]
        image_id = df.at[i, 'image_id']
        name = df.at[i, 'fileName']
        src_path = os.path.join(src_dir, status, patient_idx, name)
        dst_path = os.path.join(dst_dir, patient_id, f'{image_id}.png')
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if force_copy:
            shutil.copy2(src_path, dst_path)
        else:
            make_symlink(src_path, dst_path)

    # LEFT, RIGHT --> L, R
    df['Side'] = df['Side'].apply(lambda x: x[0])
    density_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 0: 'A'}
    df['Density'] = df['Density'].apply(lambda x: density_map[x])
    df.drop(columns=[
        'fullPath', 'fileName', 'Tumour_Contour', 'Tumour_Contour2',
        'Tumour_Contour3', 'Tumour_Contour4', 'Tumour_Contour5',
        'Tumour_Contour6'
    ],
            inplace=True)
    df.rename(columns={
        'View': 'view',
        'Side': 'laterality',
        'Status': 'ddsm_ori_status',
        'Age': 'age',
        'Density': 'density',
    },
              inplace=True)
    df = df[[
        'patient_id', 'image_id', 'view', 'laterality', 'density', 'age',
        'ddsm_ori_status', 'cancer'
    ]]
    df.to_csv(cleaned_label_path, index=False)


########################### CMMD ###########################
def stage1_process_cmmd(raw_root_dir,
                        stage1_images_dir,
                        cleaned_label_path,
                        force_copy=False):
    dcm_root_dir = os.path.join(raw_root_dir, 'CMMD')
    # the provided xlsx file can not be read with pd.read_excel()
    # df = pd.read_excel(
    #     os.path.join(raw_root_dir, 'CMMD_clinicaldata_revision.xlsx'), engine = 'openpyxl')
    df = pd.read_csv(os.path.join(raw_root_dir, 'label.csv'))

    src_dir = dcm_root_dir
    dst_dir = stage1_images_dir

    for patient_id, sub_df in tqdm(df.groupby('ID1')):
        tmp_dir = os.path.join(src_dir, patient_id)
        tmp2_dirs = os.listdir(tmp_dir)
        assert len(tmp2_dirs) == 1
        tmp3_dirs = os.listdir(os.path.join(tmp_dir, tmp2_dirs[0]))
        assert len(tmp3_dirs) == 1
        src_img_dir = os.path.join(tmp_dir, tmp2_dirs[0], tmp3_dirs[0])
        img_names = os.listdir(src_img_dir)
        for img_name in img_names:
            assert img_name.split('.')[-1] == 'dcm'
            src_img_path = os.path.join(src_img_dir, img_name)
            dst_img_path = os.path.join(dst_dir, patient_id,
                                        f'{patient_id}_{img_name}')
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            if force_copy:
                shutil.copy2(src_img_path, dst_img_path)
            else:
                make_symlink(src_img_path, dst_img_path)

    all_patient_ids = []
    all_image_ids = []
    patients = os.listdir(dst_dir)
    for patient in patients:
        patient_dir = os.path.join(dst_dir, patient)
        image_names = os.listdir(patient_dir)
        for image_name in image_names:
            all_patient_ids.append(patient)
            all_image_ids.append(image_name.split('.dcm')[0])
    dicom_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'image_id': all_image_ids
    })
    dicom_df = get_dicom_meta(dst_dir, dicom_df, extension='dcm')

    df.rename(columns={
        'ID1': 'patient_id',
        'LeftRight': 'laterality'
    },
              inplace=True)
    dicom_df.rename(columns={'ImageLaterality': 'laterality'}, inplace=True)
    dicom_df.rename(columns={
        name: '__' + name
        for name in dicom_df.columns
        if name not in ['patient_id', 'laterality']
    },
                    inplace=True)
    merged = pd.merge(
        dicom_df,
        df,
        how="outer",
        on=['patient_id', 'laterality'],
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("__", ""),
        copy=True,
        indicator=False,
        #     validate='1:1',
    )
    merged.classification.fillna('Normal', inplace=True)
    merged['cancer'] = merged.classification.apply(lambda x: 1
                                                   if x == 'Malignant' else 0)
    merged = merged[[
        'patient_id', '__image_id', 'laterality', 'Age', 'cancer', 'number',
        'abnormality', 'classification', 'subtype', '__VOILUTFunction',
        '__WindowCenter', '__WindowWidth', '__WindowCenterWidthExplanation',
        '__RescaleSlope', '__RescaleIntercept', '__Columns', '__Rows',
        '__SOPInstanceUID', '__LossyImageCompression', '__CodeMeaning',
        '__StudyID', '__PatientOrientation', '__SpecificCharacterSet',
        '__InstanceCreatorUID', '__PositionerType',
        '__PhotometricInterpretation', '__BitsAllocated', '__StudyDate',
        '__PatientName', '__ContentTime', '__PixelIntensityRelationship',
        '__Modality', '__PresentationIntentType', '__AcquisitionTime',
        '__BodyPartExamined', '__HighBit', '__PatientIdentityRemoved',
        '__BitsStored', '__InstanceCreationTime', '__StudyTime',
        '__SeriesTime', '__PatientBirthDate', '__ImageType', '__RescaleType',
        '__SeriesInstanceUID', '__ReferringPhysicianName', '__ContentDate',
        '__InstanceCreationDate', '__PixelIntensityRelationshipSign',
        '__SeriesNumber', '__StudyInstanceUID', '__PatientID',
        '__DetectorType', '__SamplesPerPixel', '__SOPClassUID',
        '__PixelRepresentation', '__CodeValue', '__OrganExposed',
        '__InstanceNumber', '__AccessionNumber', '__SeriesDate',
        '__AcquisitionDate', '__PatientAge', '__BurnedInAnnotation',
        '__PresentationLUTShape', '__Manufacturer', '__DeidentificationMethod',
        '__ImagerPixelSpacing', '__PatientSex'
    ]]
    merged.rename(columns={
        '__image_id': 'image_id',
        'Age': 'age',
        'number': '_num_images',
    },
                  inplace=True)
    for _name, sub_df in merged.groupby(['patient_id', 'laterality']):
        assert sub_df.cancer.nunique() == 1

    merged.sort_values(by=['patient_id', 'image_id'], inplace=True)
    merged.to_csv(cleaned_label_path, index=False)


########################### CDD-CESM ###########################
def stage1_process_cddcesm(raw_root_dir,
                           stage1_images_dir,
                           cleaned_label_path,
                           force_copy=False):
    df = pd.read_excel(os.path.join(raw_root_dir,
                                    'Radiology manual annotations.xlsx'),
                       sheet_name='all')
    df['image_id'] = df['Image_name'].apply(
        lambda x: x.replace('DM_', '').replace('CM_', ''))
    for _image_id, sub_df in df.groupby('image_id'):
        sub_df = sub_df.reset_index(drop=True)
        if sub_df.at[0, 'Pathology Classification/ Follow up'] == 'Malignant':
            assert sub_df['Pathology Classification/ Follow up'].nunique(
            ) == 1, sub_df
    # filter: Digital Mammograms only
    df = df[df.Type == 'DM'].reset_index(drop=True)
    df.rename(columns={
        'Image_name': 'image_name',
        'Patient_ID': 'patient_id',
        'Side': 'laterality',
        'Age': 'age',
        'Breast density (ACR)': 'density',
        'Findings': 'findings',
        'View': 'view',
        'Tags': 'tags',
        'Machine': 'machine_id',
        'Pathology Classification/ Follow up': 'classification',
    },
              inplace=True)
    df = df[[
        'patient_id', 'image_id', 'laterality', 'view', 'image_name', 'age',
        'density', 'BIRADS', 'findings', 'tags', 'machine_id',
        'classification', 'Type'
    ]]
    df['cancer'] = df.classification.apply(lambda x: 1
                                           if x == 'Malignant' else 0)
    # some name contain unexpected space
    df['image_id'] = df['image_id'].apply(lambda x: x.strip())

    # create stage1_images in well-defined structure
    for i in tqdm(range(len(df))):
        patient_id = str(df.at[i, 'patient_id'])
        image_id = df.at[i, 'image_id']
        image_name = image_id.replace('L_', 'L_DM_').replace('R_',
                                                             'R_DM_') + '.jpg'
        patient_dir = os.path.join(stage1_images_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        src_path = os.path.join(raw_root_dir, 'Low energy images of CDD-CESM',
                                image_name)
        dst_path = os.path.join(patient_dir, image_id + '.jpg')
        if force_copy:
            shutil.copy2(src_path, dst_path)
        else:
            make_symlink(src_path, dst_path)

    df.to_csv(cleaned_label_path, index=False)


########################### BMCD ###########################
def stage1_process_bmcd(raw_root_dir,
                        stage1_images_dir,
                        cleaned_label_path,
                        force_copy=False):
    dcm_root_dir = os.path.join(raw_root_dir, 'Dataset')
    df = pd.read_csv(os.path.join(raw_root_dir, 'label.csv'))
    # biopsy = 'NAN' for normal cases
    df.biopsy_result.fillna('NAN', inplace=True)
    df = df[df.biopsy_result.isin(['NAN', 'BENIGN', 'DCIS',
                                   'MALIGNANT'])].reset_index(drop=True)

    src_root_dir = dcm_root_dir
    dst_root_dir = stage1_images_dir

    df['view'] = None
    df['image_id'] = None
    all_series = []
    for i in tqdm(range(len(df))):
        dir_type = df.at[i, 'folder_type']
        dir_name = df.at[i, 'dir']
        dir_parent = 'Suspicious_cases'
        if dir_type == 'normal':
            dir_parent = 'Normal_cases'

        src_dir = os.path.join(src_root_dir, str(dir_parent), str(dir_name))
        dst_dir = os.path.join(dst_root_dir, dir_type + '_' + str(dir_name))
        os.makedirs(dst_dir, exist_ok=True)
        names = os.listdir(src_dir)
        s = df.loc[i]
        names = [name for name in names if '.dcm' in name.lower()]
        priors = [name for name in names if 'prior' in name]
        recents = [name for name in names if 'recent' in name]
        assert len(priors) == 2 or len(priors) == 0, names
        assert len(recents) == 2, f'{names} and {dir_parent} -{dir_name}'
        for ori_name in names:
            name = ori_name.split('.')[0]
            new_s = s.copy()
            if new_s['biopsy_result'] == 'DCIS' or new_s[
                    'biopsy_result'] == 'MALIGNANT':
                if 'prior' in name:
                    new_s['biopsy_result'] = 'BENIGN/NORMAL'
            new_name = str(new_s['folder_type']) + '_' + str(
                new_s['dir']) + name
            new_s['image_id'] = new_name
            src_path = os.path.join(src_dir, ori_name)
            dst_path = os.path.join(dst_dir, new_name + '.dcm')
            if force_copy:
                shutil.copy2(src_path, dst_path)
            else:
                make_symlink(src_path, dst_path)
            new_s['view'] = name.split('_')[0]
            all_series.append(new_s)

    df = pd.DataFrame(all_series)
    df['cancer'] = df.biopsy_result.apply(lambda x: 1
                                          if x in ['MALIGNANT', 'DCIS'] else 0)
    df.rename(columns={
        'dir': 'patient_id',
    }, inplace=True)
    df['patient_id'] = df['folder_type'] + '_' + df['patient_id'].apply(str)
    dicom_df = get_dicom_meta(dst_root_dir, df, extension='dcm')
    dicom_df.dropna(axis=1, how='all', inplace=True)
    dicom_df.rename(columns={
        name: '__' + name
        for name in dicom_df.columns if name != 'image_id'
    },
                    inplace=True)
    merged = pd.merge(
        df,
        dicom_df,
        how="inner",
        on='image_id',
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("", "__"),
        copy=True,
        indicator=False,
        validate='1:1',
    )
    merged = merged[[
        'patient_id', 'laterality', 'age', 'density', 'BIRADS',
        'biopsy_result', 'folder_type', 'view', 'image_id', 'cancer',
        '__WindowWidth', '__WindowCenter', '__WindowCenterWidthExplanation',
        '__RescaleType', '__RescaleSlope', '__RescaleIntercept', '__Rows',
        '__Columns', '__BitsAllocated', '__BitsStored'
    ]]
    merged.sort_values(by=['patient_id', 'image_id'], inplace=True)
    merged.to_csv(cleaned_label_path, index=False)
