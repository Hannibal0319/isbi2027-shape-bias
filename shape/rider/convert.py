"""RIDER Lung CT (coffee-break): CT series + tumour DICOM SEG -> NIfTI volumes on the CT grid, with an index of
patient, study (test/retest), slice thickness and kernel."""
import json, glob, os, sys
import numpy as np, pydicom, pydicom_seg, SimpleITK as sitk, pandas as pd

R = json.load(open('seg_index.json'))
os.makedirs('nii', exist_ok=True)
rows = []
for r in R:
    ct_dir = 'ct/' + str(r['ref'])
    if not r['ref'] or not os.path.isdir(ct_dir):
        continue
    out_img, out_msk = f"nii/{r['ref']}_ct.nii.gz", f"nii/{r['ref']}_seg.nii.gz"
    files = glob.glob(ct_dir + '/*.dcm')
    h = pydicom.dcmread(files[0], stop_before_pixels=True)
    rec = dict(patient=h.PatientID, study=h.StudyInstanceUID, date=h.StudyDate, series=r['ref'],
               thickness=float(h.SliceThickness), kernel=str(h.ConvolutionKernel), px=float(h.PixelSpacing[0]),
               ct=out_img, seg=out_msk)
    if not os.path.exists(out_msk):
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(ct_dir)
        reader.SetFileNames(names)
        ct = reader.Execute()
        seg_ds = pydicom.dcmread(glob.glob(f"seg/{r['seg']}/*.dcm")[0])
        res = pydicom_seg.SegmentReader().read(seg_ds)
        seg_img = res.segment_image(res.available_segments.pop())
        seg_on_ct = sitk.Resample(seg_img, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(ct, out_img)
        sitk.WriteImage(seg_on_ct, out_msk)
    m = sitk.GetArrayFromImage(sitk.ReadImage(out_msk))
    rec['nvox'] = int(m.sum())
    rows.append(rec)
df = pd.DataFrame(rows)
# test = earlier study date per patient
df['visit'] = df.groupby('patient').date.rank(method='dense').astype(int)
df.to_csv('index.csv', index=False)
print(len(df), 'series converted')
print(df.groupby(['thickness', 'kernel']).size())
print(df[['patient', 'visit', 'thickness', 'kernel', 'nvox']].head())
