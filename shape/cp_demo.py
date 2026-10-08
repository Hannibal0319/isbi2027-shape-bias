"""Retrospective repair of genuine CellProfiler output: run CellProfiler's own MeasureObjectSizeShape function
(cellprofiler_library) on BBBC038 nuclei at native, 1/2 and 1/4 resolution, then repair the table with fix_cellprofiler."""
import numpy as np, pandas as pd, time
from cellprofiler_library.functions.measurement import measure_object_size_shape_2d
from shapefix import fix_cellprofiler
from estimators import p_skimage

df = pd.read_csv('real_bbbc_670.csv')
crops = np.load('real_bbbc_670_crops.npz')
rows, t0 = [], time.time()
for i, rec in df.iterrows():
    lab = crops[f'm{i}'].astype(np.int32)
    props, (ff, comp), *_ = measure_object_size_shape_2d(lab, ['label', 'area', 'perimeter', 'image'], False, (1.0, 1.0))
    rows.append(dict(img=rec.img, iid=rec.iid, scale=rec.scale, AreaShape_Area=props['area'][0],
                     AreaShape_Perimeter=props['perimeter'][0], AreaShape_FormFactor=ff[0], AreaShape_Compactness=comp[0],
                     sk=p_skimage(crops[f'm{i}'])))
cp = pd.DataFrame(rows)
print('CellProfiler run: %d objects in %.0f s; max |CP perimeter - skimage| = %.2e'
      % (len(cp), time.time() - t0, (cp.AreaShape_Perimeter - cp.sk).abs().max()))
t1 = time.time(); fixed = fix_cellprofiler(cp); dt = time.time() - t1
print('fix_cellprofiler on %d rows: %.1f ms' % (len(cp), dt * 1e3))
for name, d in [('CellProfiler', cp), ('repaired (F1)', fixed)]:
    W = d.pivot_table(index=['img', 'iid'], columns='scale', values='AreaShape_FormFactor').dropna()
    sd = W[1].std()
    print(f'{name:14s} FormFactor mean {W[1].mean():.3f}/{W[2].mean():.3f}/{W[4].mean():.3f}  shift/SD 1/2: {(W[2]-W[1]).mean()/sd:+.2f}  1/4: {(W[4]-W[1]).mean()/sd:+.2f}  (n={len(W)})')
cp.to_csv('cp_demo.csv', index=False)
