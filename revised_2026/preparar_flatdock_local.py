#!/usr/bin/env python3
"""Assemble local reference tree for the unchanged FlatDock verification script.
Only writes inside revised_2026/flatdock_local and never changes source archives.
"""
from pathlib import Path
from zipfile import ZipFile
import shutil
root=Path(__file__).resolve().parent
target=root/'flatdock_local'
source=target/'2026_3_correccion_outlier'/'LD50_docking_signal_followup'/'dataset'
source.mkdir(parents=True,exist_ok=True)
for name in ('combined_dataset.csv','voxel_features_nutridock_style_synthesized.csv'):
    shutil.copy2(root/'dataset'/name,source/name)
zp=root/'results_archives'/'2026_4_VERIFICADO.zip'
with ZipFile(zp) as z:
    names=[name for name in z.namelist() if name.startswith('2026_4/ChInceptionV3/') and name.endswith('predictions_row_level.csv')]
    for n in names:
        dest=target/n
        dest.parent.mkdir(parents=True,exist_ok=True)
        with z.open(n) as src, dest.open('wb') as dst:shutil.copyfileobj(src,dst)
print('Archivos para FlatDock preparados: ',target)
print('Comprobación recomendada (sin entrenamiento):')
print(f'python REENTRENAR_FLATDOCK_10TESTS_VERIFICABLE.py --root "{target}" --audit-only')
