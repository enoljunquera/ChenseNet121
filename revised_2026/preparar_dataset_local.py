#!/usr/bin/env python3
r"""Add heavy original voxel/image modalities from Zenodo v1 to the corrected local source.
Does not replace the curated 288-row CSV or test split files included here.
Usage: python preparar_dataset_local.py "C:\path\LD50_multimodal_dataset.zip"
"""
import sys
from pathlib import Path
from zipfile import ZipFile
import shutil

root=Path(__file__).resolve().parent
target=root/'dataset'
if len(sys.argv)!=2:
    sys.exit('Uso: python preparar_dataset_local.py "C:\\ruta\\LD50_multimodal_dataset.zip"')
zp=Path(sys.argv[1]).expanduser().resolve()
if not zp.is_file():
    sys.exit(f'No encuentro el ZIP de Zenodo: {zp}')
with ZipFile(zp) as z:
    names=[n for n in z.namelist() if n in ('dataset_descriptores_bool.txt','dataset_descriptores_float.txt','dataset_descriptores_2x2x2_bool.txt','dataset_descriptores_2x2x2_float.txt') or (n.startswith('images/') and n.lower().endswith('.png'))]
    if not {'dataset_descriptores_bool.txt','dataset_descriptores_float.txt','dataset_descriptores_2x2x2_bool.txt','dataset_descriptores_2x2x2_float.txt'}.issubset(names):
        sys.exit('El ZIP no contiene los dos descriptores volumétricos principales')
    for i,n in enumerate(names,1):
        dst=target/n
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists() or dst.stat().st_size!=z.getinfo(n).file_size:
            with z.open(n) as src, dst.open('wb') as f:
                shutil.copyfileobj(src,f)
    print(f'Extraídos/verificados {len(names)} ficheros (imágenes y volúmenes) en {target}')
    print('CSV corregido preservado:',target/'combined_dataset.csv')
print('Puedes auditar (sin entrenamiento) ejecutando:')
print('python baseline_no_docking_grouped.py --audit-only')
