# main.spec
import os
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files
)
from PyInstaller.building.build_main import Analysis, PYZ, EXE

# -------------------------------
# Paths
# -------------------------------
pathex = [os.path.abspath('.')]

# -------------------------------
# SciPy
# -------------------------------
scipy_binaries = collect_dynamic_libs('scipy')
scipy_hiddenimports = collect_submodules('scipy')

# -------------------------------
# Inference package
# -------------------------------
inference_model_modules = collect_submodules('inference.models')
inference_data = collect_data_files('inference')



# -------------------------------
# Extra data files
# -------------------------------
extra_datas = [
    ('icon.ico', '.'), 
]

# -------------------------------
# Analysis
# -------------------------------
a = Analysis(
    ['main.py'],
    pathex=pathex,
    binaries=scipy_binaries,
    datas=inference_data  + extra_datas,
    hiddenimports=[
        # SciPy
        *scipy_hiddenimports,

        # Inference
        *inference_model_modules,
    ],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused heavy deps
        'scipy._lib.array_api_compat.torch',
        'torch',
        'torchvision',
        'clip',
        'inference.models.yolo_world',
    ],
    noarchive=False,
    optimize=0,
)

# -------------------------------
# PYZ
# -------------------------------
pyz = PYZ(a.pure)

# -------------------------------
# Executable
# -------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PiztechSibo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # ✅ no console window
    icon='icon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
