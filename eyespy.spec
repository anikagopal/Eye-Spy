# -*- mode: python ; coding: utf-8 -*-
from kivy_deps import sdl2, glew

a = Analysis(
    [r"C:\eyespy\main.py"],
    pathex=[r"C:\eyespy"],
    binaries=[],
    datas=[
('shape_predictor_68_face_landmarks.dat','.'),
('haarcascade_frontalface_default.xml','.'),
('haarcascade_eye.xml','.'),
('drowsiness_onlineclass.png','.'),
],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    Tree('..\\eyespy\\',
        excludes=['venv','.idea','*.txt','log','saved_data','*.cfg']),
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='eyespy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
