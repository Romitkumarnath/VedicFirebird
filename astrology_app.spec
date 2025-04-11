# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run_astrology_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include dotenv file if it exists
        ('.env', '.'),
        # Include model files that might be required by sentence-transformers
        ('cache', 'cache'),
    ],
    hiddenimports=[
        'tiktoken',
        'chromadb',
        'sentence_transformers',
        'gradio',
        'langchain',
        'langchain_community',
        'langchain_community.document_loaders',
        'PyPDF',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Vedic Astrology Bot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging, False for production
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico',  # You'll need to create/add this icon file
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Vedic Astrology Bot',
) 