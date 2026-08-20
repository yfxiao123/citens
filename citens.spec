# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the CiteLens desktop exe.
#
#   pyinstaller citens.spec --noconfirm        -> dist/CiteLens.exe (onefile)
#
# Portable by design: .env / .cache / papers / runs / data are created next
# to the exe at runtime (citens.desktop.main chdirs there first).

a = Analysis(
    ["citens/desktop.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("citens/profiles", "citens/profiles"),        # domain profiles (JSON)
        ("citens/api/static", "citens/api/static"),    # web console (vendored)
    ],
    hiddenimports=[
        # uvicorn wires its loop/protocol backends dynamically at startup
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
        # app + SSE layer (imported lazily inside desktop.main)
        "citens.api.app",
        "sse_starlette",
        # full-text toolchain: markitdown resolves converters via pkg data
        "markitdown",
        "citens.grounding.fulltext",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "pytest",
        "respx",
        "litellm",   # optional [multi] backend — huge, not needed
        "mypy",
        "ruff",
        "pip",
        "setuptools",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CiteLens",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,   # AV false positives outweigh the size win
    console=True,  # the first-run wizard + run logs live in the console
    disable_windowed_traceback=False,
)
