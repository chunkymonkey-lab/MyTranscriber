from setuptools import setup

APP = ["multi_transcriber_enhanced.py"]
DATA_FILES = [
    ("resources", ["resources/logo.png"]),
]

OPTIONS = {
    'argv_emulation': False,
    'packages': [
        'PyQt6',
        'yt_dlp',
        'ffmpeg',
        'speechbrain',
        'whisper',
        'yaml'
    ],
}

setup(
    app=APP,
    name="MyTranscriber",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)