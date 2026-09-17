# Removes the Abaqus scratch files left after a run.

import os
import glob

def cleanup_abaqus_junk(base_dir=None):
    # Remove the Abaqus scratch files from base_dir (or the CWD). Skipped if a .exception file exists.
    prev_cwd = os.getcwd()
    if base_dir is not None:
        try:
            os.chdir(base_dir)
        except OSError:
            pass

    try:
        if glob.glob("*.exception"):
            return

        extensions = [
            "*.abq", "*.cid", "*.com", "*.ipm", "*.log",
            "*.mdl", "*.pac", "*.prt", "*.res", "*.sel",
            "*.simlog", "*.stt", "*.rpy*", "*.env", "*.rec",
        ]
        for ext in extensions:
            for f in glob.glob(ext):
                try:
                    os.remove(f)
                except OSError:
                    pass
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass