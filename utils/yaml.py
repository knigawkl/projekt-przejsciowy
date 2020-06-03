import bios


def read_cfg(cfg_path: str) -> dict:
    return bios.read(cfg_path)
