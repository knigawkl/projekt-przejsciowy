import argparse


def get_parser() -> argparse.PARSER:
    """
    :return: argparser to be used in the main gmcp-tracker script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",
                        type=str,
                        required=True,
                        help="path to configuration file")
    parser.add_argument("--detector",
                        type=str,
                        required=True,
                        choices=["felzenszwalb", "ssd", "yolo"],
                        help="detector name")
    return parser
