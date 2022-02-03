# Copyright 2022, dSPACE GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you must not use this software except in compliance with the License. This
# software is not fully developed or tested. It is distributed free of charge
# and without any consideration. The software is provided "as is" in the hope
# that it may be useful to other users, but without any warranty of any kind,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import os
import tempfile
import zipfile
from pathlib import Path

import requests


def download_and_unzip(url: str, target: Path) -> None:
    r = requests.get(url)
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip') as tmpfile:
        tmpfile.write(r.content)
        with zipfile.ZipFile(tmpfile, 'r') as zf:
            zf.extractall(target)


def check_and_install_exdata() -> None:
    # Download files
    base_dir = Path(os.path.abspath(os.path.dirname(__file__))).parent

    # Download object detection saved model
    object_detection = base_dir / 'object_detection'
    target = object_detection / 'saved_model'
    if not (target / 'saved_model.pb').exists():
        print("'Object Detection' model missing. Starting download:")
        download_and_unzip(
            'https://dl2.intempora.com/index.php/s/dE5paamcKQXPSda/download',
            target
        )
        print("  Downloaded and extracted 'Object Detection' model.")

    # Download novelty detection saved model
    novelty_detection = base_dir / 'novelty_detection'
    target = novelty_detection / 'saved_model'
    if not (target / 'saved_model.pb').exists():
        print("'Novelty Detection' model missing. Starting download:")
        download_and_unzip(
            'https://dl2.intempora.com/index.php/s/QRzst9k6DLmxGbK/download',
            target
        )
        print("  Downloaded and extracted 'Novelty Detection' model.")

    # Download object detection sample data
    object_detection = base_dir / 'object_detection'
    target = base_dir / 'examples' / 'object_detection' / 'datasets'
    target.mkdir(exist_ok=True)
    if len(list(target.iterdir())) == 0:
        print("'Object Detection' sample data missing. Starting download:")
        download_and_unzip(
            'https://dl2.intempora.com/index.php/s/skMpebbdqZ8oGMs/download',
            target
        )
        print("  Downloaded and extracted 'Object Detection' sample data.")

    # Download novelty detection sample data
    novelty_detection = base_dir / 'novelty_detection'
    target = base_dir / 'examples' / 'novelty_detection' / 'datasets'
    target.mkdir(exist_ok=True)
    if len(list(target.iterdir())) == 0:
        print("'Novelty Detection' sample data missing. Starting download:")
        download_and_unzip(
            'https://dl2.intempora.com/index.php/s/cFnZg4ooBziRQre/download',
            target
        )
        print("  Downloaded and extracted 'Novelty Detection' sample data.")


if __name__ == "__main__":
    check_and_install_exdata()
