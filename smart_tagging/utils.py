# Copyright 2022, dSPACE GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you must not use this software except in compliance with the License. This
# software is not fully developed or tested. It is distributed free of charge
# and without any consideration. The software is provided "as is" in the hope
# that it may be useful to other users, but without any warranty of any kind,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


import json
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import rtmaps.types
import tensorflow as tf


def load_config(path: Path) -> Dict:
    """
    Loads the model configuration from disk.

    Args:
        path (Path): Path to .json file

    Returns:
        Dict: Configuration
    """
    with Path(path).open('r') as cfg:
        return json.load(cfg)


def load_exported_model(
    export_dir: Union[str, Path],
) -> Tuple[Callable, Dict]:
    """
    Wrapper to load exported models.

    Args:
        export_dir (str/Path): path to saved model

    Returns:
        callable_fn of export_model
    """
    export_dir = Path(export_dir)
    init = load_config(export_dir / 'init.json')

    return tf.saved_model.load(str(export_dir)), init


def get_ioelt(
    timestamp: int, data: Any
) -> rtmaps.types.Ioelt:
    """
    Args:
        timestamp (ts): Input timestamp
        data (dict, scalar): Any data
    """
    e = rtmaps.types.Ioelt()
    e.ts = timestamp
    e.data = data
    return e
