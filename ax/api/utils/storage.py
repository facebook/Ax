# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from ax.api.configs import StorageConfig

# Allow Ax to be used without SQLAlchemy.
try:
    from ax.storage.sqa_store.decoder import Decoder
    from ax.storage.sqa_store.encoder import Encoder
    from ax.storage.sqa_store.sqa_config import SQAConfig
    from ax.storage.sqa_store.structs import DBSettings
except (ModuleNotFoundError, TypeError):
    Decoder = None
    Encoder = None
    SQAConfig = None
    DBSettings = None


def db_settings_from_storage_config(
    storage_config: StorageConfig,
) -> DBSettings:
    """Construct DBSettings (expected by WithDBSettingsBase) from StorageConfig."""
    if (bundle := storage_config.registry_bundle) is not None:
        encoder = bundle.encoder
        decoder = bundle.decoder
    else:
        encoder = Encoder(config=SQAConfig())
        decoder = Decoder(config=SQAConfig())

    return DBSettings(
        creator=storage_config.creator,
        url=storage_config.url,
        encoder=encoder,
        decoder=decoder,
    )
