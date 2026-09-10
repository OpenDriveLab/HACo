"""GR00T N1.7 pipeline specialized for the independent HACO family."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil

from gr00t.experiment.dist_utils import get_rank
from gr00t.model.registry import register_model

from dexterity.models.haco import Haco, HacoConfig, HacoProcessor
from scripts.train.haco.base_pipeline import HacoBasePipeline, convert_tensors_to_lists
from scripts.train.haco.checkpoint import (
    load_official_as_haco,
    write_initialization_manifest,
)
from scripts.train.haco.data_factory import HacoDatasetFactory


class HacoPipeline(HacoBasePipeline):
    model_class = Haco
    processor_class = HacoProcessor

    def _create_model(self):
        checkpoint = self.config.training.start_from_checkpoint
        skip_loading = getattr(self.config.training, "skip_weight_loading", False)
        manifest = None
        if checkpoint is not None and not skip_loading:
            model, manifest = load_official_as_haco(
                checkpoint,
                config=self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )
        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )
        if get_rank() == 0:
            (self.save_cfg_dir / "final_model_config.json").write_text(
                model.config.to_filtered_json() + "\n", encoding="utf-8"
            )
            if manifest is not None:
                write_initialization_manifest(
                    self.save_cfg_dir / "haco_initialization_manifest.json",
                    manifest,
                )
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        logging.info(
            "HACO %s parameters: total=%s trainable=%s (%.2f%%)",
            self.config.model.experiment_id,
            f"{total:,}",
            f"{trainable:,}",
            100 * trainable / total,
        )
        return model

    def _processor_overrides(self) -> dict:
        return {
            "modality_configs": self.config.data.modality_configs,
            "image_crop_size": self.model_config.image_crop_size,
            "image_target_size": self.model_config.image_target_size,
            "random_rotation_angle": self.model_config.random_rotation_angle,
            "color_jitter_params": self.model_config.color_jitter_params,
            "model_name": self.model_config.model_name,
            "model_type": self.model_config.backbone_model_type,
            "formalize_language": self.model_config.formalize_language,
            "apply_sincos_state_encoding": (
                self.model_config.apply_sincos_state_encoding
            ),
            "max_action_horizon": self.model_config.action_horizon,
            "use_albumentations": self.model_config.use_albumentations_transforms,
            "extra_augmentation_config": self.model_config.extra_augmentation_config,
            "shortest_image_edge": self.model_config.shortest_image_edge,
            "crop_fraction": self.model_config.crop_fraction,
            "use_relative_action": False,
            "exclude_state": self.model_config.exclude_state,
            "state_dropout_prob": 0.0,
            "use_mean_std": self.model_config.use_mean_std,
            "experiment_id": self.model_config.experiment_id,
            "action_contract": self.model_config.action_contract,
            "sensor_encoder_mode": self.model_config.sensor_encoder_mode,
            "physical_integration": self.model_config.physical_integration,
            "camera_mode": self.model_config.camera_mode,
            "transformers_loading_kwargs": self.transformers_loading_kwargs,
        }

    def _create_dataset(self, save_cfg_dir: Path):
        checkpoint = self.config.training.start_from_checkpoint
        overrides = self._processor_overrides()
        # transformers.ProcessorMixin filters kwargs against the processor
        # constructor signature before instantiation.  Keep the process-local
        # environment fallback synchronized with the already-validated model
        # config so older transformers versions cannot silently select the
        # default joint contract for the two delta-free ablations.
        processor_environment = {
            "HACO_EXPERIMENT_ID": self.model_config.experiment_id,
            "HACO_SENSOR_ENCODER_MODE": self.model_config.sensor_encoder_mode,
            "HACO_PHYSICAL_INTEGRATION": self.model_config.physical_integration,
            "HACO_ACTION_CONTRACT": self.model_config.action_contract,
            "HACO_CAMERA_MODE": self.model_config.camera_mode,
        }
        os.environ.update(processor_environment)
        if checkpoint is not None:
            processor = self.processor_class.from_pretrained(
                checkpoint,
                **overrides,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                statistics=None,
                embodiment_id_mapping=None,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                **overrides,
            )
        actual_processor_contract = {
            "experiment_id": processor.experiment_id,
            "sensor_encoder_mode": processor.sensor_encoder_mode,
            "physical_integration": processor.physical_integration,
            "action_contract": processor.action_contract.name,
            "camera_mode": processor.camera_mode,
        }
        expected_processor_contract = {
            key.removeprefix("HACO_").lower(): value
            for key, value in processor_environment.items()
        }
        if actual_processor_contract != expected_processor_contract:
            raise RuntimeError(
                "HACO processor/model contract mismatch: "
                f"actual={actual_processor_contract}, "
                f"expected={expected_processor_contract}"
            )
        self.processor = processor
        if get_rank() == 0:
            (save_cfg_dir / "final_processor_config.json").write_text(
                json.dumps(
                    {key: str(value) for key, value in vars(processor).items()},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        train_dataset, eval_dataset = HacoDatasetFactory(self.config).build(
            processor
        )
        if eval_dataset is not None:
            raise AssertionError("HACO must not construct a validation dataset")
        stats = convert_tensors_to_lists(train_dataset.get_dataset_statistics())
        (save_cfg_dir / "dataset_statistics.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        normalization = os.environ.get("HACO_MULTITASK_NORMALIZATION_DIR")
        if get_rank() == 0 and normalization:
            shutil.copytree(
                Path(normalization),
                save_cfg_dir / "multitask_normalization",
                dirs_exist_ok=True,
            )
        return train_dataset, None


register_model(HacoConfig, HacoPipeline)


__all__ = ["HacoPipeline"]
