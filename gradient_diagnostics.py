import collections
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

from data.keypoint_schema import OUT_OF_FRAME_NUM
from data.keypoint_processing import ZERO_WITH_PRESENCE


class GradientDiagnosticCollator:
    """Attach read-only batch metadata to an existing data collator."""

    metadata_key = "_gradient_diagnostic_metadata"

    def __init__(self, collator):
        self.collator = collator
        processing_config = collator.keypoint_processing_config
        self.missing_representation = processing_config[
            "missing_representation"
        ]
        self.num_coords = processing_config["num_coords"]
        self.uses_presence = (
            self.missing_representation == ZERO_WITH_PRESENCE
        )

    def __call__(self, batch):
        collated = self.collator(batch)

        pose_lengths = collated["vectors_attention_mask"].sum(dim=1).tolist()
        input_token_lengths = collated["text_attention_mask"].sum(dim=1).tolist()
        target_token_lengths = (collated["labels"] != -100).sum(dim=1).tolist()

        metadata = []
        for index, instance in enumerate(batch):
            pose_length = int(pose_lengths[index])
            pose = collated["input_vectors"][index, :pose_length]
            if pose.numel():
                if self.uses_presence:
                    landmarks = pose.reshape(
                        pose.shape[0],
                        -1,
                        self.num_coords + 1,
                    )
                    coordinates = landmarks[..., :self.num_coords]
                    presence = landmarks[..., self.num_coords]
                    landmark_observed = presence >= 0.5
                    observed_mask = landmark_observed[..., None].expand_as(
                        coordinates
                    )
                    presence_nonbinary_count = int(
                        ((presence != 0.0) & (presence != 1.0)).sum().item()
                    )
                else:
                    coordinates = pose
                    observed_mask = coordinates != OUT_OF_FRAME_NUM
                    presence_nonbinary_count = 0

                missing_coordinate_pct = float(
                    (~observed_mask).float().mean().item() * 100
                )
                finite_observed_mask = (
                    observed_mask & torch.isfinite(coordinates)
                )
                observed_coordinates = coordinates[
                    finite_observed_mask
                ].float()
                nonfinite_coordinate_count = int(
                    (
                        observed_mask & ~torch.isfinite(coordinates)
                    ).sum().item()
                )

                if observed_coordinates.numel():
                    observed_coordinates_double = observed_coordinates.double()
                    absolute_coordinates = observed_coordinates_double.abs()
                    mean_abs_coordinate = float(
                        absolute_coordinates.mean().item()
                    )
                    max_abs_coordinate = float(
                        absolute_coordinates.max().item()
                    )
                    coordinate_rms = float(
                        torch.sqrt(
                            torch.mean(observed_coordinates_double.square())
                        ).item()
                    )
                    coordinate_abs_gt_5_pct = float(
                        (absolute_coordinates > 5).float().mean().item() * 100
                    )
                    coordinate_abs_gt_10_pct = float(
                        (absolute_coordinates > 10).float().mean().item() * 100
                    )
                    coordinate_abs_gt_100_pct = float(
                        (absolute_coordinates > 100).float().mean().item() * 100
                    )
                else:
                    mean_abs_coordinate = None
                    max_abs_coordinate = None
                    coordinate_rms = None
                    coordinate_abs_gt_5_pct = None
                    coordinate_abs_gt_10_pct = None
                    coordinate_abs_gt_100_pct = None

                finite_pose = torch.where(
                    finite_observed_mask,
                    coordinates,
                    torch.zeros_like(coordinates),
                ).double()
                non_frame_dims = tuple(range(1, finite_pose.ndim))
                observed_per_frame = finite_observed_mask.sum(
                    dim=non_frame_dims
                )
                frame_rms = torch.sqrt(
                    finite_pose.square().sum(dim=non_frame_dims)
                    / observed_per_frame.clamp_min(1)
                )
                max_frame_rms = float(frame_rms.max().item())
                all_missing_frames_pct = float(
                    (observed_per_frame == 0).float().mean().item() * 100
                )
            else:
                missing_coordinate_pct = 0.0
                nonfinite_coordinate_count = 0
                mean_abs_coordinate = None
                max_abs_coordinate = None
                coordinate_rms = None
                coordinate_abs_gt_5_pct = None
                coordinate_abs_gt_10_pct = None
                coordinate_abs_gt_100_pct = None
                max_frame_rms = None
                all_missing_frames_pct = 0.0
                presence_nonbinary_count = 0

            start_frame = instance.get("start_frame")
            end_frame = instance.get("end_frame")
            raw_pose_frames = (
                int(end_frame - start_frame)
                if start_frame is not None and end_frame is not None
                else None
            )

            metadata.append({
                "video_id": instance.get("video_id"),
                "instance_type": instance.get("instance_type"),
                "src_lang": instance.get("src_lang"),
                "tgt_lang": instance.get("tgt_lang"),
                "raw_pose_frames": raw_pose_frames,
                "pose_frames": pose_length,
                "input_tokens": int(input_token_lengths[index]),
                "target_tokens": int(target_token_lengths[index]),
                "missing_coordinate_pct": missing_coordinate_pct,
                "all_missing_frames_pct": all_missing_frames_pct,
                "nonfinite_coordinate_count": nonfinite_coordinate_count,
                "presence_nonbinary_count": presence_nonbinary_count,
                "mean_abs_coordinate": mean_abs_coordinate,
                "max_abs_coordinate": max_abs_coordinate,
                "coordinate_rms": coordinate_rms,
                "max_frame_rms": max_frame_rms,
                "coordinate_abs_gt_5_pct": coordinate_abs_gt_5_pct,
                "coordinate_abs_gt_10_pct": coordinate_abs_gt_10_pct,
                "coordinate_abs_gt_100_pct": coordinate_abs_gt_100_pct,
            })

        collated[self.metadata_key] = metadata
        return collated


def _numeric_summary(values):
    finite_values = [
        value
        for value in values
        if value is not None and math.isfinite(value)
    ]
    if not finite_values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(finite_values),
        "mean": sum(finite_values) / len(finite_values),
        "max": max(finite_values),
    }


def _batch_summary(examples):
    video_ids = [
        example["video_id"]
        for example in examples
        if example["video_id"] is not None
    ]
    return {
        "num_examples": len(examples),
        "num_unique_videos": len(set(video_ids)),
        "repeated_video_examples": len(video_ids) - len(set(video_ids)),
        "instance_type_counts": dict(collections.Counter(
            example["instance_type"] for example in examples
        )),
        "target_language_counts": dict(collections.Counter(
            example["tgt_lang"] for example in examples
        )),
        "raw_pose_frames": _numeric_summary([
            example["raw_pose_frames"]
            for example in examples
            if example["raw_pose_frames"] is not None
        ]),
        "pose_frames": _numeric_summary([
            example["pose_frames"] for example in examples
        ]),
        "input_tokens": _numeric_summary([
            example["input_tokens"] for example in examples
        ]),
        "target_tokens": _numeric_summary([
            example["target_tokens"] for example in examples
        ]),
        "missing_coordinate_pct": _numeric_summary([
            example["missing_coordinate_pct"] for example in examples
        ]),
        "all_missing_frames_pct": _numeric_summary([
            example["all_missing_frames_pct"] for example in examples
        ]),
        "fully_missing_examples": sum(
            example["missing_coordinate_pct"] >= 99.999
            for example in examples
        ),
        "nonfinite_coordinate_count": sum(
            example["nonfinite_coordinate_count"] for example in examples
        ),
        "examples_with_nonfinite_coordinates": sum(
            example["nonfinite_coordinate_count"] > 0
            for example in examples
        ),
        "presence_nonbinary_count": sum(
            example["presence_nonbinary_count"] for example in examples
        ),
        "mean_abs_coordinate": _numeric_summary([
            example["mean_abs_coordinate"] for example in examples
        ]),
        "max_abs_coordinate": _numeric_summary([
            example["max_abs_coordinate"] for example in examples
        ]),
        "coordinate_rms": _numeric_summary([
            example["coordinate_rms"] for example in examples
        ]),
        "max_frame_rms": _numeric_summary([
            example["max_frame_rms"] for example in examples
        ]),
        "coordinate_abs_gt_5_pct": _numeric_summary([
            example["coordinate_abs_gt_5_pct"] for example in examples
        ]),
        "coordinate_abs_gt_10_pct": _numeric_summary([
            example["coordinate_abs_gt_10_pct"] for example in examples
        ]),
        "coordinate_abs_gt_100_pct": _numeric_summary([
            example["coordinate_abs_gt_100_pct"] for example in examples
        ]),
    }


def _tensor_summary(tensor):
    tensor = tensor.detach()
    finite_tensor = tensor[torch.isfinite(tensor)]
    nonfinite_count = tensor.numel() - finite_tensor.numel()
    if not finite_tensor.numel():
        return {
            "mean": None,
            "std": None,
            "rms": None,
            "max_abs": None,
            "nonfinite_count": nonfinite_count,
        }

    finite_tensor_double = finite_tensor.double()
    return {
        "mean": float(finite_tensor_double.mean().item()),
        "std": float(finite_tensor_double.std(unbiased=False).item()),
        "rms": float(
            torch.sqrt(finite_tensor_double.square().mean()).item()
        ),
        "max_abs": float(finite_tensor_double.abs().max().item()),
        "nonfinite_count": nonfinite_count,
    }


def _optional_tensor_summary(tensor):
    if not tensor.numel():
        return None
    return _tensor_summary(tensor)


def _cosine_similarity(left, right):
    if left is None or right is None or left.numel() != right.numel():
        return None
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator == 0:
        return None
    return float(torch.dot(left, right).div(denominator).item())


class GradientDiagnosticSeq2SeqTrainer(Seq2SeqTrainer):
    """Log pre-clipping gradient norms and the batches that produced them."""

    def __init__(self, *args, gradient_spike_threshold=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_spike_threshold = gradient_spike_threshold
        self._gradient_diagnostic_examples = []
        self._gradient_diagnostic_losses = []
        self._projection_activation_summaries = []
        self._projection_missing_value_summaries = []
        self._gradient_diagnostic_file = None
        self._current_vector_mask = None
        self._previous_projection_gradient = None
        self._uses_presence = getattr(
            self.data_collator,
            "uses_presence",
            False,
        )
        self._num_coords = getattr(
            self.data_collator,
            "num_coords",
            None,
        )

        base_model = self.accelerator.unwrap_model(self.model)
        self._projection_hook = base_model.projection.register_forward_hook(
            self._capture_projection_activations
        )

    def _capture_projection_activations(self, module, inputs, output):
        if self._current_vector_mask is None:
            return

        with torch.no_grad():
            input_vectors = inputs[0]
            vector_mask = self._current_vector_mask.to(
                device=output.device,
                dtype=torch.bool,
            )
            valid_inputs = input_vectors[vector_mask]
            valid_activations = output.detach()[vector_mask]
            if self._uses_presence:
                grouped_inputs = valid_inputs.reshape(
                    valid_inputs.shape[0],
                    module.in_features // (self._num_coords + 1),
                    self._num_coords + 1,
                )
                frame_has_missing = (
                    grouped_inputs[..., self._num_coords] < 0.5
                ).any(dim=-1)
            else:
                grouped_inputs = None
                frame_has_missing = (
                    valid_inputs == OUT_OF_FRAME_NUM
                ).any(dim=-1)

            summary = _tensor_summary(valid_activations)
            if valid_activations.numel():
                frame_rms = torch.sqrt(
                    valid_activations.double().square().mean(dim=-1)
                )
                summary["max_frame_rms"] = float(frame_rms.max().item())
            else:
                summary["max_frame_rms"] = None
            summary["num_fully_observed_frames"] = int(
                (~frame_has_missing).sum().item()
            )
            summary["num_frames_with_missing"] = int(
                frame_has_missing.sum().item()
            )
            summary["fully_observed_frames"] = _optional_tensor_summary(
                valid_activations[~frame_has_missing]
            )
            summary["frames_with_missing"] = _optional_tensor_summary(
                valid_activations[frame_has_missing]
            )

            zero_input = torch.zeros(
                module.in_features,
                device=input_vectors.device,
                dtype=input_vectors.dtype,
            )
            zero_response = F.linear(
                zero_input,
                module.weight,
                module.bias,
            )

            if self._uses_presence:
                coordinate_inputs = grouped_inputs.clone()
                coordinate_inputs[..., self._num_coords] = 0.0
                coordinate_inputs = coordinate_inputs.reshape(
                    valid_inputs.shape
                )
                presence_inputs = torch.zeros_like(grouped_inputs)
                presence_inputs[..., self._num_coords] = (
                    grouped_inputs[..., self._num_coords]
                )
                presence_inputs = presence_inputs.reshape(valid_inputs.shape)

                coordinate_contribution = F.linear(
                    coordinate_inputs,
                    module.weight,
                    bias=None,
                )
                presence_contribution = F.linear(
                    presence_inputs,
                    module.weight,
                    bias=None,
                )
                summary["coordinate_contribution"] = _tensor_summary(
                    coordinate_contribution
                )
                summary["presence_contribution"] = _tensor_summary(
                    presence_contribution
                )
                summary["projection_bias"] = _tensor_summary(module.bias)

                all_present_zero_coordinates = zero_input.reshape(
                    -1,
                    self._num_coords + 1,
                )
                all_present_zero_coordinates[..., self._num_coords] = 1.0
                all_present_zero_coordinates = (
                    all_present_zero_coordinates.reshape(-1)
                )
                all_present_zero_response = F.linear(
                    all_present_zero_coordinates,
                    module.weight,
                    module.bias,
                )
                fixed_summary = {
                    "missing_representation": ZERO_WITH_PRESENCE,
                    "all_missing_zero_input_response": _tensor_summary(
                        zero_response
                    ),
                    "all_present_zero_coordinates_response": _tensor_summary(
                        all_present_zero_response
                    ),
                    "all_present_presence_contribution": _tensor_summary(
                        all_present_zero_response - zero_response
                    ),
                }
            else:
                # Isolate exactly what the sentinel entries add to the
                # projection. Bias is excluded from this contribution.
                missing_inputs = torch.where(
                    valid_inputs == OUT_OF_FRAME_NUM,
                    valid_inputs,
                    torch.zeros_like(valid_inputs),
                )
                missing_contribution = F.linear(
                    missing_inputs,
                    module.weight,
                    bias=None,
                )
                observed_plus_bias = (
                    valid_activations - missing_contribution
                )
                summary["missing_value_contribution"] = _tensor_summary(
                    missing_contribution
                )
                summary["observed_coordinates_plus_bias"] = _tensor_summary(
                    observed_plus_bias
                )

                all_missing_input = torch.full_like(
                    zero_input,
                    OUT_OF_FRAME_NUM,
                )
                all_missing_response = F.linear(
                    all_missing_input,
                    module.weight,
                    module.bias,
                )
                fixed_summary = {
                    "missing_value": OUT_OF_FRAME_NUM,
                    "zero_input_response": _tensor_summary(zero_response),
                    "all_missing_input_response": _tensor_summary(
                        all_missing_response
                    ),
                    "all_missing_value_contribution": _tensor_summary(
                        all_missing_response - zero_response
                    ),
                }

        self._projection_activation_summaries.append(summary)
        self._projection_missing_value_summaries.append(fixed_summary)

    def _open_gradient_diagnostic_file(self):
        if self._gradient_diagnostic_file is not None:
            return

        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(
            self.args.output_dir,
            f"gradient_diagnostics_rank{self.args.process_index}.jsonl",
        )
        self._gradient_diagnostic_file = open(path, "w", encoding="utf-8")
        print(f"Gradient diagnostics: {path}")

    def _write_gradient_diagnostics(self, loss, model):
        named_parameters = list(model.named_parameters())
        parameter_gradient_norms = [
            (
                name,
                float(torch.linalg.vector_norm(
                    parameter.grad.detach().float(),
                    ord=2,
                ).item()),
            )
            for name, parameter in named_parameters
            if parameter.grad is not None
        ]
        total_norm = math.sqrt(sum(
            norm * norm for _, norm in parameter_gradient_norms
        ))
        projection_norm = math.sqrt(sum(
            norm * norm
            for name, norm in parameter_gradient_norms
            if "projection" in name
        ))
        t5_norm = math.sqrt(sum(
            norm * norm
            for name, norm in parameter_gradient_norms
            if "projection" not in name
        ))

        projection_parameters = [
            parameter
            for name, parameter in named_parameters
            if "projection" in name
        ]
        projection_gradient = torch.cat([
            parameter.grad.detach().float().reshape(-1).cpu()
            for parameter in projection_parameters
            if parameter.grad is not None
        ])
        projection_gradient_cosine_previous = _cosine_similarity(
            projection_gradient,
            self._previous_projection_gradient,
        )
        self._previous_projection_gradient = projection_gradient.clone()

        projection_parameter_norm = math.sqrt(sum(
            float(torch.linalg.vector_norm(
                parameter.detach().float(),
                ord=2,
            ).item()) ** 2
            for parameter in projection_parameters
        ))

        projection_coordinate_gradient_norm = None
        projection_presence_gradient_norm = None
        projection_bias_gradient_norm = None
        projection_coordinate_parameter_norm = None
        projection_presence_parameter_norm = None
        projection_bias_parameter_norm = None
        if self._uses_presence:
            base_model = self.accelerator.unwrap_model(model)
            projection = base_model.projection
            grouped_weight = projection.weight.reshape(
                projection.weight.shape[0],
                -1,
                self._num_coords + 1,
            )
            projection_coordinate_parameter_norm = float(
                torch.linalg.vector_norm(
                    grouped_weight[..., :self._num_coords].detach().float()
                ).item()
            )
            projection_presence_parameter_norm = float(
                torch.linalg.vector_norm(
                    grouped_weight[..., self._num_coords].detach().float()
                ).item()
            )
            projection_bias_parameter_norm = float(
                torch.linalg.vector_norm(
                    projection.bias.detach().float()
                ).item()
            )

            if projection.weight.grad is not None:
                grouped_weight_gradient = projection.weight.grad.reshape(
                    projection.weight.shape[0],
                    -1,
                    self._num_coords + 1,
                )
                projection_coordinate_gradient_norm = float(
                    torch.linalg.vector_norm(
                        grouped_weight_gradient[
                            ..., :self._num_coords
                        ].detach().float()
                    ).item()
                )
                projection_presence_gradient_norm = float(
                    torch.linalg.vector_norm(
                        grouped_weight_gradient[
                            ..., self._num_coords
                        ].detach().float()
                    ).item()
                )
            if projection.bias.grad is not None:
                projection_bias_gradient_norm = float(
                    torch.linalg.vector_norm(
                        projection.bias.grad.detach().float()
                    ).item()
                )

        is_spike = total_norm >= self.gradient_spike_threshold
        record = {
            "optimizer_step": self.state.global_step + 1,
            "epoch": self.state.epoch,
            "rank": self.args.process_index,
            "loss": loss,
            "microbatch_losses": self._gradient_diagnostic_losses,
            "gradient_norm": total_norm,
            "projection_gradient_norm": projection_norm,
            "t5_gradient_norm": t5_norm,
            "projection_gradient_cosine_previous": (
                projection_gradient_cosine_previous
            ),
            "projection_parameter_norm": projection_parameter_norm,
            "projection_coordinate_gradient_norm": (
                projection_coordinate_gradient_norm
            ),
            "projection_presence_gradient_norm": (
                projection_presence_gradient_norm
            ),
            "projection_bias_gradient_norm": projection_bias_gradient_norm,
            "projection_coordinate_parameter_norm": (
                projection_coordinate_parameter_norm
            ),
            "projection_presence_parameter_norm": (
                projection_presence_parameter_norm
            ),
            "projection_bias_parameter_norm": projection_bias_parameter_norm,
            "projection_activation_summaries": (
                self._projection_activation_summaries
            ),
            "projection_missing_value_summaries": (
                self._projection_missing_value_summaries
            ),
            "max_grad_norm": self.args.max_grad_norm,
            "will_be_clipped": (
                self.args.max_grad_norm is not None
                and self.args.max_grad_norm > 0
                and total_norm > self.args.max_grad_norm
            ),
            "is_spike": is_spike,
            "batch": _batch_summary(self._gradient_diagnostic_examples),
            "video_ids": [
                example["video_id"]
                for example in self._gradient_diagnostic_examples
            ],
        }
        if is_spike:
            record["examples"] = self._gradient_diagnostic_examples
            record["top_gradient_parameters"] = [
                {"name": name, "gradient_norm": norm}
                for name, norm in sorted(
                    parameter_gradient_norms,
                    key=lambda item: item[1],
                    reverse=True,
                )[:20]
            ]

        self._open_gradient_diagnostic_file()
        self._gradient_diagnostic_file.write(json.dumps(record) + "\n")
        self._gradient_diagnostic_file.flush()

        if is_spike:
            print(
                "Gradient spike: "
                f"step={record['optimizer_step']} "
                f"rank={record['rank']} "
                f"norm={total_norm:.4f} "
                f"projection_norm={projection_norm:.4f} "
                f"t5_norm={t5_norm:.4f} "
                f"loss={loss:.4f} "
                f"batch={record['batch']}"
            )

        self._gradient_diagnostic_examples = []
        self._gradient_diagnostic_losses = []
        self._projection_activation_summaries = []
        self._projection_missing_value_summaries = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        metadata = inputs.pop(GradientDiagnosticCollator.metadata_key)
        self._gradient_diagnostic_examples.extend(metadata)
        self._current_vector_mask = inputs["vectors_attention_mask"]

        loss = super().training_step(
            model,
            inputs,
            num_items_in_batch=num_items_in_batch,
        )
        self._current_vector_mask = None
        detached_loss = float(loss.detach().float().item())
        self._gradient_diagnostic_losses.append(detached_loss)

        # At a synchronization step, gradients from all accumulated
        # microbatches and all DDP ranks are available, but Trainer has not
        # clipped them yet.
        if self.accelerator.sync_gradients:
            mean_loss = (
                sum(self._gradient_diagnostic_losses)
                / len(self._gradient_diagnostic_losses)
            )
            self._write_gradient_diagnostics(mean_loss, model)

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
        **gen_kwargs,
    ):
        # Evaluation uses the same wrapped collator, but its metadata must not
        # be forwarded to the model's forward or generate methods.
        inputs.pop(GradientDiagnosticCollator.metadata_key, None)
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
