"""Keep the existing supported-class Agent objective and held-out matching protocol."""

from .agent_training import (class_aware_center_matches, compute_stage2_agent_loss,
    decode_supported_agent_predictions, prepare_navformer_agent_batch)


def temporal_agent_loss(model, predictions, batch, config, device):
    settings = config["temporal_agent"]["recipe"]
    target = prepare_navformer_agent_batch(batch, device)
    result = compute_stage2_agent_loss(model, predictions, target, settings["train"], settings["agent_loss"])
    return result["total_agent_loss"]


def empty_agent_agreement():
    return {name: {"predictions": 0, "teacher_count": 0, "matched": 0, "center_error_sum_m": 0.,
                   "size_error_sum_m": 0., "velocity_error_sum_mps": 0., "yaw_error_sum_rad": 0.}
            for name in ("vehicle", "pedestrian")}


def update_agent_agreement(metrics, model, predictions, batch, config, device):
    recipe = config["temporal_agent"]["recipe"]
    target = prepare_navformer_agent_batch(batch, device)
    for index in range(len(batch["sample_token"])):
        single = {key: predictions[key][index].float() for key in ("agent_cls_logits", "agent_boxes", "agent_velocity")}
        decoded = decode_supported_agent_predictions(single, target["class_support_mask"][index],
            recipe["eval"]["confidence_threshold"], model.geometry_lift.x_range, model.geometry_lift.y_range,
            recipe["agent_loss"]["z_range"], recipe["agent_loss"]["size_norm"])
        valid = target["valid_mask"][index]
        labels, boxes = target["labels"][index, valid], target["boxes_metric"][index, valid]
        for class_id, name in enumerate(metrics):
            metrics[name]["predictions"] += int((decoded["labels"] == class_id).sum())
            metrics[name]["teacher_count"] += int((labels == class_id).sum())
        matches = class_aware_center_matches(decoded["centers"], decoded["labels"], boxes[:, :3], labels,
                                             recipe["eval"]["match_distance_m"])
        for pred_index, target_index, distance in matches:
            name = ("vehicle", "pedestrian")[int(labels[target_index])]
            metrics[name]["matched"] += 1
            metrics[name]["center_error_sum_m"] += distance
            metrics[name]["size_error_sum_m"] += float((decoded["sizes"][pred_index] - boxes[target_index, 3:6]).abs().mean())
            metrics[name]["velocity_error_sum_mps"] += float((decoded["velocity"][pred_index]
                - target["velocity_mps"][index, valid][target_index]).norm())
            difference = decoded["yaw"][pred_index] - boxes[target_index, 6]
            metrics[name]["yaw_error_sum_rad"] += float(difference.sin().atan2(difference.cos()).abs())


def summarize_agent_agreement(metrics, config):
    result = {}
    for name, row in metrics.items():
        result[name] = {key: row[key] for key in ("predictions", "teacher_count", "matched")}
        result[name].update(precision=row["matched"] / row["predictions"] if row["predictions"] else 0.,
            recall=row["matched"] / row["teacher_count"] if row["teacher_count"] else 0.,
            mean_matched_center_error_m=row["center_error_sum_m"] / row["matched"] if row["matched"] else None)
        for field in ("size_error_sum_m", "velocity_error_sum_mps", "yaw_error_sum_rad"):
            result[name]["mean_matched_" + field.replace("_sum", "")] = row[field] / row["matched"] if row["matched"] else None
    recipe = config["temporal_agent"]["recipe"]["eval"]
    return {"reference": "NAVFORMER_OFFLINE_PSEUDO_NOT_REAL_GT", "official_mAP": False,
            "confidence_threshold": recipe["confidence_threshold"],
            "match_distance_m": recipe["match_distance_m"], "classes": result}
