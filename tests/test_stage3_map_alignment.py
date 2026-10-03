import unittest

import torch

from quest.map_teacher import (
    align_teacher_map_to_quest_bev, resolve_lidar2ego, soft_map_distillation_loss,
)


class Stage3MapAlignmentTest(unittest.TestCase):
    def test_identity_lidar_to_ego(self):
        matrix = resolve_lidar2ego({"lidar2global": torch.eye(4), "ego2global": torch.eye(4)})
        torch.testing.assert_close(matrix, torch.eye(4, dtype=torch.float64))
        direct = resolve_lidar2ego({"lidar2ego": torch.eye(4)})
        torch.testing.assert_close(direct, matrix)

    def test_translation_lidar_to_ego(self):
        lidar2global = torch.eye(4)
        lidar2global[:2, 3] = torch.tensor([1.0, -2.0])
        matrix = resolve_lidar2ego({"lidar2global": lidar2global, "ego2global": torch.eye(4)})
        torch.testing.assert_close(matrix[:2, 3], torch.tensor([1.0, -2.0], dtype=torch.float64))

    def test_ninety_degree_rotation(self):
        rotation = torch.eye(4)
        rotation[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        matrix = resolve_lidar2ego({"lidar2global": rotation, "ego2global": torch.eye(4)})
        torch.testing.assert_close(matrix, rotation.double())
        teacher = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        aligned, valid = align_teacher_map_to_quest_bev(
            teacher, (-1, -1, 1, 1), (-1, -1, 1, 1), 2, 2, "y", 1, 1,
            lidar2ego=matrix,
        )
        torch.testing.assert_close(aligned, torch.tensor([[[0.2, 0.4], [0.1, 0.3]]]))
        self.assertTrue(bool(valid.all()))

    def test_rotation_translation_and_explicit_agreement(self):
        ego2global = torch.eye(4)
        ego2global[:2, 3] = torch.tensor([5.0, -3.0])
        explicit = torch.eye(4)
        explicit[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        explicit[:2, 3] = torch.tensor([1.0, 2.0])
        lidar2global = ego2global @ explicit
        matrix = resolve_lidar2ego({
            "lidar2ego": explicit, "lidar2global": lidar2global,
            "ego2global": ego2global,
        })
        torch.testing.assert_close(matrix, explicit.double())

    def test_inconsistent_explicit_and_derived_raise(self):
        wrong = torch.eye(4)
        wrong[0, 3] = 2
        with self.assertRaisesRegex(ValueError, "disagrees"):
            resolve_lidar2ego({
                "lidar2ego": wrong, "lidar2global": torch.eye(4),
                "ego2global": torch.eye(4),
            })

    def test_out_of_teacher_range_is_not_kd_supervision(self):
        teacher = torch.zeros(1, 4, 4)
        lidar2ego = torch.eye(4)
        lidar2ego[0, 3] = 1.0
        _, valid = align_teacher_map_to_quest_bev(
            teacher, (-2, -2, 2, 2), (-2, -2, 2, 2), 4, 4, "y", 1, 1,
            lidar2ego=lidar2ego,
        )
        self.assertFalse(bool(valid[:, -1].any()))
        self.assertTrue(bool(valid[:, :-1].all()))
        logits = torch.zeros(1, 1, 4, 4)
        logits[0, 0, :, -1] = 10.0
        support = torch.tensor([True])
        weights = torch.ones(1)
        masked = soft_map_distillation_loss(logits, teacher[None], valid[None], support, weights)
        unmasked = soft_map_distillation_loss(
            logits, teacher[None], torch.ones_like(valid[None]), support, weights,
        )
        torch.testing.assert_close(masked, torch.tensor(0.69314718), atol=1e-6, rtol=0)
        self.assertGreater(float(unmasked), float(masked))


if __name__ == "__main__":
    unittest.main()
