import unittest

import torch
from torch import nn

from quest.metric3d_checkpoint import (
    MASK_TOKEN_COMPATIBILITY, UNUSED_MASK_TOKEN_KEY, load_metric3d_state,
)


class AuditedEncoderFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1024))
        self.projection = nn.Linear(2, 2)

    def prepare_tokens_with_masks(self, x, masks=None):
        if masks is not None:
            return x + self.mask_token.sum()
        return x

    def forward(self, x, masks=None):
        return self.prepare_tokens_with_masks(x, masks)


class UnsafeEncoderFixture(AuditedEncoderFixture):
    def forward(self, x, masks=None):
        return x + self.mask_token.sum()


def teacher_fixture(encoder_type=AuditedEncoderFixture):
    model = nn.Module()
    model.depth_model = nn.Module()
    model.depth_model.encoder = encoder_type()
    model.depth_model.decoder = nn.Linear(2, 1)
    return model


class Metric3DCheckpointTest(unittest.TestCase):
    def test_complete_checkpoint_needs_no_compatibility(self):
        model = teacher_fixture()
        report = load_metric3d_state(model, model.state_dict())
        self.assertEqual(report["omitted_non_inference_keys"], [])
        self.assertTrue(hasattr(model.depth_model.encoder, "mask_token"))

    def test_only_unused_mask_parameter_may_be_absent(self):
        model = teacher_fixture()
        state = {key: value.clone() for key, value in model.state_dict().items() if key != UNUSED_MASK_TOKEN_KEY}
        report = load_metric3d_state(model, state)
        self.assertEqual(report["compatibility"], MASK_TOKEN_COMPATIBILITY)
        self.assertEqual(report["omitted_non_inference_keys"], [UNUSED_MASK_TOKEN_KEY])
        self.assertFalse(hasattr(model.depth_model.encoder, "mask_token"))
        self.assertEqual(set(model.state_dict()), set(state))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key])
        image = torch.ones(1, 3, 2, 2)
        torch.testing.assert_close(model.depth_model.encoder(image), image)
        with self.assertRaisesRegex(ValueError, "only one unmasked"):
            model.depth_model.encoder(image, masks=torch.ones(1, dtype=torch.bool))

    def test_missing_encoder_or_decoder_weights_still_fail(self):
        for key in ("depth_model.encoder.projection.weight", "depth_model.decoder.bias"):
            model = teacher_fixture()
            state = dict(model.state_dict())
            del state[UNUSED_MASK_TOKEN_KEY], state[key]
            with self.assertRaisesRegex(ValueError, "Incomplete/mismatched"):
                load_metric3d_state(model, state)
            self.assertTrue(hasattr(model.depth_model.encoder, "mask_token"))

    def test_unexpected_parameter_still_fails(self):
        model = teacher_fixture()
        state = dict(model.state_dict())
        state["unknown_head.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "Incomplete/mismatched"):
            load_metric3d_state(model, state)

    def test_shape_mismatch_still_fails(self):
        model = teacher_fixture()
        state = dict(model.state_dict())
        state["depth_model.decoder.weight"] = torch.zeros(4, 4)
        with self.assertRaises(RuntimeError):
            load_metric3d_state(model, state)

    def test_changed_mask_semantics_fail_closed(self):
        model = teacher_fixture(UnsafeEncoderFixture)
        state = dict(model.state_dict())
        del state[UNUSED_MASK_TOKEN_KEY]
        with self.assertRaisesRegex(ValueError, "outside the audited"):
            load_metric3d_state(model, state)


if __name__ == "__main__":
    unittest.main()
