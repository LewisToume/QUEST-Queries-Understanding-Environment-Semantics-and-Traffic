"""Strict Metric3D loading with one source-audited, non-inference omission."""

import ast
import hashlib
import inspect
import textwrap

import torch


UNUSED_MASK_TOKEN_KEY = "depth_model.encoder.mask_token"
MASK_TOKEN_COMPATIBILITY = "metric3d_unmasked_encoder_mask_token_v1"


def verify_mask_token_source(encoder):
    """Fail closed if the server encoder reads mask_token outside masks!=None."""
    try:
        source = textwrap.dedent(inspect.getsource(type(encoder)))
        tree = ast.parse(source)
        parameter = inspect.signature(encoder.prepare_tokens_with_masks).parameters["masks"]
    except (OSError, TypeError, SyntaxError, AttributeError, KeyError, ValueError) as error:
        raise ValueError("Cannot audit native encoder mask_token source; refusing compatibility") from error
    reads = {node for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and node.attr == "mask_token"
             and isinstance(node.value, ast.Name) and node.value.id == "self"
             and isinstance(node.ctx, ast.Load)}
    guarded = set()
    for method in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name == "prepare_tokens_with_masks"):
        for branch in (node for node in ast.walk(method) if isinstance(node, ast.If)):
            test = branch.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "masks"
                    and len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot)
                    and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None):
                for statement in branch.body:
                    guarded.update(ast.walk(statement))
    if parameter.default is not None or not reads or not reads.issubset(guarded):
        raise ValueError("Native encoder uses mask_token outside the audited masked-input branch")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def require_unmasked_tensor(module, args, kwargs):
    if (len(args) != 1 or not torch.is_tensor(args[0]) or args[0].ndim != 4
            or any(value is not None for value in kwargs.values())):
        raise ValueError("Checkpoint omits mask_token: only one unmasked image tensor is supported")


def load_metric3d_state(model, state):
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    print(f"checkpoint_missing_keys={missing} unexpected_keys={unexpected}")
    if unexpected or set(missing) - {UNUSED_MASK_TOKEN_KEY}:
        raise ValueError("Incomplete/mismatched teacher weights; no random-head label export allowed")
    report = {"compatibility": None, "omitted_non_inference_keys": [], "strict": True}
    if missing:
        encoder = model.get_submodule("depth_model.encoder")
        token = getattr(encoder, "mask_token", None)
        if not isinstance(token, torch.nn.Parameter) or tuple(token.shape) != (1, 1024):
            raise ValueError("Unexpected Metric3Dv2 ViT-L mask_token layout")
        report["encoder_source_sha256"] = verify_mask_token_source(encoder)
        # Match the official encoder's inference-only parameter lifecycle. Do
        # not synthesize a tensor or modify the downloaded checkpoint/source.
        del encoder.mask_token
        encoder.register_forward_pre_hook(require_unmasked_tensor, with_kwargs=True)
        report.update(compatibility=MASK_TOKEN_COMPATIBILITY, omitted_non_inference_keys=missing)
    model.load_state_dict(state, strict=True)
    print("checkpoint_inference_weights=STRICT_PASS omitted_non_inference_keys=" + str(report["omitted_non_inference_keys"]))
    return report
