"""Runtime helper tests: precision, device, AMP, compile, parallelism."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from training.models import ModelFactory
from training.models import (
    amp_context,
    apply_precision,
    apply_runtime,
    compile_model,
    dtype_from_precision,
    enable_gradient_checkpointing,
    precision_from_dtype,
    resolve_device,
    wrap_data_parallel,
    wrap_distributed,
)
from training.models.exceptions import ModelConfigurationError


def test_resolve_device_default():
    dev = resolve_device()
    assert dev.type in ("cpu", "cuda")


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"


def test_dtype_roundtrip():
    assert dtype_from_precision("bfloat16") is torch.bfloat16
    assert dtype_from_precision("float16") is torch.float16
    assert dtype_from_precision("float32") is torch.float32
    assert precision_from_dtype(torch.float16) == "float16"
    assert precision_from_dtype(torch.int64) == "float32"


def test_dtype_unknown_raises():
    with pytest.raises(ModelConfigurationError):
        dtype_from_precision("float8")


def test_amp_float32_is_noop(model, batch):
    with amp_context("float32", "cpu"):
        out = model(batch)
    assert out.crop_logits.shape == (4, 3)


def test_amp_bfloat16_cpu(model, batch):
    with torch.no_grad():
        with amp_context("bfloat16", "cpu"):
            out = model(batch)
    assert out.crop_logits.shape == (4, 3)


def test_amp_float16_cpu_or_clean_error():
    # float16 autocast support on CPU varies by PyTorch build; the helper must
    # either run the block or raise a clear ModelConfigurationError.
    try:
        with amp_context("float16", "cpu"):
            pass
    except ModelConfigurationError:
        pass


def test_apply_precision_norms_stay_float32(tabular_only_config):
    model = ModelFactory.create(tabular_only_config)
    try:
        model.to_precision("bfloat16")
        assert next(model.parameters()).dtype == torch.bfloat16
        for mod in model.modules():
            if isinstance(mod, nn.LayerNorm):
                for param in mod.parameters():
                    assert param.dtype == torch.float32
        assert model.config.runtime.precision == "bfloat16"
    finally:
        model.to_precision("float32")
    assert model.config.runtime.precision == "float32"


def test_apply_precision_float16_forward(tabular_only_config):
    model = ModelFactory.create(tabular_only_config)
    model.to_precision("float16")
    # fp16 weights consume fp16 tensors (or autocast) — a realistic inference
    # path casts the batch alongside the model.
    batch = {
        key: value.half() if value.is_floating_point() else value
        for key, value in model.sample_batch(batch_size=2).items()
    }
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.crop_logits.shape == (2, 4)
    model.to_precision("float32")


def test_gradient_checkpointing_toggle(tabular_only_config):
    model = ModelFactory.create(tabular_only_config)
    try:
        model.enable_gradient_checkpointing(True)
        assert model.config.runtime.gradient_checkpointing is True
        assert model.tab_encoder.gradient_checkpointing is True
        assert model.shared_encoder.gradient_checkpointing is True
        model.enable_gradient_checkpointing(False)
        assert model.tab_encoder.gradient_checkpointing is False
        assert model.config.runtime.gradient_checkpointing is False
    finally:
        model.enable_gradient_checkpointing(False)


def test_gradient_checkpointing_multimodal_forward_backward(model, batch):
    model.train()
    model.enable_gradient_checkpointing(True)
    try:
        out = model(batch)
        (out.crop_logits.sum() + out.yield_pred.sum()).backward()
        missing = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert missing == []
    finally:
        model.enable_gradient_checkpointing(False)
        model.zero_grad()


def test_compile_eager_forward(model, batch):
    compiled = compile_model(model, mode="default", backend="eager")
    with torch.no_grad():
        out = compiled(batch)
    assert tuple(out.crop_logits.shape) == (4, 3)


def test_compile_via_model_method(model, batch):
    compiled = model.compile(backend="eager")
    with torch.no_grad():
        out = compiled(batch)
    assert out.shared_representation.shape == (4, 128)


def test_data_parallel_requires_cuda():
    if torch.cuda.is_available():
        pytest.skip("requires a CPU-only environment")
    with pytest.raises(ModelConfigurationError):
        wrap_data_parallel(nn.Linear(2, 2))


def test_distributed_requires_init():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pytest.skip("distributed already initialized")
    with pytest.raises(ModelConfigurationError):
        wrap_distributed(nn.Linear(2, 2))


def test_apply_runtime_applies_precision_and_checkpointing(tabular_only_config):
    model = ModelFactory.create(tabular_only_config)
    model.config.runtime.precision = "bfloat16"
    model.config.runtime.gradient_checkpointing = True
    try:
        model = apply_runtime(model)
        assert model.config.runtime.precision == "bfloat16"
        assert next(model.parameters()).dtype == torch.bfloat16
        assert model.tab_encoder.gradient_checkpointing is True
    finally:
        model.config.runtime.gradient_checkpointing = False
        model.to_precision("float32")


def test_create_with_runtime(tabular_only_config):
    tabular_only_config.runtime.gradient_checkpointing = True
    model = ModelFactory.create_with_runtime(tabular_only_config)
    assert model.tab_encoder.gradient_checkpointing is True
    model.enable_gradient_checkpointing(False)
