"""Small CPU checks exercise the unchanged native H3/L6 recursion."""
import pytest
import torch
import torch.nn.functional as F

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config():
    return {"batch_size": 2, "seq_len": 9, "puzzle_emb_ndim": 0,
            "puzzle_emb_len": 0, "num_puzzle_identifiers": 1, "vocab_size": 11,
            "H_cycles": 3, "L_cycles": 6, "H_layers": 0, "L_layers": 2,
            "hidden_size": 32, "expansion": 2, "num_heads": 4,
            "pos_encodings": "rope", "forward_dtype": "float32",
            "halt_max_steps": 16, "halt_exploration_prob": .1,
            "controller_width": 128, "controller_heads": 8, "controller_layers": 3,
            "controller_ff_multiplier": 4, "controller_history": 4,
            "adapter_rank": 4, "adapter_alpha": .2}


def batch():
    return {"inputs": torch.tensor([[1, 2, 1, 4, 1, 6, 1, 8, 1], [9, 1, 7, 1, 5, 1, 3, 1, 1]]),
            "labels": torch.tensor([[2, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 2]]),
            "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}


def models():
    torch.manual_seed(17)
    native = TinyRecursiveReasoningModel_ACTV1(config())
    torch.manual_seed(29)
    plastic = PlasticTinyRecursiveReasoningModel_ACTV1(config())
    plastic.load_native_state(native.state_dict())
    return native, plastic


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_off_is_bitwise_native_for_repeated_ACT_updates():
    native, plastic = models()
    native.eval()
    plastic.eval().set_mode("off")
    data = batch()
    nc, pc = native.initial_carry(data), plastic.initial_carry(data)
    with torch.no_grad():
        for _ in range(3):
            nc, no = native(nc, data)
            pc, po = plastic(pc, data)
            for name in no:
                assert torch.equal(no[name], po[name]), name
            assert torch.equal(nc.inner_carry.z_H, pc.inner_carry.z_H)
            assert torch.equal(nc.inner_carry.z_L, pc.inner_carry.z_L)
            assert torch.equal(nc.halted, pc.halted)
    assert plastic.last_trace["controller_calls"] == 0
    assert all(adapter.gate is None for adapter in plastic.adapters)


def test_native_recursion_gradient_schedule_and_dynamic_gradients():
    _, model = models()
    model.configure_training("joint").train()
    data = batch()
    carry, outputs = model(model.initial_carry(data), data)
    trace = model.last_trace
    assert trace["reasoning_calls"] == trace["controller_calls"] == 21
    assert trace["gradient_disabled_calls"] == 14
    assert trace["gradient_enabled_calls"] == 7
    assert [row["grad_enabled"] for row in trace["calls"]] == [False] * 14 + [True] * 7
    assert all(not row["gate_requires_grad"] for row in trace["calls"][:14])
    assert all(row["gate_requires_grad"] for row in trace["calls"][14:])
    assert all(row["gate"].abs().max() <= 1 for row in trace["calls"])
    assert max(row["history_length"] for row in trace["calls"]) == 4
    loss = F.cross_entropy(outputs["logits"].reshape(-1, 11), data["labels"].reshape(-1))
    loss.backward()
    assert model.controller.observation.weight.grad.norm() > 0
    assert model.controller.transformer.layers[0].self_attn.in_proj_weight.grad.norm() > 0
    assert model.controller.gates.weight.grad.norm() > 0
    for adapter in model.adapters:
        assert adapter.u.grad.norm() > 0 and adapter.v.grad.norm() > 0
        assert adapter.base.weight.grad.norm() > 0
    assert not carry.inner_carry.z_H.requires_grad
    assert not carry.inner_carry.z_L.requires_grad
    assert all(adapter.gate is None for adapter in model.adapters)


def test_labels_are_never_controller_observations():
    _, model = models()
    model.eval()
    data = batch()
    changed = dict(data, labels=10 - data["labels"])
    with torch.no_grad():
        _, first = model(model.initial_carry(data), data)
        first_gates = [row["gate"].clone() for row in model.last_trace["calls"]]
        _, second = model(model.initial_carry(changed), changed)
    assert torch.equal(first["logits"], second["logits"])
    assert all(torch.equal(left, right["gate"]) for left, right in zip(first_gates, model.last_trace["calls"]))


@pytest.mark.parametrize("mode", ["off", "dynamic", "static", "fixed"])
def test_inference_has_no_parameter_mutation_and_no_cross_batch_history(mode):
    _, model = models()
    model.eval().set_mode(mode)
    data = batch()
    before = snapshot(model)
    with torch.no_grad():
        _, first = model(model.initial_carry(data), data)
        other = dict(data, inputs=data["inputs"].flip(0))
        model(model.initial_carry(other), other)
        _, repeated = model(model.initial_carry(data), data)
        singles = []
        for index in range(2):
            single = {name: value[index:index + 1] for name, value in data.items()}
            _, out = model(model.initial_carry(single), single)
            singles.append(out["logits"])
    assert torch.equal(first["logits"], repeated["logits"])
    torch.testing.assert_close(first["logits"], torch.cat(singles), rtol=1e-5, atol=2e-6)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(adapter.gate is None for adapter in model.adapters)
    assert model._history == [] and model._previous is None and model._controller_state is None


def test_fixed_uses_the_same_first_gate_for_all_native_calls_and_disallows_training():
    _, model = models()
    data = batch()
    model.eval().set_mode("fixed")
    with torch.no_grad():
        model(model.initial_carry(data), data)
    gates = [row["gate"] for row in model.last_trace["calls"]]
    assert len(gates) == 21 and all(torch.equal(gates[0], gate) for gate in gates)
    model.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        model(model.initial_carry(data), data)


def test_cleanup_runs_when_native_reasoning_raises(monkeypatch):
    _, model = models()
    data = batch()
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate native failure")
    monkeypatch.setattr(model.inner.L_level.layers[0].mlp, "forward", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        model(model.initial_carry(data), data)
    assert all(adapter.gate is None for adapter in model.adapters)
    assert model._history == [] and model._fixed_gate is None


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype"])
def test_invalid_native_state_fails_before_mutation(damage):
    native, model = models()
    state = snapshot(native)
    name = "inner.L_level.layers.0.mlp.down_proj.weight"
    if damage == "missing":
        del state[name]
    elif damage == "shape":
        state[name] = state[name][:1]
    else:
        state[name] = state[name].double()
    before = snapshot(model)
    with pytest.raises(ValueError):
        model.load_native_state(state)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("policy", ["native", "dynamic", "static", "joint", "static_joint", "probe"])
def test_training_policy_exposes_only_requested_components(policy):
    _, model = models()
    model.configure_training(policy)
    names = {name for name, value in model.named_parameters() if value.requires_grad}
    a = {name for name in names if name.startswith("inner.") and not name.endswith((".u", ".v"))}
    uv = {name for name in names if name.endswith((".u", ".v"))}
    b = {name for name in names if name.startswith("controller.")}
    assert bool(a) == (policy in ("native", "joint", "static_joint"))
    assert bool(uv) == (policy in ("dynamic", "static", "joint", "static_joint"))
    assert bool(b) == (policy in ("dynamic", "joint", "probe"))
    assert ("static_gates" in names) == (policy in ("static", "static_joint"))
