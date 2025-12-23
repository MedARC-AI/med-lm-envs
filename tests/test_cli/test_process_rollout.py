from medarc_verifiers.cli.process.rollout import derive_base_env_id


def test_derive_base_env_id_strips_suffix() -> None:
    base, index = derive_base_env_id("medqa-rollout12")
    assert base == "medqa"
    assert index == 12


def test_derive_base_env_id_handles_simple_r_suffix() -> None:
    base, index = derive_base_env_id("medbullets-r3")
    assert base == "medbullets"
    assert index == 3


def test_derive_base_env_id_respects_combine_flag() -> None:
    base, index = derive_base_env_id("medqa-rollout7", combine_rollouts=False)
    assert base == "medqa-rollout7"
    assert index == 0


def test_derive_base_env_id_handles_missing_value() -> None:
    base, index = derive_base_env_id(None)
    assert base == ""
    assert index == 0
