import pandas as pd
import pytest

from alpha_research.artifacts import ArtifactContract
from scripts import build_strategy_monitoring as monitoring_script
from scripts import refresh_strategy_outputs as refresh_script


def test_monitoring_writer_loads_the_named_inputs_and_returns_the_manifest(
    tmp_path,
    monkeypatch,
):
    processed_directory = tmp_path / "processed"
    monitoring_directory = processed_directory / "monitoring"
    inputs = {name: pd.DataFrame({"source": [name]}) for name in monitoring_script.INPUT_FILENAMES}
    original_inputs = {name: data.copy(deep=True) for name, data in inputs.items()}
    loaded_paths = []
    expected_datasets = {"monitoring": pd.DataFrame({"value": [1.0]})}
    expected_manifest = pd.DataFrame({"dataset": ["monitoring"]})

    def fake_load(path):
        loaded_paths.append(path)
        filename_lookup = {
            filename: name for name, filename in monitoring_script.INPUT_FILENAMES.items()
        }
        return inputs[filename_lookup[path.name]]

    def fake_build(**kwargs):
        assert kwargs.keys() == inputs.keys()
        return expected_datasets

    def fake_write(datasets, directory):
        assert datasets is expected_datasets
        assert directory == monitoring_directory
        return expected_manifest

    monkeypatch.setattr(monitoring_script, "load_parquet", fake_load)
    monkeypatch.setattr(monitoring_script, "build_strategy_monitoring_datasets", fake_build)
    monkeypatch.setattr(monitoring_script, "write_monitoring_artifacts", fake_write)

    actual_manifest = monitoring_script.rebuild_monitoring_artifacts(
        processed_directory,
        monitoring_directory,
    )

    assert actual_manifest is expected_manifest
    assert loaded_paths == [
        processed_directory / filename for filename in monitoring_script.INPUT_FILENAMES.values()
    ]

    for name, original in original_inputs.items():
        pd.testing.assert_frame_equal(inputs[name], original)


@pytest.mark.parametrize(
    ("argv", "expected_exit_code"),
    [(["--help"], 0), (["--not-an-option"], 2)],
    ids=["help", "invalid-argument"],
)
def test_monitoring_cli_inspection_exits_before_rebuilding(
    argv,
    expected_exit_code,
    monkeypatch,
):
    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("CLI inspection attempted to rebuild monitoring artifacts.")

    monkeypatch.setattr(
        monitoring_script,
        "rebuild_monitoring_artifacts",
        unexpected_rebuild,
    )

    with pytest.raises(SystemExit) as error:
        monitoring_script.main(argv)

    assert error.value.code == expected_exit_code


def _install_refresh_stubs(monkeypatch, *, mismatched_reference: bool = False):
    factor_panel = pd.DataFrame({"factor_input": [1.0]})
    benchmark_prices = pd.DataFrame({"benchmark_input": [2.0]})
    attribution = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "value": [3.0]})
    monitoring = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "value": [4.0]})
    refreshed = {
        "attribution": {"attribution_test": attribution},
        "monitoring": {"monitoring_test": monitoring},
    }
    references = {
        "attribution_test.parquet": attribution.copy(deep=True),
        "monitoring_test.parquet": monitoring.copy(deep=True),
    }

    if mismatched_reference:
        references["monitoring_test.parquet"].loc[0, "value"] = -1.0

    monkeypatch.setattr(
        refresh_script,
        "ATTRIBUTION_ARTIFACT_CONTRACTS",
        {
            "attribution_test": ArtifactContract(
                filename="attribution_test.parquet",
                key_columns=("date",),
                required_columns=("date", "value"),
            )
        },
    )
    monkeypatch.setattr(
        refresh_script,
        "MONITORING_ARTIFACT_CONTRACTS",
        {
            "monitoring_test": ArtifactContract(
                filename="monitoring_test.parquet",
                key_columns=("date",),
                required_columns=("date", "value"),
            )
        },
    )

    def fake_load(path):
        if path.name == "factor_panel.parquet":
            return factor_panel
        if path.name == "spy_benchmark.parquet":
            return benchmark_prices
        return references[path.name]

    def fake_build(actual_factor_panel, actual_benchmark_prices):
        assert actual_factor_panel is factor_panel
        assert actual_benchmark_prices is benchmark_prices
        return refreshed

    monkeypatch.setattr(refresh_script, "load_parquet", fake_load)
    monkeypatch.setattr(refresh_script, "build_complete_research_refresh", fake_build)
    monkeypatch.setattr(refresh_script, "validate_attribution_artifacts", lambda data: None)
    monkeypatch.setattr(refresh_script, "validate_monitoring_artifacts", lambda data: None)

    return refreshed


def test_refresh_script_dry_run_reconciles_without_writing(
    tmp_path,
    monkeypatch,
    capsys,
):
    _install_refresh_stubs(monkeypatch)

    def unexpected_write(*args, **kwargs):
        raise AssertionError("Dry-run mode attempted to write artifacts.")

    monkeypatch.setattr(refresh_script, "write_attribution_artifacts", unexpected_write)
    monkeypatch.setattr(refresh_script, "write_monitoring_artifacts", unexpected_write)

    refresh_script.run_strategy_refresh(
        processed_directory=tmp_path / "processed",
        raw_directory=tmp_path / "raw",
        monitoring_directory=tmp_path / "processed" / "monitoring",
    )

    output = capsys.readouterr().out
    assert "[1/6] Loading input datasets (dry run)..." in output
    assert "Checked attribution.attribution_test: PASS" in output
    assert "Checked monitoring.monitoring_test: PASS" in output
    assert "All refresh reconciliations pass: True" in output
    assert "Dry run only: no artifacts were written." in output


def test_refresh_script_dry_run_rejects_value_mismatches(tmp_path, monkeypatch):
    _install_refresh_stubs(monkeypatch, mismatched_reference=True)

    with pytest.raises(ValueError, match="reconciliation failed"):
        refresh_script.run_strategy_refresh(
            processed_directory=tmp_path / "processed",
            raw_directory=tmp_path / "raw",
            monitoring_directory=tmp_path / "processed" / "monitoring",
        )


def test_refresh_script_write_mode_uses_explicit_artifact_destinations(
    tmp_path,
    monkeypatch,
    capsys,
):
    refreshed = _install_refresh_stubs(monkeypatch)
    processed_directory = tmp_path / "processed"
    monitoring_directory = processed_directory / "monitoring"
    write_calls = []

    def fake_attribution_write(datasets, directory):
        write_calls.append(("attribution", datasets, directory))
        return pd.DataFrame({"dataset": ["attribution_test"]})

    def fake_monitoring_write(datasets, directory):
        write_calls.append(("monitoring", datasets, directory))
        return pd.DataFrame({"dataset": ["monitoring_test"]})

    monkeypatch.setattr(
        refresh_script,
        "write_attribution_artifacts",
        fake_attribution_write,
    )
    monkeypatch.setattr(
        refresh_script,
        "write_monitoring_artifacts",
        fake_monitoring_write,
    )

    refresh_script.run_strategy_refresh(
        write=True,
        processed_directory=processed_directory,
        raw_directory=tmp_path / "raw",
        monitoring_directory=monitoring_directory,
    )

    assert write_calls == [
        ("attribution", refreshed["attribution"], processed_directory),
        ("monitoring", refreshed["monitoring"], monitoring_directory),
    ]
    output = capsys.readouterr().out
    assert "[1/6] Loading input datasets (write)..." in output
    assert "Attribution artifacts written and read back." in output
    assert "Monitoring artifacts written and read back." in output
    assert "All strategy outputs were refreshed successfully." in output


def test_refresh_main_routes_dry_run_and_write_arguments(monkeypatch):
    calls = []

    monkeypatch.setattr(
        refresh_script,
        "run_strategy_refresh",
        lambda *, write=False: calls.append(write),
    )

    refresh_script.main([])
    refresh_script.main(["--write"])

    assert calls == [False, True]
