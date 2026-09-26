import json
import sys

import pytest

from vbc.cli import main


def test_cli_validates_and_excludes_participant_ids(tmp_path, monkeypatch):
    csv_file = tmp_path / "endpoints.csv"
    csv_file.write_text("participant_id,A,B\nS01,1,3\nS02,2,2\nS03,3,1\n", encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vbc",
            str(csv_file),
            "--id-column",
            "participant_id",
            "--primary",
            "A",
            "--output",
            str(output),
        ],
    )

    main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["observation_count"] == 3
    assert payload["preprocessing"] == "none"
    assert set(payload["endpoints"]) == {"B"}
    assert payload["participant_ids"] == ["S01", "S02", "S03"]


@pytest.mark.parametrize("data, message", [
    ("A,A\n1,2\n3,4\n", "unique"),
    ("A,B\n1,2,3\n3,4\n", "number of CSV fields"),
    ("A,B\n1\n3,4\n", "number of CSV fields"),
    ("A,B\n1,nan\n3,4\n", "non-finite"),
])
def test_cli_rejects_malformed_data(tmp_path, monkeypatch, data, message):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(data, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["vbc", str(csv_file), "--primary", "A"])
    with pytest.raises(SystemExit, match=message):
        main()


def test_cli_rejects_duplicate_participant_ids(tmp_path, monkeypatch):
    csv_file = tmp_path / "endpoints.csv"
    csv_file.write_text("participant_id,A,B\nS01,1,3\nS01,2,2\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vbc", str(csv_file), "--id-column", "participant_id", "--primary", "A"],
    )

    with pytest.raises(SystemExit, match="participant IDs must be unique"):
        main()
