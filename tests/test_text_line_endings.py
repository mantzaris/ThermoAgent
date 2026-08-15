"""Repository policy tests for generated and maintained v3/v4 text artifacts."""

from pathlib import Path


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def test_v3_v4_and_generators_do_not_contain_crlf() -> None:
    """New work must not recreate the reviewed CRLF artifact problem.

    Frozen v1/v2 result namespaces are intentionally outside this test's
    scope: the v4 cleanup must not mechanically rewrite those studies.
    """

    repository = Path(__file__).resolve().parents[1]
    roots = [
        repository / "thermoagent",
        repository / "scripts",
        repository / "configs",
        repository / "results" / "human_operator_v3",
        repository / "results" / "human_operator_v4",
    ]
    offenders = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                if b"\r\n" in path.read_bytes():
                    offenders.append(str(path.relative_to(repository)))
    assert not offenders, "CRLF text artifacts found: %s" % ", ".join(offenders)
