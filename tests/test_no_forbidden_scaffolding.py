from pathlib import Path


def test_no_forbidden_service_or_container_files() -> None:
    root = Path(__file__).resolve().parents[1]
    names = {path.name.lower() for path in root.rglob("*") if ".git" not in path.parts}

    forbidden = {
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "launchagent.plist",
        "crontab",
    }

    assert forbidden.isdisjoint(names)
    assert not list(root.rglob("*.plist"))
