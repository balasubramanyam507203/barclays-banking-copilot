from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_project_file(
    relative_path: str,
) -> str:
    path = PROJECT_ROOT / relative_path

    assert path.exists(), (
        f"Expected Docker file is missing: {path}"
    )

    return path.read_text(
        encoding="utf-8"
    )


def test_compose_defines_required_services() -> None:
    compose_text = read_project_file(
        "compose.yaml"
    )

    for service_name in (
        "postgres:",
        "redis:",
        "backend:",
        "frontend:",
    ):
        assert service_name in compose_text

    assert (
        "condition: service_healthy"
        in compose_text
    )


def test_backend_image_runs_migrations() -> None:
    entrypoint_text = read_project_file(
        "backend/docker-entrypoint.sh"
    )

    assert (
        "alembic upgrade head"
        in entrypoint_text
    )

    assert (
        "python -m app.build_index"
        in entrypoint_text
    )

    assert (
        "uvicorn app.api.main:app"
        in entrypoint_text
    )


def test_frontend_uses_standalone_output() -> None:
    next_config_text = read_project_file(
        "frontend/next.config.ts"
    )

    dockerfile_text = read_project_file(
        "frontend/Dockerfile"
    )

    assert (
        'output: "standalone"'
        in next_config_text
    )

    assert (
        "/app/.next/standalone"
        in dockerfile_text
    )


def test_secret_files_are_ignored() -> None:
    ignore_additions = read_project_file(
        "step25_gitignore_additions.txt"
    )

    assert ".env.docker" in ignore_additions
    assert ".docker-secrets/" in ignore_additions
