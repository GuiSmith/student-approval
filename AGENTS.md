# Repository Guidelines

- Do not create `.codex` or `.agents` directory

## Project Structure & Module Organization

- `main.py`: API application, request validation, model artifact loading, and `/health` plus `/predict` endpoints.
- `test_api.py`: script-based integration tests that call a running API and write logs.
- `modelo/student_approval_model.pkl`: serialized model artifact. `main.py` also checks `model/student_approval_model.pkl`.
- `modelo/IA-Trabalho-Final-Relatorio.pdf`: project report/reference material.
- `requirements.txt`: Python runtime dependencies.
- `Dockerfile`: container build and default Uvicorn startup command.

Generated logs go to `logs/` by default and should not be treated as source.

## Build, Test, and Development Commands

Start or rebuild the containerized app from the repository root:

```bash
make up
```

`make up` runs `docker compose up -d --build` through the project `Makefile`.

Run checks inside the running container, not from the host Python environment:

```bash
docker exec ml-api python -m py_compile main.py
```

Run integration tests inside the container:

```bash
docker exec ml-api python test_api.py
```

Override test targets or log location inside the container when needed:

```bash
docker exec ml-api sh -c 'API_BASE_URL=http://127.0.0.1:8000 LOG_DIR=logs python test_api.py'
```

Use Docker Compose directly only when `make` is unavailable:

```bash
docker compose up -d --build
```

## Coding Style & Naming Conventions

Use Python 3.11-compatible code with 4-space indentation and type hints for API-facing functions. Keep constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions/variables in `snake_case`. Follow the current validation helper pattern.

No formatter or linter is configured. If adding one, document the command here and avoid mixing formatting-only edits with behavior changes.

## Testing Guidelines

Tests are integration-style checks implemented in `test_api.py` with `requests`; they are not pytest tests. Start the API with `make up`, then run `docker exec ml-api python test_api.py`. Add cases to the `tests` list in `main()` and create focused payload helper functions for custom inputs. Use descriptive log filenames, such as `campos_faltantes.log.1`.

Cover both successful predictions and validation failures when changing request fields, domain ranges, artifact loading, or response shape.

## Commit & Pull Request Guidelines

The current history only contains `first commit`, so there is no established convention. Use short imperative commit messages, for example `Add payload validation tests` or `Update model artifact lookup`.

Pull requests should include a concise summary, containerized test results (`make up` plus `docker exec ml-api python test_api.py`, or why not run), and notes for model artifact changes. Link related issues when available and include request/response examples for API behavior changes.

## Security & Configuration Tips

Do not commit secrets, credentials, or private datasets. Keep large generated files and logs out of commits unless intentional. Validate that `modelo/student_approval_model.pkl` is present before building or deploying.
