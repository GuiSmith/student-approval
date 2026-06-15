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

Set up local dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Run integration tests in another terminal while the API is running:

```bash
python test_api.py
```

Override test targets or log location:

```bash
API_BASE_URL=http://127.0.0.1:8000 LOG_DIR=logs python test_api.py
```

Build and run the container:

```bash
docker build -t student-approval .
docker run --rm -p 8000:8000 student-approval
```

## Coding Style & Naming Conventions

Use Python 3.11-compatible code with 4-space indentation and type hints for API-facing functions. Keep constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions/variables in `snake_case`. Follow the current validation helper pattern.

No formatter or linter is configured. If adding one, document the command here and avoid mixing formatting-only edits with behavior changes.

## Testing Guidelines

Tests are integration-style checks implemented in `test_api.py` with `requests`; they are not pytest tests. Start the API first, then run `python test_api.py`. Add cases to the `tests` list in `main()` and create focused payload helper functions for custom inputs. Use descriptive log filenames, such as `campos_faltantes.log.1`.

Cover both successful predictions and validation failures when changing request fields, domain ranges, artifact loading, or response shape.

## Commit & Pull Request Guidelines

The current history only contains `first commit`, so there is no established convention. Use short imperative commit messages, for example `Add payload validation tests` or `Update model artifact lookup`.

Pull requests should include a concise summary, test results (`python test_api.py`, Docker run, or why not run), and notes for model artifact changes. Link related issues when available and include request/response examples for API behavior changes.

## Security & Configuration Tips

Do not commit secrets, credentials, or private datasets. Keep large generated files and logs out of commits unless intentional. Validate that `modelo/student_approval_model.pkl` is present before building or deploying.
