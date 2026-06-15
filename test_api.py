import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))


BASE_PAYLOAD: dict[str, Any] = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "teacher",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 4,
    "subject": "math",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _write_log(
    filename: str,
    test_name: str,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    status_code: int | None,
    response_json: Any,
    passed: bool,
    error: str | None = None,
) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / filename
    lines = [
        f"test_name: {test_name}",
        f"timestamp: {_timestamp()}",
        f"method: {method}",
        f"url: {url}",
    ]
    if payload is not None:
        lines.extend(["payload:", _format_json(payload)])
    lines.append(f"status_code: {status_code}")
    lines.extend(["response_json:", _format_json(response_json)])
    if error:
        lines.append(f"error: {error}")
    lines.append(f"result: {'PASS' if passed else 'FAIL'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _request_json(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[int | None, Any, str | None]:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, json=payload, timeout=10)
    except requests.RequestException as exc:
        return None, None, str(exc)

    try:
        response_json = response.json()
    except ValueError:
        response_json = response.text

    return response.status_code, response_json, None


def _valid_payload() -> dict[str, Any]:
    return deepcopy(BASE_PAYLOAD)


def _missing_fields_payload() -> dict[str, Any]:
    payload = _valid_payload()
    del payload["school"]
    del payload["age"]
    return payload


def _extra_fields_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["extra_field"] = "unexpected"
    return payload


def _forbidden_fields_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["G1"] = 10
    payload["G2"] = 11
    payload["G3"] = 12
    return payload


def _invalid_categories_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["school"] = "ABC"
    payload["sex"] = "X"
    payload["subject"] = "science"
    return payload


def _empty_strings_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["school"] = "   "
    payload["Mjob"] = ""
    return payload


def _out_of_range_numbers_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["age"] = 30
    payload["absences"] = 94
    payload["health"] = 0
    return payload


def _bool_in_number_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["Medu"] = True
    return payload


def _decimal_in_integer_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["studytime"] = 2.5
    return payload


def _multiple_errors_payload() -> dict[str, Any]:
    payload = _valid_payload()
    payload["school"] = "ABC"
    payload["subject"] = "   "
    payload["age"] = 30
    payload["Medu"] = True
    payload["studytime"] = 2.5
    payload["G1"] = 10
    payload["unexpected"] = "value"
    return payload


def _is_success_response(status_code: int | None, response_json: Any) -> bool:
    return (
        status_code == 200
        and isinstance(response_json, dict)
        and response_json.get("success") is True
    )


def _run_test(test: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_BASE_URL}{test['path']}"
    status_code, response_json, error = _request_json(
        test["method"], test["path"], test.get("payload")
    )
    passed = bool(test["assertion"](status_code, response_json, error))
    log_path = _write_log(
        test["log_file"],
        test["name"],
        test["method"],
        url,
        test.get("payload"),
        status_code,
        response_json,
        passed,
        error,
    )
    return {"name": test["name"], "passed": passed, "log_path": log_path}


def main() -> int:
    tests = [
        {
            "name": "GET /health",
            "log_file": "health.log.1",
            "method": "GET",
            "path": "/health",
            "assertion": lambda status, body, error: (
                error is None and status == 200 and body == {"status": "ok"}
            ),
        },
        {
            "name": "POST /predict com payload valido",
            "log_file": "predict_valido.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _valid_payload(),
            "assertion": lambda status, body, error: (
                error is None and _is_success_response(status, body)
            ),
        },
        {
            "name": "POST /predict com campos faltantes",
            "log_file": "campos_faltantes.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _missing_fields_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com campos extras",
            "log_file": "campos_extras.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _extra_fields_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com campos proibidos G1, G2 e G3",
            "log_file": "campos_proibidos.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _forbidden_fields_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com valores categoricos invalidos",
            "log_file": "categoricas_invalidas.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _invalid_categories_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com strings vazias",
            "log_file": "strings_vazias.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _empty_strings_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com valores numericos fora do intervalo",
            "log_file": "numericos_fora_intervalo.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _out_of_range_numbers_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com bool em campo numerico",
            "log_file": "bool_em_numerico.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _bool_in_number_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com decimal em campo inteiro",
            "log_file": "decimal_em_inteiro.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _decimal_in_integer_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
        {
            "name": "POST /predict com multiplos erros no mesmo payload",
            "log_file": "multiplos_erros.log.1",
            "method": "POST",
            "path": "/predict",
            "payload": _multiple_errors_payload(),
            "assertion": lambda status, body, error: error is None and status == 422,
        },
    ]

    results = [_run_test(test) for test in tests]
    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed

    print(f"Total de testes: {len(results)}")
    print(f"Testes aprovados: {passed}")
    print(f"Testes com falha: {failed}")
    print(f"Caminho dos logs gerados: {LOG_DIR.resolve()}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
