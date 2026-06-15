from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


ARTIFACT_PATHS = (
    Path("model/student_approval_model.pkl"),
    Path("modelo/student_approval_model.pkl"),
)
FORBIDDEN_FIELDS = {"G1", "G2", "G3"}
CATEGORICAL_CHOICES = {
    "school": ("GP", "MS"),
    "sex": ("F", "M"),
    "address": ("U", "R"),
    "famsize": ("LE3", "GT3"),
    "Pstatus": ("T", "A"),
    "Mjob": ("teacher", "health", "services", "at_home", "other"),
    "Fjob": ("teacher", "health", "services", "at_home", "other"),
    "reason": ("home", "reputation", "course", "other"),
    "guardian": ("mother", "father", "other"),
    "schoolsup": ("yes", "no"),
    "famsup": ("yes", "no"),
    "paid": ("yes", "no"),
    "activities": ("yes", "no"),
    "nursery": ("yes", "no"),
    "higher": ("yes", "no"),
    "internet": ("yes", "no"),
    "romantic": ("yes", "no"),
    "subject": ("math", "portuguese"),
}
INT_RANGES = {
    "age": (15, 22),
    "absences": (0, 93),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
}


class PredictionInput(BaseModel):
    class Config:
        extra = "allow"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _find_artifact_path() -> Path:
    for path in ARTIFACT_PATHS:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in ARTIFACT_PATHS)
    raise RuntimeError(f"Model artifact not found. Searched: {searched}")


def _load_artifact() -> dict[str, Any]:
    artifact_path = _find_artifact_path()
    artifact = joblib.load(artifact_path)
    required_keys = {
        "model",
        "encoder",
        "scaler",
        "categorical_features",
        "features_to_scale",
        "expected_columns",
        "target_labels",
    }
    missing = required_keys - set(artifact)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise RuntimeError(f"Artifact is missing required keys: {missing_keys}")
    return artifact


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = _load_artifact()
    yield


app = FastAPI(title="Student Approval Prediction API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _raise_validation_errors(errors: list[dict[str, Any]]) -> None:
    if not errors:
        return
    detail: dict[str, Any] | list[dict[str, Any]]
    detail = errors[0] if len(errors) == 1 else errors
    raise HTTPException(status_code=422, detail=detail)


def _validate_payload(
    payload: dict[str, Any], expected_columns: list[str]
) -> list[dict[str, Any]]:
    received_fields = set(payload)
    expected_fields = set(expected_columns)
    errors: list[dict[str, Any]] = []

    forbidden_received = received_fields & FORBIDDEN_FIELDS
    for field in sorted(forbidden_received):
        errors.append(
            {
                "field": field,
                "message": "Field is not accepted by this API.",
                "received": payload[field],
            }
        )

    missing = expected_fields - received_fields
    extra = received_fields - expected_fields
    for field in sorted(missing):
        errors.append(
            {
                "field": field,
                "message": "Field is required.",
                "received": None,
            }
        )
    for field in sorted(extra - FORBIDDEN_FIELDS):
        errors.append(
            {
                "field": field,
                "message": "Unexpected field.",
                "received": payload[field],
            }
        )

    return errors


def _validate_string_choice(
    field: str, value: Any, choices: tuple[str, ...]
) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(value, str):
        return None, {
            "field": field,
            "message": "Value must be a string.",
            "received": value,
        }

    normalized = value.strip()
    if not normalized:
        return None, {
            "field": field,
            "message": "Value must not be empty.",
            "received": value,
        }

    if normalized not in choices:
        allowed = ", ".join(choices)
        return None, {
            "field": field,
            "message": f"Value must be one of: {allowed}.",
            "received": value,
        }

    return normalized, None


def _validate_int_range(
    field: str, value: Any, minimum: int, maximum: int
) -> tuple[int | None, dict[str, Any] | None]:
    message = f"Value must be an integer between {minimum} and {maximum}."

    if isinstance(value, bool):
        return None, {"field": field, "message": message, "received": value}

    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdigit():
            return None, {"field": field, "message": message, "received": value}
        normalized = int(stripped)
    else:
        return None, {"field": field, "message": message, "received": value}

    if not minimum <= normalized <= maximum:
        return None, {"field": field, "message": message, "received": value}

    return normalized, None


def _validate_domain_payload(
    payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_payload: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    for field, choices in CATEGORICAL_CHOICES.items():
        if field not in payload:
            continue
        normalized, error = _validate_string_choice(field, payload[field], choices)
        if error:
            errors.append(error)
        else:
            normalized_payload[field] = normalized

    for field, (minimum, maximum) in INT_RANGES.items():
        if field not in payload:
            continue
        normalized, error = _validate_int_range(
            field, payload[field], minimum, maximum
        )
        if error:
            errors.append(error)
        else:
            normalized_payload[field] = normalized

    return normalized_payload, errors


def _transform_payload(payload: dict[str, Any], artifact: dict[str, Any]) -> pd.DataFrame:
    expected_columns = list(artifact["expected_columns"])
    categorical_features = list(artifact["categorical_features"])
    features_to_scale = list(artifact["features_to_scale"])

    errors = _validate_payload(payload, expected_columns)
    payload, domain_errors = _validate_domain_payload(payload)
    errors.extend(domain_errors)
    _raise_validation_errors(errors)

    df = pd.DataFrame([payload], columns=expected_columns)

    for feature in features_to_scale:
        df[feature] = pd.to_numeric(df[feature], errors="raise")

    processed = df.copy()
    processed[features_to_scale] = artifact["scaler"].transform(
        processed[features_to_scale]
    )

    encoded = artifact["encoder"].transform(processed[categorical_features])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    encoder = artifact["encoder"]
    if hasattr(encoder, "get_feature_names_out"):
        encoded_columns = encoder.get_feature_names_out(categorical_features)
    else:
        encoded_columns = [
            f"encoded_{index}" for index in range(encoded.shape[1])
        ]

    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=processed.index)
    processed = processed.drop(columns=categorical_features)
    processed = pd.concat([processed, encoded_df], axis=1)

    model = artifact["model"]
    if hasattr(model, "feature_names_in_"):
        model_columns = list(model.feature_names_in_)
        missing_model_columns = [col for col in model_columns if col not in processed]
        if missing_model_columns:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Processed data does not match model feature names.",
                    "missing_model_columns": missing_model_columns,
                },
            )
        processed = processed[model_columns]

    return processed


def _label_for(value: Any, target_labels: Any) -> str:
    value = _jsonable(value)

    if isinstance(target_labels, dict):
        if value in target_labels:
            return str(target_labels[value])
        if str(value) in target_labels:
            return str(target_labels[str(value)])
        for label, encoded_value in target_labels.items():
            if _jsonable(encoded_value) == value:
                return str(label)
        return str(value)

    if isinstance(target_labels, (list, tuple)):
        if isinstance(value, int) and 0 <= value < len(target_labels):
            return str(target_labels[value])
        return str(value)

    return str(value)


def _probabilities(model: Any, proba: Any, target_labels: Any) -> dict[str, float]:
    probabilities = proba[0]
    classes = getattr(model, "classes_", range(len(probabilities)))
    return {
        _label_for(class_value, target_labels): float(probability)
        for class_value, probability in zip(classes, probabilities)
    }


@app.post("/predict")
def predict(input_data: PredictionInput, request: Request) -> dict[str, Any]:
    artifact = request.app.state.artifact
    if hasattr(input_data, "model_dump"):
        payload = input_data.model_dump()
    else:
        payload = input_data.dict()

    try:
        model_input = _transform_payload(payload, artifact)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model = artifact["model"]
    prediction = model.predict(model_input)[0]
    probabilities = model.predict_proba(model_input)

    prediction_value = _jsonable(prediction)
    return {
        "success": True,
        "prediction": prediction_value,
        "label": _label_for(prediction_value, artifact["target_labels"]),
        "probabilities": _probabilities(
            model, probabilities, artifact["target_labels"]
        ),
    }
