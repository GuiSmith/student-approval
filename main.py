from contextlib import asynccontextmanager
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel


ARTIFACT_PATHS = (
    Path("model/student_approval_model.pkl"),
    Path("modelo/student_approval_model.pkl"),
)
SHAP_TOP_FACTORS = 5
SHAP_NSAMPLES = 100
LOGGER = logging.getLogger(__name__)
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
PREDICTION_BODY_FIELDS = (
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "subject",
)
PREDICTION_BODY_EXAMPLE = {
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
PREDICTION_BODY_SCHEMA = {
    "type": "object",
    "required": list(PREDICTION_BODY_FIELDS),
    "additionalProperties": False,
    "properties": {
        **{
            field: {
                "type": "string",
                "enum": list(choices),
                "description": f"Accepted values: {', '.join(choices)}.",
            }
            for field, choices in CATEGORICAL_CHOICES.items()
        },
        **{
            field: {
                "type": "integer",
                "minimum": minimum,
                "maximum": maximum,
                "description": (
                    f"Integer value between {minimum} and {maximum}."
                ),
            }
            for field, (minimum, maximum) in INT_RANGES.items()
        },
    },
    "example": PREDICTION_BODY_EXAMPLE,
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


def _build_shap_explainer(artifact: dict[str, Any]) -> Any:
    model = artifact["model"]
    model_columns = list(model.feature_names_in_)
    background = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns,
    )
    return shap.KernelExplainer(model.predict_proba, background)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = _load_artifact()
    app.state.shap_lock = Lock()
    try:
        app.state.shap_explainer = _build_shap_explainer(app.state.artifact)
    except Exception:
        LOGGER.exception("Failed to initialize SHAP explainer")
        app.state.shap_explainer = None
    yield


app = FastAPI(title="Student Approval Prediction API", lifespan=lifespan)


def _custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    openapi_schema["paths"]["/predict"]["post"]["requestBody"] = {
        "content": {
            "application/json": {
                "schema": PREDICTION_BODY_SCHEMA,
                "example": PREDICTION_BODY_EXAMPLE,
            }
        },
        "required": True,
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = _custom_openapi


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


def _class_index(model: Any, prediction: Any) -> int:
    classes = list(getattr(model, "classes_", []))
    prediction = _jsonable(prediction)
    return classes.index(prediction)


def _shap_values_for_class(
    shap_values: Any,
    class_index: int,
    feature_count: int,
) -> np.ndarray:
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[class_index])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[0, :, class_index]
        elif values.ndim == 2 and values.shape == (feature_count, 2):
            values = values[:, class_index]

    values = np.asarray(values).reshape(-1)
    if len(values) != feature_count:
        raise ValueError(
            "SHAP output does not match the processed feature count."
        )
    return values


def _original_feature_name(
    processed_feature: str,
    categorical_features: list[str],
) -> str:
    for feature in categorical_features:
        if processed_feature.startswith(f"{feature}_"):
            return feature
    return processed_feature


def _build_explanation(
    payload: dict[str, Any],
    model_input: pd.DataFrame,
    prediction: Any,
    artifact: dict[str, Any],
    explainer: Any,
) -> dict[str, Any]:
    if explainer is None:
        raise RuntimeError("SHAP explainer is unavailable.")

    model = artifact["model"]
    class_index = _class_index(model, prediction)
    raw_shap_values = explainer.shap_values(
        model_input,
        nsamples=SHAP_NSAMPLES,
        silent=True,
    )
    shap_values = _shap_values_for_class(
        raw_shap_values,
        class_index,
        model_input.shape[1],
    )

    impacts: dict[str, float] = {}
    categorical_features = list(artifact["categorical_features"])
    for processed_feature, shap_value in zip(model_input.columns, shap_values):
        feature = _original_feature_name(
            str(processed_feature),
            categorical_features,
        )
        impacts[feature] = impacts.get(feature, 0.0) + float(shap_value)

    top_factors = sorted(
        impacts.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:SHAP_TOP_FACTORS]

    prediction_value = _jsonable(prediction)
    return {
        "method": "shap",
        "prediction_label": _label_for(
            prediction_value,
            artifact["target_labels"],
        ),
        "top_factors": [
            {
                "feature": feature,
                "value": _jsonable(payload[feature]),
                "direction": "increase" if impact >= 0 else "decrease",
                "impact": round(float(impact), 4),
            }
            for feature, impact in top_factors
        ],
    }


def _safe_build_explanation(
    payload: dict[str, Any],
    model_input: pd.DataFrame,
    prediction: Any,
    artifact: dict[str, Any],
    explainer: Any,
    shap_lock: Any,
) -> dict[str, Any] | None:
    try:
        with shap_lock:
            return _build_explanation(
                payload,
                model_input,
                prediction,
                artifact,
                explainer,
            )
    except Exception:
        LOGGER.exception("Failed to calculate SHAP explanation")
        return None


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
    explanation = _safe_build_explanation(
        payload,
        model_input,
        prediction_value,
        artifact,
        getattr(request.app.state, "shap_explainer", None),
        getattr(request.app.state, "shap_lock", Lock()),
    )
    return {
        "success": True,
        "prediction": prediction_value,
        "label": _label_for(prediction_value, artifact["target_labels"]),
        "probabilities": _probabilities(
            model, probabilities, artifact["target_labels"]
        ),
        "explanation": explanation,
    }
