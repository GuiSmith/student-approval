import unittest
from unittest.mock import patch
from types import SimpleNamespace
from threading import Lock

import numpy as np
import pandas as pd

import main


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, model_input):
        return np.array([0])

    def predict_proba(self, model_input):
        return np.array([[0.8, 0.2]])


class FakeExplainer:
    def shap_values(self, model_input, nsamples, silent):
        self.call = {
            "columns": list(model_input.columns),
            "nsamples": nsamples,
            "silent": silent,
        }
        return np.array(
            [
                [
                    [0.4, -0.4],
                    [0.2, -0.2],
                    [-0.1, 0.1],
                ]
            ]
        )


class ShapExplanationTests(unittest.TestCase):
    def setUp(self):
        self.model_input = pd.DataFrame(
            [[0.5, 1.0, 1.0]],
            columns=["failures", "absences", "school_MS"],
        )
        self.artifact = {
            "model": FakeModel(),
            "categorical_features": ["school"],
            "target_labels": {0: "Reprovado", 1: "Aprovado"},
        }
        self.payload = {
            "failures": 2,
            "absences": 10,
            "school": "MS",
        }

    def test_builds_sorted_human_readable_factors(self):
        explainer = FakeExplainer()

        explanation = main._build_explanation(
            self.payload,
            self.model_input,
            0,
            self.artifact,
            explainer,
        )

        self.assertEqual(explanation["method"], "shap")
        self.assertEqual(explanation["prediction_label"], "Reprovado")
        self.assertEqual(
            explanation["top_factors"],
            [
                {
                    "feature": "failures",
                    "value": 2,
                    "direction": "increase",
                    "impact": 0.4,
                },
                {
                    "feature": "absences",
                    "value": 10,
                    "direction": "increase",
                    "impact": 0.2,
                },
                {
                    "feature": "school",
                    "value": "MS",
                    "direction": "decrease",
                    "impact": -0.1,
                },
            ],
        )
        self.assertEqual(explainer.call["columns"], list(self.model_input.columns))
        self.assertEqual(explainer.call["nsamples"], main.SHAP_NSAMPLES)

    @patch("main._build_explanation", side_effect=RuntimeError("SHAP failed"))
    def test_fallback_returns_none_without_raising(self, _build_explanation):
        with self.assertLogs("main", level="ERROR") as logs:
            explanation = main._safe_build_explanation(
                self.payload,
                self.model_input,
                0,
                self.artifact,
                FakeExplainer(),
                Lock(),
            )

        self.assertIsNone(explanation)
        self.assertTrue(
            any(
                "Failed to calculate SHAP explanation" in message
                for message in logs.output
            )
        )

    @patch("main._build_explanation", side_effect=RuntimeError("SHAP failed"))
    @patch("main._transform_payload")
    def test_prediction_response_survives_shap_failure(
        self,
        transform_payload,
        _build_explanation,
    ):
        transform_payload.return_value = self.model_input
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    artifact=self.artifact,
                    shap_explainer=FakeExplainer(),
                    shap_lock=Lock(),
                )
            )
        )

        with self.assertLogs("main", level="ERROR"):
            response = main.predict(
                main.PredictionInput(**self.payload),
                request,
            )

        self.assertTrue(response["success"])
        self.assertEqual(response["prediction"], 0)
        self.assertEqual(response["label"], "Reprovado")
        self.assertEqual(
            response["probabilities"],
            {"Reprovado": 0.8, "Aprovado": 0.2},
        )
        self.assertIsNone(response["explanation"])


if __name__ == "__main__":
    unittest.main()
