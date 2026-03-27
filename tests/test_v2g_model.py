"""Unit tests for the V2G optimization module."""

from __future__ import annotations

from pathlib import Path
import unittest

from ortools.linear_solver import pywraplp

from src import v2g_model


class TestV2GModel(unittest.TestCase):
    """Core behavior tests for the optimization pipeline."""

    def test_generate_default_data_shapes(self) -> None:
        """Default dataset should expose expected dimensions and keys."""
        data = v2g_model.generate_default_data(num_hours=12, num_evs=2, seed=123)

        self.assertEqual(len(data["T"]), 12)
        self.assertEqual(len(data["EVs"]), 2)
        self.assertEqual(data["M"], ["DA", "ID"])
        self.assertIn("buy_price_tm", data)
        self.assertIn("sell_price_tm", data)

    def test_pipeline_returns_expected_fields(self) -> None:
        """Pipeline should solve and return standard KPI fields."""
        results = v2g_model.run_pipeline()

        self.assertIn(results["status"], {"OPTIMAL", "FEASIBLE"})
        self.assertIn("objective_value", results)
        self.assertIn("net_profit", results)
        self.assertIn("final_energy_by_ev", results)
        self.assertIn("exported_file", results)

    def test_export_file_is_created(self) -> None:
        """Pipeline run should generate a JSON artifact in results/."""
        results = v2g_model.run_pipeline()
        export_path = Path(results["exported_file"])

        self.assertTrue(export_path.exists())
        self.assertEqual(export_path.suffix, ".json")

        # Clean up test artifact to keep the repository tidy.
        export_path.unlink(missing_ok=True)

    def test_status_to_text_unknown(self) -> None:
        """Unknown status codes should map to UNKNOWN."""
        self.assertEqual(v2g_model.status_to_text(999999), "UNKNOWN")

    def test_extract_results_handles_infeasible_status(self) -> None:
        """Result extraction should return message for infeasible status."""
        data = v2g_model.generate_default_data(num_hours=6, num_evs=1, seed=1)
        model = v2g_model.create_model(data)

        results = v2g_model.extract_results(model, pywraplp.Solver.INFEASIBLE)

        self.assertEqual(results["status"], "INFEASIBLE")
        self.assertIn("message", results)


if __name__ == "__main__":
    unittest.main()
