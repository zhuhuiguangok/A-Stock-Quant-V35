import json
import time
from unittest.mock import patch

from django.http import JsonResponse
from django.test import SimpleTestCase

from stock_app import views


class AnalysisProgressApiTests(SimpleTestCase):
    def setUp(self):
        views._analysis_tasks.clear()

    @patch("stock_app.views.dual_verify_stocks")
    def test_background_task_completes_with_selection_result(self, select_view):
        select_view.return_value = JsonResponse(
            {"status": "success", "trend_stocks": [], "bottom_stocks": []}
        )

        response = self.client.post(
            "/api/select/start/",
            data={"train": False},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 202)
        task_id = response.json()["task_id"]

        result = None
        for _ in range(50):
            result = self.client.get(
                "/api/select/progress/",
                {"task_id": task_id},
            ).json()
            if result["state"] == "completed":
                break
            time.sleep(0.02)

        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["percent"], 100)
        self.assertEqual(result["result"]["status"], "success")

    def test_unknown_task_returns_404(self):
        response = self.client.get(
            "/api/select/progress/",
            {"task_id": "missing"},
        )

        self.assertEqual(response.status_code, 404)


class AnalysisProgressTemplateTests(SimpleTestCase):
    def test_homepage_uses_real_progress_api(self):
        content = self.client.get("/").content.decode("utf-8")

        self.assertIn('id="analysisProgressBar"', content)
        self.assertIn('id="analysisProgressDetail"', content)
        self.assertIn("/select/start/", content)
        self.assertIn("/select/progress/", content)
        self.assertNotIn("loaderSteps = [", content)
