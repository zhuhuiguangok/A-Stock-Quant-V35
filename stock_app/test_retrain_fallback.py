"""模型重训线程兜底的回归测试。

背景：v2 重训时训练线程被 BaseException（非 Exception 子类）静默杀死，
_TASK 状态卡在 running 两小时。修复后 start_retrain 用 except BaseException
+ finally 保险。本测试验证：
1. SystemExit（BaseException 子类，旧代码接不住）现在会被捕获并置 failed；
2. 任何逃逸路径下 finally 保险都会把 running 强制置 failed；
3. 正常完成路径不受影响。
"""
import time
from unittest import mock

from django.test import SimpleTestCase

from stock_app import model_retrain_service as mrs


class RetrainFallbackTest(SimpleTestCase):
    def tearDown(self):
        mrs._TASK.update(state="idle", message="测试复位", error=None,
                         started_at=None, finished_at=None)

    def _wait_state(self, seconds=6):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if mrs.get_task().get("state") != "running":
                return mrs.get_task()
            time.sleep(0.05)
        return mrs.get_task()

    def test_system_exit_is_caught_and_marks_failed(self):
        """BaseException（SystemExit）逃逸 → 状态必须变 failed 而非卡 running。"""
        with mock.patch.object(mrs, "_run_retrain", side_effect=SystemExit(1)), \
             mock.patch.object(mrs, "_tlog"):
            mrs._TASK.update(state="idle")
            mrs.start_retrain(trigger="test")
            task = self._wait_state()
        self.assertEqual(task["state"], "failed")
        self.assertIn("SystemExit", task["message"])

    def test_finally_insurance_forces_failed(self):
        """模拟 except 分支自己也出错 → finally 保险强制把 running 置 failed。"""
        real_set_task = mrs._set_task

        def broken_set_task(**kwargs):
            # 模拟 except 里置 failed 时自身再抛异常
            if kwargs.get("state") == "failed":
                raise RuntimeError("set_task itself exploded")
            real_set_task(**kwargs)

        with mock.patch.object(mrs, "_run_retrain", side_effect=SystemExit(1)), \
             mock.patch.object(mrs, "_set_task", side_effect=broken_set_task), \
             mock.patch.object(mrs, "_tlog"):
            mrs._TASK.update(state="idle")
            mrs.start_retrain(trigger="test")
            task = self._wait_state()
        self.assertEqual(task["state"], "failed")
        self.assertIn("训练线程异常终止", task["message"])

    def test_success_path_unaffected(self):
        """正常完成路径：state=completed，兜底不误伤。"""
        fake_result = {"rows": 123, "sample_note": "全量 123 行",
                       "span": "2022~2026", "trained": True}
        with mock.patch.object(mrs, "_run_retrain", return_value=fake_result), \
             mock.patch.object(mrs, "_tlog"):
            mrs._TASK.update(state="idle")
            mrs.start_retrain(trigger="test")
            task = self._wait_state()
        self.assertEqual(task["state"], "completed")
        self.assertIn("trained=True", task["message"])
