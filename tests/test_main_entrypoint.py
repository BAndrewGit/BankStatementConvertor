import importlib
import os
import unittest
from unittest.mock import Mock, patch

import main


class MainEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patcher = patch.dict("os.environ", {"APP_UI": ""}, clear=False)
        self._env_patcher.start()

    def tearDown(self) -> None:
        self._env_patcher.stop()

    def test_main_runs_single_flow_successfully(self):
        fake_result = Mock()
        fake_result.run_report_path = r"C:\out\run_report.json"
        fake_result.to_dict.return_value = {"ok": True}

        with patch("main.Tk") as mocked_tk:
            root = Mock()
            mocked_tk.return_value = root

            with patch("main.filedialog.askopenfilenames", return_value=(r"C:\in\statement.pdf",)):
                with patch("main.filedialog.askdirectory", return_value=r"C:\out"):
                    with patch("main.run_end_to_end", return_value=fake_result) as mocked_run:
                        with patch("main.run_end_to_end_many") as mocked_run_many:
                            with patch("main.messagebox.showinfo") as mocked_info:
                                exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_run.assert_called_once_with(pdf_path=r"C:\in\statement.pdf", export_dir=r"C:\out")
        mocked_run_many.assert_not_called()
        mocked_info.assert_not_called()
        root.withdraw.assert_called_once()
        root.destroy.assert_called_once()

    def test_main_runs_batch_flow_when_multiple_pdfs_selected(self):
        fake_result = Mock()
        fake_result.run_report_path = r"C:\out\run_report.json"
        fake_result.to_dict.return_value = {"ok": True}

        with patch("main.Tk") as mocked_tk:
            root = Mock()
            mocked_tk.return_value = root

            with patch(
                "main.filedialog.askopenfilenames",
                return_value=(r"C:\in\statement_1.pdf", r"C:\in\statement_2.pdf"),
            ):
                with patch("main.filedialog.askdirectory", return_value=r"C:\out"):
                    with patch("main.run_end_to_end") as mocked_run:
                        with patch("main.run_end_to_end_many", return_value=fake_result) as mocked_run_many:
                            with patch("main.messagebox.showinfo") as mocked_info:
                                exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_run.assert_not_called()
        mocked_run_many.assert_called_once_with(
            pdf_paths=[r"C:\in\statement_1.pdf", r"C:\in\statement_2.pdf"],
            export_dir=r"C:\out",
        )
        mocked_info.assert_not_called()
        root.withdraw.assert_called_once()
        root.destroy.assert_called_once()

    def test_main_can_show_success_popup_when_enabled(self):
        fake_result = Mock()
        fake_result.run_report_path = r"C:\out\run_report.json"
        fake_result.final_dataset_csv_path = r"C:\out\final_dataset.csv"
        fake_result.to_dict.return_value = {"ok": True}

        with patch.dict("os.environ", {"APP_SHOW_SUCCESS_DIALOG": "1"}, clear=False):
            with patch("main.Tk") as mocked_tk:
                root = Mock()
                mocked_tk.return_value = root

                with patch("main.filedialog.askopenfilenames", return_value=(r"C:\in\statement.pdf",)):
                    with patch("main.filedialog.askdirectory", return_value=r"C:\out"):
                        with patch("main.run_end_to_end", return_value=fake_result):
                            with patch("main.run_end_to_end_many") as mocked_run_many:
                                with patch("main.messagebox.showinfo") as mocked_info:
                                    exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_run_many.assert_not_called()
        mocked_info.assert_called_once()

    def test_main_returns_one_when_pdf_not_selected(self):
        with patch("main.Tk") as mocked_tk:
            root = Mock()
            mocked_tk.return_value = root

            with patch("main.filedialog.askopenfilenames", return_value=()):
                exit_code = main.main()

        self.assertEqual(exit_code, 1)
        root.destroy.assert_called_once()

    def test_main_returns_one_when_output_folder_not_selected(self):
        with patch("main.Tk") as mocked_tk:
            root = Mock()
            mocked_tk.return_value = root

            with patch("main.filedialog.askopenfilenames", return_value=(r"C:\in\statement.pdf",)):
                with patch("main.filedialog.askdirectory", return_value=""):
                    with patch("main.run_end_to_end") as mocked_run:
                        with patch("main.messagebox.showerror") as mocked_error:
                            exit_code = main.main()

        self.assertEqual(exit_code, 1)
        mocked_run.assert_not_called()
        mocked_error.assert_not_called()
        root.destroy.assert_called_once()

    def test_main_uses_desktop_ui_when_app_ui_env_is_desktop(self):
        with patch.dict("os.environ", {"APP_UI": "desktop"}, clear=False):
            with patch("main.launch_desktop_app", return_value=0) as mocked_launch:
                with patch("main.Tk") as mocked_tk:
                    exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_launch.assert_called_once_with()
        mocked_tk.assert_not_called()

    def test_desktop_relaunches_from_legacy_python_when_project_venv_exists(self):
        expected_python = os.path.join(
            os.path.dirname(os.path.abspath(main.__file__)),
            ".venv313",
            "Scripts",
            "python.exe",
        )
        with patch.dict("os.environ", {"APP_UI": "desktop"}, clear=False):
            with patch("main.sys.version_info", (3, 10, 0)):
                with patch("main.os.path.exists", return_value=True):
                    with patch("main.sys.executable", r"C:\Python310\python.exe"):
                        process = Mock(returncode=0)
                        with patch("main.subprocess.run", return_value=process) as mocked_run:
                            exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_run.assert_called_once()
        args, kwargs = mocked_run.call_args
        self.assertEqual(args[0], [expected_python, os.path.abspath(main.__file__)])
        self.assertEqual(kwargs["env"]["APP_SKIP_VENV_RELAUNCH"], "1")

    def test_desktop_relaunch_is_skipped_when_guard_env_is_set(self):
        with patch.dict(
            "os.environ",
            {"APP_UI": "desktop", "APP_SKIP_VENV_RELAUNCH": "1"},
            clear=False,
        ):
            with patch("main.launch_desktop_app", return_value=0) as mocked_launch:
                with patch("main.subprocess.run") as mocked_run:
                    exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_launch.assert_called_once_with()
        mocked_run.assert_not_called()


class MainModuleLoadTests(unittest.TestCase):
    def test_module_load_calls_load_dotenv(self):
        with patch("dotenv.load_dotenv") as mocked_load:
            importlib.reload(main)
        self.assertTrue(mocked_load.called)


if __name__ == "__main__":
    unittest.main()


