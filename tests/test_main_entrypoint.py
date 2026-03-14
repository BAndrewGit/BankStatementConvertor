import unittest
from unittest.mock import Mock, patch

import main


class MainEntrypointTests(unittest.TestCase):
    def test_main_runs_single_flow_successfully(self):
        fake_result = Mock()
        fake_result.run_report_path = r"C:\out\run_report.json"
        fake_result.to_dict.return_value = {"ok": True}

        with patch("main.Tk") as mocked_tk:
            root = Mock()
            mocked_tk.return_value = root

            with patch("main.filedialog.askopenfilename", return_value=r"C:\in\statement.pdf"):
                with patch("main.filedialog.askdirectory", return_value=r"C:\out"):
                    with patch("main.run_end_to_end", return_value=fake_result) as mocked_run:
                        with patch("main.messagebox.showinfo") as mocked_info:
                            exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_run.assert_called_once_with(pdf_path=r"C:\in\statement.pdf", export_dir=r"C:\out")
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

                with patch("main.filedialog.askopenfilename", return_value=r"C:\in\statement.pdf"):
                    with patch("main.filedialog.askdirectory", return_value=r"C:\out"):
                        with patch("main.run_end_to_end", return_value=fake_result):
                            with patch("main.messagebox.showinfo") as mocked_info:
                                exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mocked_info.assert_called_once()

    def test_main_returns_one_when_pdf_not_selected(self):
        with patch("main.Tk") as mocked_tk:
            root = Mock()
            mocked_tk.return_value = root

            with patch("main.filedialog.askopenfilename", return_value=""):
                exit_code = main.main()

        self.assertEqual(exit_code, 1)
        root.destroy.assert_called_once()

    def test_main_returns_one_when_output_folder_not_selected(self):
        with patch("main.Tk") as mocked_tk:
            root = Mock()
            mocked_tk.return_value = root

            with patch("main.filedialog.askopenfilename", return_value=r"C:\in\statement.pdf"):
                with patch("main.filedialog.askdirectory", return_value=""):
                    with patch("main.run_end_to_end") as mocked_run:
                        with patch("main.messagebox.showerror") as mocked_error:
                            exit_code = main.main()

        self.assertEqual(exit_code, 1)
        mocked_run.assert_not_called()
        mocked_error.assert_not_called()
        root.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()


