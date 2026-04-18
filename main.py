import json
import os
import subprocess
import sys
from tkinter import Tk, filedialog, messagebox

from dotenv import load_dotenv

from src.pipelines.run_end_to_end import run_end_to_end, run_end_to_end_many
from src.ui import launch_desktop_app


load_dotenv()

_RELAUNCH_GUARD_ENV = "APP_SKIP_VENV_RELAUNCH"


def _desktop_mode_enabled() -> bool:
    return os.getenv("APP_UI", "").strip().lower() in {"pyside6", "qt", "desktop"}


def _maybe_relaunch_with_project_venv_for_desktop() -> int | None:
    if not _desktop_mode_enabled():
        return None
    if os.getenv(_RELAUNCH_GUARD_ENV, "0") == "1":
        return None
    if sys.version_info >= (3, 11):
        return None

    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, ".venv313", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        return None

    if os.path.abspath(sys.executable) == os.path.abspath(venv_python):
        return None

    child_env = dict(os.environ)
    child_env[_RELAUNCH_GUARD_ENV] = "1"
    process = subprocess.run([venv_python, os.path.abspath(__file__)], env=child_env)
    return int(process.returncode)


def main() -> int:
    relaunch_code = _maybe_relaunch_with_project_venv_for_desktop()
    if relaunch_code is not None:
        return relaunch_code

    if _desktop_mode_enabled():
        try:
            return int(launch_desktop_app())
        except Exception as exc:
            print(f"Desktop UI failed to start: {exc}", file=sys.stderr)
            return 2

    root = Tk()
    root.withdraw()

    try:
        pdf_paths = filedialog.askopenfilenames(
            title="Selecteaza unul sau mai multe extrase PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not pdf_paths:
            print("Procesare anulata: nu ai selectat niciun PDF.")
            return 1

        output_dir = filedialog.askdirectory(
            title="Selecteaza folderul unde salvezi rezultatul",
            mustexist=True,
        )
        if not output_dir:
            print("Procesare anulata: nu ai selectat folderul de export.")
            return 1

        if len(pdf_paths) == 1:
            result = run_end_to_end(pdf_path=pdf_paths[0], export_dir=output_dir)
        else:
            result = run_end_to_end_many(pdf_paths=list(pdf_paths), export_dir=output_dir)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        show_success_dialog = os.getenv("APP_SHOW_SUCCESS_DIALOG", "0") == "1"
        if show_success_dialog:
            monthly_path = getattr(result, "final_dataset_monthly_csv_path", None)
            monthly_line = f"\nDataset lunar: {monthly_path}" if monthly_path else ""
            messagebox.showinfo(
                "Procesare finalizata",
                "Fluxul complet a fost rulat cu succes.\n"
                f"Output folder: {output_dir}\n"
                f"Dataset final: {result.final_dataset_csv_path}\n"
                f"Raport rulare: {result.run_report_path}"
                f"{monthly_line}",
            )
        return 0
    except Exception as exc:
        messagebox.showerror("Pipeline error", f"Procesarea a esuat:\n{exc}")
        print(f"Procesare esuata: {exc}", file=sys.stderr)
        return 2
    finally:
        root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
