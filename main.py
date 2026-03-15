import json
import os
import sys
from tkinter import Tk, filedialog, messagebox

from src.pipelines.run_end_to_end import run_end_to_end, run_end_to_end_many


def main() -> int:
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
