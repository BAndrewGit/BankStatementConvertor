import json
import sys
from tkinter import Tk, filedialog, messagebox

from pdf_pipeline.project_pipeline import run_full_pipeline


def run_gui_pipeline() -> int:
    root = Tk()
    root.withdraw()

    pdf_path = filedialog.askopenfilename(
        title="Selecteaza extrasul PDF",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    if not pdf_path:
        print("Procesare anulata: nu ai selectat niciun PDF.")
        return 1

    export_dir = filedialog.askdirectory(
        title="Selecteaza folderul unde exporti fisierele CSV",
        mustexist=True,
    )
    if not export_dir:
        print("Procesare anulata: nu ai selectat folderul de export.")
        return 1

    try:
        result = run_full_pipeline(pdf_path=pdf_path, export_dir=export_dir)
    except Exception as exc:
        messagebox.showerror("Pipeline error", f"Procesarea a esuat:\n{exc}")
        print(f"Procesare esuata: {exc}", file=sys.stderr)
        return 2

    summary = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    print(summary)
    messagebox.showinfo(
        "Procesare finalizata",
        "Pipeline complet rulat cu succes.\n"
        f"Output folder: {export_dir}\n"
        f"Raport: {result.output_files['summary_json']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_gui_pipeline())
