from pathlib import Path

import pandas as pd

from thermoagent.v8_figures import architecture_figure, configure_style


def test_v8_architecture_figure_has_vector_pdf_png_and_source_data(tmp_path: Path):
    configure_style()
    architecture_figure(tmp_path)
    pdf = tmp_path / "figures" / "pdf" / "v8_belief_monitoring_architecture.pdf"
    png = tmp_path / "figures" / "png" / "v8_belief_monitoring_architecture.png"
    source = tmp_path / "figures" / "source_data" / "v8_belief_monitoring_architecture.csv"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG")
    frame = pd.read_csv(source)
    assert (frame.record_type == "node").sum() == 8
    assert (frame.record_type == "edge").sum() == 9
