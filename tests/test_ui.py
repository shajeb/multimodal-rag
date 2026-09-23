from pathlib import Path
from streamlit.testing.v1 import AppTest
from rag.config import Settings
from rag.service import RAGService
from scripts.create_sample import create_sample


def test_streamlit_upload_workspace_and_chat(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "index")
    monkeypatch.setattr(Settings, "load", lambda: settings)
    service = RAGService(settings)
    service.ingest(create_sample(tmp_path / "solar.pdf"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Multimodal document workspace"
    app.chat_input[0].set_value("How many solar panels?").run(timeout=30)
    assert not app.exception
    assert any("24" in item.value for item in app.markdown)
