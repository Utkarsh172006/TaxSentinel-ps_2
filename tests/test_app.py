from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_metrics_page_is_available_without_loading_a_dataset():
    app = AppTest.from_file("app/Home.py", default_timeout=30).run()
    app.radio[0].set_value("Model & metrics").run()

    assert not app.exception
    assert [element.value for element in app.title] == ["Model & metrics"]
    assert len(app.slider) == 1
    assert not app.info
