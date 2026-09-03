import re
import sys
from pathlib import Path

import pytest


MODEL_EFFECT = Path(__file__).resolve().parents[2]
WEB_DIR = MODEL_EFFECT / "visualization" / "viewer" / "web"
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))


def test_viewer_defaults_to_english_and_loads_i18n_first():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert '<html lang="en" class="i18n-pending">' in html
    assert 'id="languageSelect"' in html
    assert '<option value="en">English</option>' in html
    assert '<option value="zh">Chinese</option>' in html
    assert html.index('src="i18n.js') < html.index('src="camera_baselines.js')
    assert html.index('src="i18n.js') < html.index('src="app.js')


def test_viewer_i18n_covers_dynamic_canvas_and_dialog_text():
    source = (WEB_DIR / "i18n.js").read_text(encoding="utf-8")

    assert "const STORAGE_KEY = 'wuji-viewer-language'" in source
    assert "new MutationObserver" in source
    assert "['fillText', 'strokeText', 'measureText']" in source
    assert "['alert', 'confirm', 'prompt']" in source
    assert "CJK_RUN_RE" in source
    assert "window.ViewerI18n" in source
    assert re.search(r"language\s*=\s*readInitialLanguage\(\)", source)


def test_viewer_exposes_tunable_ukf_parameters_and_guidance():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    for field in ("ukfQ", "ukfR", "ukfBeta"):
        assert f'id="{field}"' in html
    assert "q ↑ 更跟手，r/beta ↑ 更平滑" in html
    assert "function effectiveHandMode()" in source
    assert "smooth@${fmt(state.ukf.q)}" in source


def test_viewer_serves_i18n_script():
    pytest.importorskip("flask")
    from visualization.viewer.routes import create_app

    app = create_app(object())
    response = app.test_client().get("/i18n.js")

    assert response.status_code == 200
    assert response.mimetype == "application/javascript"
    assert b"window.ViewerI18n" in response.data


def test_benchmark_supports_multi_table_jpg_export():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    export_keys = re.findall(r'data-bench-table-export="([^"]+)"', html)
    assert export_keys == [
        "hot3d-camera",
        "arctic-camera",
        "vidihand-arctic",
        "vidihand-hot3d",
    ]
    assert 'id="bfTableExportSeparate"' in html
    assert 'id="bfTableExportCombined"' in html
    assert "function benchRenderTableExport" in source
    assert "async function exportBenchTablesAsJpg(combined)" in source
    assert "'image/jpeg',0.95" in source
