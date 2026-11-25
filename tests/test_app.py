"""Behavioral UI checks require no browser or private student data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest

def test_segmentation_source_change_invalidates_previous_prediction():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
    label=next(v for v in app.sidebar.radio[0].options if v.startswith('07'))
    app.sidebar.radio[0].set_value(label).run(timeout=60)
    next(b for b in app.button if b.label=='Compare segmentation methods').click().run(timeout=60)
    assert 'seg_result' in app.session_state
    next(s for s in app.selectbox if s.label=='Sample image').set_value('Camera').run(timeout=60)
    assert 'seg_result' not in app.session_state
    assert not app.exception and not app.error
