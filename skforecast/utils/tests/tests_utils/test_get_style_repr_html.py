# Unit test get_style_repr_html
# ==============================================================================
import pytest
from ...utils import get_style_repr_html


@pytest.mark.parametrize("is_fitted", 
                         [True, False], 
                         ids=lambda is_fitted: f'is_fitted: {is_fitted}')
def test_get_style_repr_html(is_fitted):
    """
    Check if the style is correct.
    """

    style, unique_id = get_style_repr_html(is_fitted=is_fitted)
    
    background_color = "#f0f8ff" if is_fitted else "#f9f1e2"
    section_color = "#b3dbfd" if is_fitted else "#fae3b3"

    expected_style = f"""
    <style>
        .container-{unique_id} {{
            font-family: 'Arial', sans-serif;
            font-size: 0.9em;
            color: #333333;
            border: 1px solid #ddd;
            background-color: {background_color};
            padding: 5px 15px;
            border-radius: 8px;
            max-width: 600px;
            #margin: auto;
        }}
        .container-{unique_id} h2 {{
            font-size: 1.5em;
            color: #222222;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-bottom: 15px;
            margin-top: 5px;
        }}
        .container-{unique_id} details {{
            margin: 10px 0;
        }}
        .container-{unique_id} summary {{
            font-weight: bold;
            font-size: 1.1em;
            color: #000000;
            cursor: pointer;
            margin-bottom: 5px;
            background-color: {section_color};
            padding: 5px;
            border-radius: 5px;
        }}
        .container-{unique_id} summary:hover {{
            color: #000000;
            background-color: #e0e0e0;
        }}
        .container-{unique_id} ul {{
            font-family: 'Courier New', monospace;
            list-style-type: none;
            padding-left: 20px;
            margin: 10px 0;
            line-height: normal;
        }}
        .container-{unique_id} li {{
            margin: 5px 0;
            font-family: 'Courier New', monospace;
        }}
        .container-{unique_id} li strong {{
            font-weight: bold;
            color: #444444;
        }}
        .container-{unique_id} li::before {{
            content: "- ";
            color: #666666;
        }}
        .container-{unique_id} a {{
            color: #001633;
            text-decoration: none;
        }}
        .container-{unique_id} a:hover {{
            color: #359ccb; 
        }}
    </style>
    """

    assert isinstance(unique_id, str)
    assert style == expected_style
