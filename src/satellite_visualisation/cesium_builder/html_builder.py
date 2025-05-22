import os
from src import logger

log = logger.get_logger(__name__)


def write_html_file(
    viz_string_content, output_dir, html_file_name_base, top_html_file, bottom_html_file
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "visualisation_output")
    os.makedirs(output_dir, exist_ok=True)
    output_html_file = os.path.join(output_dir, f"{html_file_name_base.replace(' ', '_')}.html")
    log.info(f"Attempting to write HTML file to: {output_html_file}")
    try:
        with open(output_html_file, "w", encoding="utf-8") as writer_html:
            if os.path.exists(top_html_file):
                with open(top_html_file, "r", encoding="utf-8") as fi:
                    writer_html.write(fi.read())
            writer_html.write(viz_string_content)
            if os.path.exists(bottom_html_file):
                with open(bottom_html_file, "r", encoding="utf-8") as fb:
                    writer_html.write(fb.read())
        log.info(f"Successfully wrote visualization to: {output_html_file}")
        print(f"ACTION: HTML file generated at: {output_html_file}")
        print(
            "Please open this file via a local web server (e.g., 'python -m http.server' in project root)."
        )
    except Exception as e:
        log.error(f"Error writing HTML file {output_html_file}: {e}")
        print(f"ERROR: {e}")
