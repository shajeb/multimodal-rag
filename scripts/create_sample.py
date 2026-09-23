"""Generate a small, synthetic multimodal PDF without external data."""
from pathlib import Path
import pymupdf as fitz


def create_sample(path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((45, 45), "Solar installation report", fontsize=20)
    page.insert_text((45, 75), "The pilot installation contains 24 solar panels. Each panel is rated at 400 watts.")
    page.insert_text((45, 95), "Total rated capacity is 9.6 kilowatts. The reported annual output is 12,000 kWh.")
    xs, ys = [45, 210, 375], [130, 160, 190, 220]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for row, cells in enumerate([["Metric", "Value"], ["Panels", "24"], ["Capacity (kW)", "9.6"]]):
        for col, value in enumerate(cells):
            page.insert_text((xs[col]+8, ys[row]+20), value)
    chart = fitz.open()
    cp = chart.new_page(width=330, height=180)
    cp.insert_text((10, 20), "Quarterly energy output (kWh)")
    for i, value in enumerate([2200, 3600, 3800, 2400]):
        x = 30 + i*70
        cp.draw_rect(fitz.Rect(x, 145-value/40, x+35, 145), fill=(0.15, 0.45, 0.7))
        cp.insert_text((x, 165), f"Q{i+1}")
        cp.insert_text((x, 135-value/40), str(value), fontsize=9)
    image = cp.get_pixmap().tobytes("png")
    page.insert_image(fitz.Rect(45, 260, 375, 440), stream=image)
    chart.close()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()
    return path


if __name__ == "__main__":
    print(create_sample(Path("data/sample/solar_report.pdf")))
