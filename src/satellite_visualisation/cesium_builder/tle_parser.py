import ephem


def parse_tle_file(tle_file_path):
    with open(tle_file_path, "r", encoding="utf-8") as f_tle:
        all_lines = [line.strip() for line in f_tle if line.strip()]
    header = None
    if (
        all_lines
        and len(all_lines[0].split()) == 2
        and all(part.isdigit() for part in all_lines[0].split())
    ):
        header = tuple(map(int, all_lines[0].split()))
        tle_lines = all_lines[1:]
    else:
        tle_lines = all_lines
    return header, tle_lines


def generate_satellites_from_tle(tle_lines):
    idx = 0
    satellites = []
    while idx + 2 < len(tle_lines):
        name, line1, line2 = tle_lines[idx], tle_lines[idx + 1], tle_lines[idx + 2]
        try:
            sat = ephem.readtle(name, line1, line2)
            satellites.append(sat)
        except Exception:
            pass
        idx += 3
    return satellites
