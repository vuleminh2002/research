import json, random
from tqdm import tqdm

INPUT_FILE = "businesses.jsonl"
OUTPUT_FILE = "geocode_train_two_samples.jsonl"
NUM_SAMPLES = 2
CANDIDATE_COUNT = 50
MIN_INSIDE = 10
MAX_ATTEMPTS = 50

print("📦 Loading businesses...")
with open(INPUT_FILE) as f:
    all_points = [json.loads(line) for line in f]
print(f"✅ Loaded {len(all_points):,} businesses.\n")


def make_rectangle_around(anchor, lat_scale, lon_scale):
    lat_center, lon_center = anchor["lat"], anchor["lon"]
    lat1, lat2 = lat_center - lat_scale / 2, lat_center + lat_scale / 2
    lon1, lon2 = lon_center - lon_scale / 2, lon_center + lon_scale / 2
    return {
        "top_left": {"lat": round(max(lat1, lat2), 4), "lon": round(min(lon1, lon2), 4)},
        "bottom_right": {"lat": round(min(lat1, lat2), 4), "lon": round(max(lon1, lon2), 4)},
    }


def label_point(point, rect):
    lat, lon = point["lat"], point["lon"]
    tl, br = rect["top_left"], rect["bottom_right"]
    inside_lat = br["lat"] <= lat <= tl["lat"]
    inside_lon = tl["lon"] <= lon <= br["lon"]
    return inside_lat and inside_lon, inside_lat, inside_lon


def nearby_candidates(anchor, radius_deg=1.0, max_candidates=3000):
    lat_c, lon_c = anchor["lat"], anchor["lon"]
    return [
        p for p in all_points
        if abs(p["lat"] - lat_c) <= radius_deg and abs(p["lon"] - lon_c) <= radius_deg
    ][:max_candidates]


def generate_example():
    for _ in range(MAX_ATTEMPTS):
        anchor = random.choice(all_points)
        local_pool = nearby_candidates(anchor)
        if len(local_pool) < 60:
            continue

        lat_scale = lon_scale = random.uniform(0.02, 0.2)
        for _ in range(10):
            rect = make_rectangle_around(anchor, lat_scale, lon_scale)
            sample = random.sample(local_pool, CANDIDATE_COUNT)

            inside_ids, outside_ids, reasoning_lines = [], [], []

            lat_lo = rect['bottom_right']['lat']
            lat_hi = rect['top_left']['lat']
            lon_lo = rect['top_left']['lon']
            lon_hi = rect['bottom_right']['lon']

            for idx, p in enumerate(sample, 1):
                inside, inside_lat, inside_lon = label_point(p, rect)
                lat_phrase = f"lat {p['lat']:.4f} {'in' if inside_lat else 'not in'} [{lat_lo:.4f}, {lat_hi:.4f}]"
                lon_phrase = f"lon {p['lon']:.4f} {'in' if inside_lon else 'not in'} [{lon_lo:.4f}, {lon_hi:.4f}]"
                decision_phrase = "inside" if inside else "outside"

                reasoning_line = f"{idx}. {p['id']} → {lat_phrase}; {lon_phrase} → {decision_phrase}"
                reasoning_lines.append(reasoning_line)

                (inside_ids if inside else outside_ids).append(p["id"])

            if len(inside_ids) >= MIN_INSIDE:
                return {
                    "instruction": (
                        "Classify each candidate business as inside or outside the given rectangular range "
                        "based on its latitude and longitude. Output reasoning for each candidate, then list "
                        "final inside_ids and outside_ids."
                    ),
                    "input": (
                        "Rectangle:\n"
                        f"  top_left: ({rect['top_left']['lat']:.4f}, {rect['top_left']['lon']:.4f})\n"
                        f"  bottom_right: ({rect['bottom_right']['lat']:.4f}, {rect['bottom_right']['lon']:.4f})\n"
                        "Candidates:\n" +
                        "\n".join([f"  {p['id']}: ({p['lat']:.4f}, {p['lon']:.4f})" for p in sample])
                    ),
                    "output": (
                        "Reasoning:\n" +
                        "\n".join(reasoning_lines) + "\n\n" +
                        f"inside_ids: {inside_ids}\n" +
                        f"outside_ids: {outside_ids}"
                    ),
                }

            # Expand rectangle gradually to include more inside points
            lat_scale *= 1.3
            lon_scale *= 1.3

    return None


print("🚀 Generating 2 examples with ≥10 inside points each...")
with open(OUTPUT_FILE, "w") as f:
    count = 0
    for _ in tqdm(range(100)):  # try multiple times until 2 successful examples
        ex = generate_example()
        if ex:
            f.write(json.dumps(ex) + "\n")
            count += 1
            if count == NUM_SAMPLES:
                break

print(f"\n✅ Done! Created {count} samples and saved to {OUTPUT_FILE}")
