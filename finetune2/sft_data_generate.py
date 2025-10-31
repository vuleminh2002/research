import json, random
from tqdm import tqdm

INPUT_FILE = "businesses.jsonl"
OUTPUT_FILE = "geocode_train_randomized2.jsonl"
NUM_SAMPLES = 20
CANDIDATE_RANGE = (8, 14)
MAX_ATTEMPTS = 15
INSIDE_TARGET_RANGE = (0, 10)  # random target number of inside points per example

print("📦 Loading businesses...")
with open(INPUT_FILE) as f:
    all_points = [json.loads(line) for line in f]
print(f"✅ Loaded {len(all_points):,} businesses.\n")


def make_rectangle_around(anchor, lat_scale, lon_scale):
    lat_center, lon_center = anchor["lat"], anchor["lon"]
    lat_range = lat_scale
    lon_range = lon_scale
    lat1, lat2 = lat_center - lat_range / 2, lat_center + lat_range / 2
    lon1, lon2 = lon_center - lon_range / 2, lon_center + lon_range / 2
    return {
        "top_left": {"lat": round(max(lat1, lat2), 4), "lon": round(min(lon1, lon2), 4)},
        "bottom_right": {"lat": round(min(lat1, lat2), 4), "lon": round(max(lon1, lon2), 4)},
    }


def label_point(point, rect):
    lat, lon = point["lat"], point["lon"]
    tl, br = rect["top_left"], rect["bottom_right"]
    inside_lat = br["lat"] <= lat <= tl["lat"]
    inside_lon = tl["lon"] <= lon <= br["lon"]
    inside = inside_lat and inside_lon
    return inside, inside_lat, inside_lon


def nearby_candidates(anchor, radius_deg=1.0, max_candidates=2000):
    lat_c, lon_c = anchor["lat"], anchor["lon"]
    return [
        p for p in all_points
        if abs(p["lat"] - lat_c) <= radius_deg and abs(p["lon"] - lon_c) <= radius_deg
    ][:max_candidates]


def generate_example():
    target_inside = random.randint(*INSIDE_TARGET_RANGE)
    for _ in range(MAX_ATTEMPTS):
        anchor = random.choice(all_points)
        local_pool = nearby_candidates(anchor)
        if len(local_pool) < 25:
            continue

        lat_scale = lon_scale = random.uniform(0.02, 0.2)
        for _ in range(6):
            rect = make_rectangle_around(anchor, lat_scale, lon_scale)
            sample = random.sample(local_pool, random.randint(*CANDIDATE_RANGE))

            inside_ids, outside_ids, reasoning_lines = [], [], []

            # Precompute bounds
            lat_lo = rect['bottom_right']['lat']
            lat_hi = rect['top_left']['lat']
            lon_lo = rect['top_left']['lon']
            lon_hi = rect['bottom_right']['lon']

            for p in sample:
                inside, inside_lat, inside_lon = label_point(p, rect)
                lat_mark = "✓" if inside_lat else "✗"
                lon_mark = "✓" if inside_lon else "✗"
                decision_phrase = "both inside → inside" if inside else "one/both outside → outside"

                reasoning_line = (
                    f"- {p['id']}: "
                    f"lat {p['lat']:.4f} {lat_mark} [{lat_lo:.4f}–{lat_hi:.4f}], "
                    f"lon {p['lon']:.4f} {lon_mark} [{lon_lo:.4f}–{lon_hi:.4f}] "
                    f"→ {decision_phrase}"
                )
                reasoning_lines.append(reasoning_line)

                if inside:
                    inside_ids.append(p["id"])
                else:
                    outside_ids.append(p["id"])

            if abs(len(inside_ids) - target_inside) <= 1:
                return {
                    "instruction": (
                        "Classify each candidate business as inside or outside the given rectangular range "
                        "based on its latitude and longitude. Output step-by-step reasoning, then the final "
                        "inside_ids and outside_ids lists."
                    ),
                    "input": (
                        "Rectangle:\n"
                        f"  top_left: ({rect['top_left']['lat']:.4f}, {rect['top_left']['lon']:.4f})\n"
                        f"  bottom_right: ({rect['bottom_right']['lat']:.4f}, {rect['bottom_right']['lon']:.4f})\n"
                        "Candidates:\n" +
                        "\n".join([f"  {p['id']}: ({p['lat']:.4f}, {p['lon']:.4f})" for p in sample])
                    ),
                    "output": (
                        "Step-by-step reasoning:\n" +
                        "\n".join(reasoning_lines) + "\n\n" +
                        f"inside_ids: {inside_ids}\n" +
                        f"outside_ids: {outside_ids}"
                    ),
                }

            lat_scale *= 1.5
            lon_scale *= 1.5

    return None


print("🚀 Generating randomized inside-count dataset...")
with open(OUTPUT_FILE, "w") as f:
    for _ in tqdm(range(NUM_SAMPLES)):
        ex = generate_example()
        if ex:
            f.write(json.dumps(ex) + "\n")

print(f"\n✅ Done! Saved dataset to {OUTPUT_FILE}")
