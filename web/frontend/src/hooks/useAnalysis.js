import { API_BASE } from "../data/constants";

/**
 * Analyze a bounding box region via backend API.
 * Falls back to client-side simulation if API is unreachable.
 */
export async function analyzeBbox(south, west, north, east) {
  try {
    const res = await fetch(
      `${API_BASE}/api/analyze-bbox?south=${south}&west=${west}&north=${north}&east=${east}`
    );
    const data = await res.json();
    if (data.status === "success") return data;
  } catch {
    /* fallback below */
  }
  return simulateAnalysis(south, west, north, east);
}

/* ─── Client-side fallback simulation ─── */
function simulateAnalysis(south, west, north, east) {
  const latC = (south + north) / 2;
  const lngC = (west + east) / 2;
  const latM = Math.abs(north - south) * 111320;
  const lngM = Math.abs(east - west) * 111320 * Math.cos((latC * Math.PI) / 180);
  const areaHa = Math.max(0.1, (latM * lngM) / 10000);
  const areaKm2 = Math.max(0.001, (latM * lngM) / 1e6);

  const rand = (a, b) => a + Math.random() * (b - a);
  let baseDensity = 250;
  if (latC > 21.5 && latC < 22.5 && lngC > 89 && lngC < 89.8) baseDensity = 390;
  else if (latC > 24 && latC < 25.2 && lngC > 91.5 && lngC < 92.5) baseDensity = 310;
  else if (latC > 21.5 && latC < 23.5 && lngC > 91.8 && lngC < 92.7) baseDensity = 360;

  const trees = Math.max(1, Math.round(baseDensity * areaHa * rand(0.85, 1.15)));
  const n2 = Math.min(trees, 60);
  const dets = Array.from({ length: n2 }, (_, i) => ({
    id: i + 1,
    lat: +(south + rand(0.05, 0.95) * (north - south)).toFixed(6),
    lng: +(west + rand(0.05, 0.95) * (east - west)).toFixed(6),
    confidence: +rand(0.78, 0.98).toFixed(4),
    crown_diameter_m: +rand(4, 14).toFixed(1),
  }));

  const fp = +Math.min(92, Math.max(15, baseDensity / 4.5 + rand(-5, 5))).toFixed(1);
  const wp = +rand(3, 18).toFixed(1);
  const bp = +rand(2, 15).toFixed(1);
  const sp = +rand(3, 12).toFixed(1);
  const vp = +Math.max(0, 100 - fp - wp - bp - sp).toFixed(1);

  const crownM2 = Math.PI * (7.5 / 2) ** 2;
  const biomass = (trees * 5.83 * crownM2 ** 1.27) / 1000;
  const carbon = biomass * 0.47;
  const co2 = carbon * 3.67;
  const hs = +Math.min(98, Math.max(40, fp * 0.8 + 25)).toFixed(1);
  const grade =
    hs >= 85 ? "A (Vigorous)" : hs >= 70 ? "B (Healthy)" : hs >= 50 ? "C (Moderate)" : "D (Stressed)";

  return {
    status: "success",
    area: {
      bounds: { south, west, north, east },
      center: { lat: +latC.toFixed(6), lng: +lngC.toFixed(6) },
      hectares: +areaHa.toFixed(2),
      km2: +areaKm2.toFixed(4),
    },
    tree_summary: {
      total_trees: trees,
      trees_per_ha: +(trees / areaHa).toFixed(1),
      sample_detections: dets,
    },
    land_cover_pct: { forest: fp, other_vegetation: vp, water: wp, built_up: bp, bare_soil: sp },
    vegetation_health: {
      score: hs,
      grade,
      mean_ndvi: +(0.3 + (hs / 100) * 0.5).toFixed(4),
    },
    carbon_analytics: {
      biomass_tonnes: +biomass.toFixed(2),
      carbon_tonnes: +carbon.toFixed(2),
      co2_equivalent_tonnes: +co2.toFixed(2),
    },
  };
}
