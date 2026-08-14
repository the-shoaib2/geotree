import { useState, useCallback } from "react";
import Header from "./components/Header";
import MapView from "./components/MapView";
import AnalyticsPanel from "./components/AnalyticsPanel";
import { analyzeBbox } from "./hooks/useAnalysis";
import { REGIONS } from "./data/constants";

export default function App() {
  const [showLabels, setShowLabels] = useState(true);
  const [flyTarget, setFlyTarget] = useState(null);
  const [bbox, setBbox] = useState(null);
  const [selectedPin, setSelectedPin] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [regionName, setRegionName] = useState("");
  const [activeRegion, setActiveRegion] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  /* ─── Shared analysis runner ─── */
  const runAnalysis = useCallback(async (s, w, n, e, name) => {
    setIsAnalyzing(true);
    setRegionName(name);
    try {
      const data = await analyzeBbox(s, w, n, e);
      setAnalysisData(data);
    } catch (err) {
      console.error("Analysis failed:", err);
    }
    setIsAnalyzing(false);
  }, []);

  /* ─── Map Click Selection Pin Handler ─── */
  const handlePointSelect = useCallback(async (lat, lng) => {
    const delta = 0.008;
    const s = lat - delta;
    const w = lng - delta;
    const n = lat + delta;
    const e = lng + delta;

    setSelectedPin({ lat, lng, name: "Selected Location" });
    setBbox({ south: s, west: w, north: n, east: e });
    setActiveRegion(null);

    await runAnalysis(s, w, n, e, "Selected Map Pin");
  }, [runAnalysis]);

  /* ─── Region Preset Click Handler ─── */
  const handleRegionSelect = useCallback(async (regionKey) => {
    const reg = REGIONS[regionKey];
    if (!reg) return;

    setActiveRegion(regionKey);
    const [lat, lng] = reg.center;
    const delta = reg.delta || 0.08;

    const s = lat - delta;
    const w = lng - delta;
    const n = lat + delta;
    const e = lng + delta;

    setFlyTarget({ center: reg.center, zoom: reg.zoom });
    setSelectedPin({ lat, lng, name: reg.name });
    setBbox({ south: s, west: w, north: n, east: e });

    await runAnalysis(s, w, n, e, reg.name);
  }, [runAnalysis]);

  /* ─── Draw created rectangle handler ─── */
  const handleDrawCreated = useCallback(async (s, w, n, e) => {
    const centerLat = (s + n) / 2;
    const centerLng = (w + e) / 2;
    setSelectedPin({ lat: centerLat, lng: centerLng, name: "Custom Drawn Area" });
    setBbox({ south: s, west: w, north: n, east: e });
    setActiveRegion(null);
    await runAnalysis(s, w, n, e, "Custom Drawn Area");
  }, [runAnalysis]);

  /* ─── Analyze button ─── */
  const handleAnalyze = useCallback(async () => {
    if (!bbox) return;
    await runAnalysis(bbox.south, bbox.west, bbox.north, bbox.east, regionName || "Selected Area");
  }, [bbox, regionName, runAnalysis]);

  /* ─── Clear selection ─── */
  const handleClear = useCallback(() => {
    setBbox(null);
    setSelectedPin(null);
    setAnalysisData(null);
    setRegionName("");
    setActiveRegion(null);
  }, []);

  return (
    <div style={{ position: "relative", height: "100vh", width: "100vw", overflow: "hidden", background: "#070b14" }}>
      {/* 100% Full-Screen Satellite Map Layer */}
      <div style={{ position: "absolute", inset: 0, zIndex: 1 }}>
        <MapView
          flyTarget={flyTarget}
          showLabels={showLabels}
          onToggleLabels={() => setShowLabels((prev) => !prev)}
          bbox={bbox}
          selectedPin={selectedPin}
          onPointSelect={handlePointSelect}
          onRegionSelect={handleRegionSelect}
          onDrawCreated={handleDrawCreated}
          onClearDraw={handleClear}
          analysisData={analysisData}
        />
      </div>

      {/* Floating Transparent Glass Header */}
      <Header activeRegion={activeRegion} onSelectRegion={handleRegionSelect} />

      {/* Floating Glassmorphic Analytics Sidebar */}
      <AnalyticsPanel
        data={analysisData}
        regionName={regionName}
        selectedPin={selectedPin}
      />

      {/* 📍 Floating Center-Bottom Action & Status Bar */}
      <div
        style={{
          position: "absolute",
          bottom: "28px",
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 2000,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "8px",
          pointerEvents: "none",
        }}
      >
        {/* Selection Status Badge */}
        {analysisData && (
          <div
            style={{
              background: "rgba(15, 23, 42, 0.85)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              border: "1px solid rgba(16, 185, 129, 0.4)",
              padding: "6px 16px",
              borderRadius: "9999px",
              fontSize: "12px",
              fontWeight: 700,
              color: "#ffffff",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.45), 0 0 15px rgba(16, 185, 129, 0.2)",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              pointerEvents: "auto",
            }}
          >
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#10b981", boxShadow: "0 0 8px #10b981" }} />
            <span>
              {regionName || "Selected Location"}: {analysisData.area.hectares} ha • {analysisData.tree_summary.total_trees.toLocaleString()} trees detected
            </span>
          </div>
        )}

        {/* Center Bottom Main Analyze Button */}
        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing || !bbox}
          style={{
            background: isAnalyzing
              ? "linear-gradient(135deg, #06b6d4 0%, #0284c7 100%)"
              : !bbox
              ? "rgba(30, 41, 59, 0.8)"
              : "linear-gradient(135deg, #10b981 0%, #059669 100%)",
            color: "#ffffff",
            border: bbox ? "1px solid rgba(255, 255, 255, 0.25)" : "1px solid rgba(255, 255, 255, 0.1)",
            padding: "12px 30px",
            borderRadius: "9999px",
            fontSize: "14px",
            fontWeight: 800,
            cursor: isAnalyzing || !bbox ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            boxShadow: isAnalyzing
              ? "0 0 25px rgba(6, 182, 212, 0.5)"
              : bbox
              ? "0 8px 28px rgba(16, 185, 129, 0.4), 0 0 15px rgba(16, 185, 129, 0.3)"
              : "0 4px 16px rgba(0, 0, 0, 0.3)",
            transition: "all 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
            pointerEvents: "auto",
            animation: isAnalyzing ? "analyzeGlow 1.5s ease infinite" : "none",
          }}
        >
          <i
            className={`fa-solid ${isAnalyzing ? "fa-spinner fa-spin" : "fa-satellite"}`}
            style={{ fontSize: "16px" }}
          />
          <span>
            {isAnalyzing ? "Analyzing Imagery..." : bbox ? "Analyze Selected Location" : "Click Map to Drop Pin"}
          </span>
        </button>
      </div>

      {/* Loading Overlay with Frosted Backdrop */}
      {isAnalyzing && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(7, 11, 20, 0.65)",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            zIndex: 9999,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              background: "rgba(15, 23, 42, 0.88)",
              backdropFilter: "blur(24px)",
              border: "1px solid rgba(16, 185, 129, 0.4)",
              borderRadius: "24px",
              padding: "28px 40px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 16,
              boxShadow: "0 16px 48px rgba(0, 0, 0, 0.6), 0 0 30px rgba(16, 185, 129, 0.3)",
            }}
          >
            <div
              style={{
                width: 50,
                height: 50,
                border: "4px solid rgba(16, 185, 129, 0.2)",
                borderTopColor: "#10b981",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }}
            />
            <span
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: "#ffffff",
                display: "flex",
                alignItems: "center",
                gap: 10,
              }}
            >
              <i className="fa-solid fa-satellite" style={{ color: "#10b981" }} />
              Running GeoTree Satellite AI Analysis...
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
