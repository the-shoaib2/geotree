import { useState, useCallback } from "react";
import Header from "./components/Header";
import MapView from "./components/MapView";
import AnalyticsPanel from "./components/AnalyticsPanel";
import { analyzeBbox } from "./hooks/useAnalysis";

export default function App() {
  const [showLabels, setShowLabels] = useState(true);
  const [flyTarget] = useState(null);
  const [bbox, setBbox] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [regionName, setRegionName] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  /* ─── Draw created ─── */
  const handleDrawCreated = useCallback(async (s, w, n, e) => {
    setBbox({ south: s, west: w, north: n, east: e });
    await runAnalysis(s, w, n, e, "Selected Map Area");
  }, []);

  /* ─── Analyze button ─── */
  const handleAnalyze = useCallback(async () => {
    if (!bbox) return;
    await runAnalysis(bbox.south, bbox.west, bbox.north, bbox.east, regionName || "Selected Area");
  }, [bbox, regionName]);

  /* ─── Shared analysis runner ─── */
  async function runAnalysis(s, w, n, e, name) {
    setIsAnalyzing(true);
    setRegionName(name);
    try {
      const data = await analyzeBbox(s, w, n, e);
      setAnalysisData(data);
    } catch (err) {
      console.error("Analysis failed:", err);
    }
    setIsAnalyzing(false);
  }

  /* ─── Clear drawing ─── */
  const handleClear = useCallback(() => {
    setBbox(null);
    setAnalysisData(null);
    setRegionName("");
  }, []);

  return (
    <div style={{ position: "relative", height: "100vh", width: "100vw", overflow: "hidden" }}>
      {/* 100% Full-Screen Satellite Map Layer */}
      <div style={{ position: "absolute", inset: 0, zIndex: 1 }}>
        <MapView
          flyTarget={flyTarget}
          showLabels={showLabels}
          onToggleLabels={() => setShowLabels((prev) => !prev)}
          bbox={bbox}
          onDrawCreated={handleDrawCreated}
          onClearDraw={handleClear}
        />
      </div>

      {/* Floating Transparent Top Header */}
      <Header />

      {/* Floating Glassmorphic Analytics Overlay Modal Widget */}
      <AnalyticsPanel
        data={analysisData}
        regionName={regionName}
      />

      {/* 📍 Floating Center-Bottom Analyze Action Bar */}
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
              background: "rgba(255, 255, 255, 0.88)",
              backdropFilter: "blur(16px)",
              WebkitBackdropFilter: "blur(16px)",
              padding: "5px 14px",
              borderRadius: "9999px",
              fontSize: "11px",
              fontWeight: 600,
              color: "#334155",
              boxShadow: "0 4px 16px rgba(0, 0, 0, 0.08)",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              pointerEvents: "auto",
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#16a34a" }} />
            <span>
              {regionName || "Selected Area"}: {analysisData.area.hectares} ha • {analysisData.tree_summary.total_trees.toLocaleString()} trees
            </span>
          </div>
        )}

        {/* Center Bottom Main Analyze Button */}
        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing || !bbox}
          style={{
            background: isAnalyzing ? "#0284c7" : !bbox ? "#94a3b8" : "#0f172a",
            color: "#ffffff",
            border: "none",
            padding: "12px 28px",
            borderRadius: "9999px",
            fontSize: "14px",
            fontWeight: 700,
            cursor: isAnalyzing || !bbox ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            boxShadow: "0 10px 30px rgba(15, 23, 42, 0.28)",
            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
            pointerEvents: "auto",
            animation: isAnalyzing ? "analyzeGlow 1.5s ease infinite" : "none",
          }}
          onMouseEnter={(e) => {
            if (!isAnalyzing && bbox) {
              e.currentTarget.style.background = "#1e293b";
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow = "0 14px 36px rgba(15, 23, 42, 0.35)";
            }
          }}
          onMouseLeave={(e) => {
            if (!isAnalyzing && bbox) {
              e.currentTarget.style.background = "#0f172a";
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "0 10px 30px rgba(15, 23, 42, 0.28)";
            }
          }}
        >
          <i
            className={`fa-solid ${isAnalyzing ? "fa-spinner fa-spin" : "fa-satellite"}`}
            style={{ fontSize: "15px" }}
          />
          <span>{isAnalyzing ? "Analyzing Imagery..." : bbox ? "Analyze Selected Area" : "Draw Area on Map"}</span>
        </button>
      </div>

      {/* Loading Overlay */}
      {isAnalyzing && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(255,255,255,0.35)",
            backdropFilter: "blur(6px)",
            zIndex: 9999,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14, margin: "auto" }}>
            <div
              style={{
                width: 42,
                height: 42,
                border: "3px solid rgba(15,23,42,0.1)",
                borderTopColor: "#0f172a",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }}
            />
            <span style={{ fontSize: 13, fontWeight: 600, color: "#0f172a", display: "flex", alignItems: "center", gap: 8 }}>
              <i className="fa-solid fa-satellite" style={{ color: "#0284c7" }} />
              Analyzing satellite imagery...
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
