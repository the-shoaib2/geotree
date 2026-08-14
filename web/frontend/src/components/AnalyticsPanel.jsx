import styles from "./AnalyticsPanel.module.css";

const LAND_COVER = [
  { key: "Forest", label: "Forest Cover", color: "#10b981" },
  { key: "Veg", label: "Vegetation", color: "#34d399" },
  { key: "Water", label: "Water Bodies", color: "#06b6d4" },
  { key: "Built", label: "Built-up Area", color: "#f43f5e" },
  { key: "Bare", label: "Bare Soil", color: "#f59e0b" },
];

const lcKeyMap = { Forest: "forest", Veg: "other_vegetation", Water: "water", Built: "built_up", Bare: "bare_soil" };

export default function AnalyticsPanel({ data, regionName, selectedPin }) {
  const d = data;
  const score = d?.vegetation_health?.score || 0;
  const circumference = 2 * Math.PI * 48;
  const fill = (score / 100) * circumference;
  const ringColor = score >= 80 ? "#10b981" : score >= 60 ? "#f59e0b" : "#f43f5e";

  return (
    <aside className={styles.panel}>
      {/* Header Card */}
      <div className={styles.panelHeader}>
        <div className={styles.headerRow}>
          <div className={styles.headerIcon}>
            <i className="fa-solid fa-chart-pie" style={{ color: "#10b981" }} />
          </div>
          <div>
            <div className={styles.panelTitle}>{regionName || "GIS Analytics"}</div>
            <div className={styles.panelSub}>
              {d ? `${d.area.hectares} ha • ${d.area.km2} km²` : selectedPin ? `Pin: ${selectedPin.lat.toFixed(3)}°, ${selectedPin.lng.toFixed(3)}°` : "Click map to drop selection pin"}
            </div>
          </div>
        </div>
      </div>

      <div className={styles.body}>
        {/* Key Metrics */}
        <div className={styles.sectionLabel}>Key Metrics</div>
        <div className={styles.metricsGrid}>
          <MetricCard icon="fa-tree" label="Trees Count" value={d ? d.tree_summary.total_trees.toLocaleString() : "--"} color="#10b981" />
          <MetricCard icon="fa-border-all" label="Density" value={d ? `${d.tree_summary.trees_per_ha}/ha` : "--"} color="#06b6d4" />
          <MetricCard icon="fa-chart-area" label="Area" value={d ? `${d.area.hectares} ha` : "--"} color="#f59e0b" />
          <MetricCard icon="fa-cloud-arrow-down" label="Carbon" value={d ? `${d.carbon_analytics.carbon_tonnes.toLocaleString()} t` : "--"} color="#38bdf8" />
        </div>

        {/* Vegetation Health */}
        <div className={styles.sectionLabel}>Vegetation Health</div>
        <div className={styles.healthBox}>
          <div className={styles.healthRing}>
            <svg viewBox="0 0 110 110">
              <circle className={styles.ringBg} cx="55" cy="55" r="48" />
              <circle
                className={styles.ringFill}
                cx="55"
                cy="55"
                r="48"
                style={{ stroke: ringColor, strokeDasharray: `${fill} ${circumference}` }}
              />
            </svg>
            <div className={styles.healthScore}>{d ? Math.round(score) : "--"}</div>
          </div>
          <div className={styles.healthInfo}>
            <div className={styles.gradeText}>{d ? d.vegetation_health.grade : "Grade: Pending"}</div>
            <div className={styles.ndviText}>Mean NDVI: {d ? d.vegetation_health.mean_ndvi : "--"}</div>
            <div className={styles.healthBarTrack}>
              <div className={styles.healthBarFill} style={{ width: `${score}%`, background: ringColor }} />
            </div>
          </div>
        </div>

        {/* Land Cover Classification */}
        <div className={styles.sectionLabel}>Land Cover Segmentation</div>
        <div className={styles.card}>
          {LAND_COVER.map(({ key, label, color }) => {
            const pct = d ? d.land_cover_pct[lcKeyMap[key]] || 0 : 0;
            return (
              <div className={styles.distItem} key={key}>
                <div className={styles.distHeader}>
                  <span>
                    <span className={styles.dot} style={{ background: color }} /> {label}
                  </span>
                  <span className={styles.pctBadge}>{pct}%</span>
                </div>
                <div className={styles.barBg}>
                  <div className={styles.barFill} style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Biomass & CO2 Analytics */}
        <div className={styles.sectionLabel}>Biomass & Carbon Metrics</div>
        <div className={styles.card}>
          <BioRow label="Total Biomass" value={d ? `${d.carbon_analytics.biomass_tonnes.toLocaleString()} tonnes` : "--"} />
          <BioRow label="CO₂ Equivalent Offset" value={d ? `${d.carbon_analytics.co2_equivalent_tonnes.toLocaleString()} tonnes` : "--"} />
        </div>

        {/* Pin Location Coordinates */}
        {selectedPin && (
          <div className={styles.coordsCard}>
            <i className="fa-solid fa-location-dot" style={{ color: "#10b981" }} />
            <span>Pin Location: [{selectedPin.lat.toFixed(4)}°, {selectedPin.lng.toFixed(4)}°]</span>
          </div>
        )}
      </div>
    </aside>
  );
}

function MetricCard({ icon, label, value, color }) {
  return (
    <div className={styles.metricCard}>
      <div className={styles.metricIcon} style={{ color }}><i className={`fa-solid ${icon}`} /></div>
      <div className={styles.metricLbl}>{label}</div>
      <div className={styles.metricVal}>{value}</div>
    </div>
  );
}

function BioRow({ label, value }) {
  return (
    <div className={styles.bioRow}>
      <span className={styles.bioLabel}>{label}</span>
      <span className={styles.bioVal}>{value}</span>
    </div>
  );
}
