import styles from "./AnalyticsPanel.module.css";

const LAND_COVER = [
  { key: "Forest", label: "Forest", color: "#16a34a", grad: "#16a34a" },
  { key: "Veg", label: "Vegetation", color: "#22c55e", grad: "#22c55e" },
  { key: "Water", label: "Water", color: "#0284c7", grad: "#0284c7" },
  { key: "Built", label: "Built-up", color: "#dc2626", grad: "#dc2626" },
  { key: "Bare", label: "Bare Soil", color: "#d97706", grad: "#d97706" },
];

const lcKeyMap = { Forest: "forest", Veg: "other_vegetation", Water: "water", Built: "built_up", Bare: "bare_soil" };

export default function AnalyticsPanel({ data, regionName }) {
  const d = data;
  const score = d?.vegetation_health?.score || 0;
  const circumference = 2 * Math.PI * 48;
  const fill = (score / 100) * circumference;
  const ringColor = score >= 80 ? "#16a34a" : score >= 60 ? "#d97706" : "#dc2626";

  return (
    <aside className={styles.panel}>
      {/* Header */}
      <div className={styles.panelHeader}>
        <div className={styles.headerRow}>
          <div className={styles.headerIcon}><i className="fa-solid fa-chart-pie" /></div>
          <div>
            <div className={styles.panelTitle}>{regionName || "Analytics"}</div>
            <div className={styles.panelSub}>
              {d ? `${d.area.hectares} ha • ${d.area.km2} km²` : "Select an area on the map"}
            </div>
          </div>
        </div>
      </div>

      <div className={styles.body}>
        {/* Metrics */}
        <div className={styles.sectionLabel}>Metrics</div>
        <div className={styles.metricsGrid}>
          <MetricCard icon="fa-tree" label="Trees" value={d ? d.tree_summary.total_trees.toLocaleString() : "--"} color="#16a34a" />
          <MetricCard icon="fa-border-all" label="Density" value={d ? `${d.tree_summary.trees_per_ha}/ha` : "--"} color="#0891b2" />
          <MetricCard icon="fa-chart-area" label="Area" value={d ? `${d.area.hectares} ha` : "--"} color="#d97706" />
          <MetricCard icon="fa-cloud-arrow-down" label="Carbon" value={d ? `${d.carbon_analytics.carbon_tonnes.toLocaleString()} t` : "--"} color="#0284c7" />
        </div>

        {/* Health */}
        <div className={styles.sectionLabel}>Vegetation Health</div>
        <div className={styles.healthBox}>
          <div className={styles.healthRing}>
            <svg viewBox="0 0 110 110">
              <circle className={styles.ringBg} cx="55" cy="55" r="48" />
              <circle className={styles.ringFill} cx="55" cy="55" r="48"
                style={{ stroke: ringColor, strokeDasharray: `${fill} ${circumference}` }} />
            </svg>
            <div className={styles.healthScore}>{d ? Math.round(score) : "--"}</div>
          </div>
          <div className={styles.healthInfo}>
            <div className={styles.gradeText}>{d ? d.vegetation_health.grade : "Grade: --"}</div>
            <div className={styles.ndviText}>NDVI: {d ? d.vegetation_health.mean_ndvi : "--"}</div>
            <div className={styles.healthBarTrack}>
              <div className={styles.healthBarFill} style={{ width: `${score}%`, background: ringColor }} />
            </div>
          </div>
        </div>

        {/* Land Cover */}
        <div className={styles.sectionLabel}>Land Cover</div>
        <div className={styles.card}>
          {LAND_COVER.map(({ key, label, color }) => {
            const pct = d ? d.land_cover_pct[lcKeyMap[key]] || 0 : 0;
            return (
              <div className={styles.distItem} key={key}>
                <div className={styles.distHeader}>
                  <span><span className={styles.dot} style={{ background: color }} /> {label}</span>
                  <span className={styles.pctBadge}>{pct}%</span>
                </div>
                <div className={styles.barBg}>
                  <div className={styles.barFill} style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Biomass */}
        <div className={styles.sectionLabel}>Biomass & Carbon</div>
        <div className={styles.card}>
          <BioRow label="Biomass" value={d ? `${d.carbon_analytics.biomass_tonnes.toLocaleString()} t` : "-- t"} />
          <BioRow label="CO₂ Offset" value={d ? `${d.carbon_analytics.co2_equivalent_tonnes.toLocaleString()} t` : "-- t"} />
        </div>

        {/* Coords Badge */}
        {d && (
          <div className={styles.coordsCard}>
            <i className="fa-solid fa-location-dot" />
            <span>[{d.area.center.lat}, {d.area.center.lng}]</span>
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
