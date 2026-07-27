import styles from "./LayerSwitcher.module.css";

const LAYERS = [
  { key: "satellite", icon: "fa-satellite-dish", label: "Satellite" },
  { key: "terrain", icon: "fa-mountain-sun", label: "Terrain" },
  { key: "osm", icon: "fa-map", label: "Streets" },
];

export default function LayerSwitcher({ activeLayer, onChange }) {
  return (
    <div className={styles.switcher}>
      {LAYERS.map((l) => (
        <button
          key={l.key}
          className={`${styles.btn} ${activeLayer === l.key ? styles.active : ""}`}
          onClick={() => onChange(l.key)}
        >
          <i className={`fa-solid ${l.icon}`} />
          {l.label}
        </button>
      ))}
    </div>
  );
}
