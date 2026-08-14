import { REGIONS } from "../data/constants";
import styles from "./Header.module.css";

export default function Header({ activeRegion, onSelectRegion }) {
  return (
    <header className={styles.header}>
      {/* Left Logo Pill with Frosted Glass Backdrop */}
      <div className={styles.left}>
        <div className={styles.logoIcon}>
          <i className="fa-solid fa-tree" />
        </div>
        <div className={styles.logoText}>
          <span className={styles.title}>GeoTree</span>
        </div>
        <span className={styles.badge}>
          <span className={styles.pulseDot} /> Live Cloud
        </span>
      </div>

      {/* Center Region Quick Selector Pills (Glassmorphic) */}
      <div className={styles.centerRegions}>
        {Object.entries(REGIONS).map(([key, reg]) => (
          <button
            key={key}
            className={`${styles.regionPill} ${activeRegion === key ? styles.activeRegion : ""}`}
            onClick={() => onSelectRegion(key)}
            title={`Fly to ${reg.name}`}
          >
            <i className={`fa-solid ${reg.icon}`} />
            <span>{reg.name.split(" ")[0]}</span>
          </button>
        ))}
      </div>
    </header>
  );
}
