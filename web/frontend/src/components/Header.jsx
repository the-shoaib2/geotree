import styles from "./Header.module.css";

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <div className={styles.logoIcon}>
          <i className="fa-solid fa-tree" />
        </div>
        <div className={styles.logoText}>
          <span className={styles.title}>GeoTree</span>
        </div>
        <span className={styles.badge}>
          <span className={styles.pulseDot} /> Live
        </span>
      </div>
    </header>
  );
}
