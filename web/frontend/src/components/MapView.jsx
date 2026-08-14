import { useEffect, useRef, useState, useCallback } from "react";
import { MapContainer, TileLayer, Rectangle, Polygon, Marker, Popup, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import "leaflet-draw";
import { SATELLITE_LAYER, LABELS_LAYER, BANGLADESH_BOUNDS, BANGLADESH_POLYGON, WORLD_OUTER_BOUNDS, REGIONS } from "../data/constants";
import styles from "./MapView.module.css";

/* ─── Custom Animated Map Selection Pin Marker Icon ─── */
const selectionPinIcon = L.divIcon({
  className: "custom-pin-marker-container",
  html: `
    <div class="pin-pulse-ring"></div>
    <div class="pin-icon-body">
      <i class="fa-solid fa-location-dot"></i>
    </div>
  `,
  iconSize: [40, 40],
  iconAnchor: [20, 36],
  popupAnchor: [0, -36],
});

/* ─── Custom Preset Region Pin Marker Icon ─── */
const createRegionPinIcon = (name, icon) =>
  L.divIcon({
    className: "region-pin-marker-container",
    html: `
      <div class="region-pin-body">
        <i class="fa-solid ${icon || 'fa-tree'}"></i>
        <span>${name}</span>
      </div>
    `,
    iconSize: [120, 30],
    iconAnchor: [60, 15],
    popupAnchor: [0, -15],
  });

/* ─── Map Click Listener Handler ─── */
function MapClickHandler({ onPointSelect, enabled }) {
  useMapEvents({
    click(e) {
      if (enabled) {
        onPointSelect(e.latlng.lat, e.latlng.lng);
      }
    },
  });
  return null;
}

/* ─── Fly-to helper with smooth ease ─── */
function FlyTo({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.flyTo(center, zoom, {
        duration: 1.8,
        easeLinearity: 0.25,
        animate: true,
      });
    }
  }, [center, zoom, map]);
  return null;
}

/* ─── Map Ref Binder ─── */
function MapRefBinder({ onMapReady }) {
  const map = useMap();
  useEffect(() => {
    onMapReady(map);
  }, [map, onMapReady]);
  return null;
}

/* ─── Imperative Draw Controller ─── */
function DrawController({ activeDrawType, onCreated, onDrawEnd }) {
  const map = useMap();
  const drawnRef = useRef(new L.FeatureGroup());
  const activeHandlerRef = useRef(null);

  useEffect(() => {
    const drawnItems = drawnRef.current;
    map.addLayer(drawnItems);

    const handler = (e) => {
      drawnItems.clearLayers();
      drawnItems.addLayer(e.layer);
      e.layer.setStyle({ color: "#10b981", weight: 2.5, dashArray: "6,4", fillColor: "#10b981", fillOpacity: 0.12 });
      const b = e.layer.getBounds();
      onCreated?.(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
      onDrawEnd?.();
    };

    map.on(L.Draw.Event.CREATED, handler);

    return () => {
      map.off(L.Draw.Event.CREATED, handler);
      map.removeLayer(drawnItems);
    };
  }, [map, onCreated, onDrawEnd]);

  useEffect(() => {
    if (activeHandlerRef.current) {
      activeHandlerRef.current.disable();
      activeHandlerRef.current = null;
    }

    if (activeDrawType === "rectangle") {
      activeHandlerRef.current = new L.Draw.Rectangle(map, {
        shapeOptions: { color: "#10b981", weight: 2.5, fillColor: "#10b981", fillOpacity: 0.12 },
      });
      activeHandlerRef.current.enable();
    }
  }, [map, activeDrawType]);

  return null;
}

export default function MapView({
  flyTarget,
  showLabels,
  onToggleLabels,
  bbox,
  selectedPin,
  onPointSelect,
  onRegionSelect,
  onDrawCreated,
  onClearDraw,
  analysisData,
}) {
  const [mapInstance, setMapInstance] = useState(null);
  const [activeDrawType, setActiveDrawType] = useState(null);
  const [pinToolActive, setPinToolActive] = useState(true);

  const handleStartDraw = () => {
    setActiveDrawType((prev) => (prev ? null : "rectangle"));
    if (!activeDrawType) setPinToolActive(false);
  };

  const handleTogglePinTool = () => {
    setPinToolActive((prev) => !prev);
    if (!pinToolActive) setActiveDrawType(null);
  };

  const handleDrawEnd = useCallback(() => {
    setActiveDrawType(null);
    setPinToolActive(true);
  }, []);

  const handleClear = () => {
    setActiveDrawType(null);
    onClearDraw?.();
  };

  const handleZoomIn = () => {
    mapInstance?.zoomIn(0.75);
  };

  const handleZoomOut = () => {
    mapInstance?.zoomOut(0.75);
  };

  const handleFitLocation = () => {
    mapInstance?.flyTo([23.685, 90.356], 7.25, {
      duration: 1.8,
      easeLinearity: 0.25,
    });
  };

  return (
    <div className={styles.container}>
      <MapContainer
        center={[23.685, 90.356]}
        zoom={7.25}
        minZoom={6.5}
        maxZoom={18}
        zoomSnap={0.25}
        zoomDelta={0.5}
        wheelDebounceTime={40}
        wheelPxPerZoomLevel={120}
        bounceAtZoomLimits={true}
        maxBounds={BANGLADESH_BOUNDS}
        maxBoundsViscosity={0.75}
        inertia={true}
        inertiaDeceleration={3000}
        easeLinearity={0.25}
        zoomControl={false}
        attributionControl={false}
        className={styles.map}
      >
        <MapRefBinder onMapReady={setMapInstance} />

        {/* Map Click Listener for Selection Pin Drop */}
        <MapClickHandler onPointSelect={onPointSelect} enabled={pinToolActive && !activeDrawType} />

        {/* Satellite Base Layer */}
        <TileLayer
          url={SATELLITE_LAYER.url}
          attribution={SATELLITE_LAYER.attribution}
          maxZoom={SATELLITE_LAYER.maxZoom}
          keepBuffer={4}
          updateWhenZooming={false}
          updateWhenIdle={true}
        />

        {/* Outer World Dark Mask for 100% Bangladesh Boundary Focus */}
        <Polygon
          positions={[WORLD_OUTER_BOUNDS, BANGLADESH_POLYGON]}
          pathOptions={{
            fillColor: "#030712",
            fillOpacity: 0.96,
            color: "#10b981",
            weight: 2.2,
          }}
        />

        {/* Satellite Reference Labels */}
        {showLabels && (
          <TileLayer
            url={LABELS_LAYER.url}
            attribution={LABELS_LAYER.attribution}
            maxZoom={LABELS_LAYER.maxZoom}
            keepBuffer={4}
          />
        )}

        <FlyTo center={flyTarget?.center} zoom={flyTarget?.zoom || 12} />
        <DrawController
          activeDrawType={activeDrawType}
          onCreated={onDrawCreated}
          onDrawEnd={handleDrawEnd}
        />

        {/* Preset Bangladesh Region Pin Markers */}
        {Object.entries(REGIONS).map(([key, reg]) => (
          <Marker
            key={key}
            position={reg.center}
            icon={createRegionPinIcon(reg.name.split(" ")[0], reg.icon)}
            eventHandlers={{
              click: () => onRegionSelect?.(key),
            }}
          />
        ))}

        {/* Active Selection Pin Marker with Glassmorphic Popup */}
        {selectedPin && (
          <Marker position={[selectedPin.lat, selectedPin.lng]} icon={selectionPinIcon}>
            <Popup autoPan={true} closeButton={true}>
              <div className={styles.popupContainer}>
                <div className={styles.popupTitle}>
                  <i className="fa-solid fa-location-dot" style={{ color: "#10b981" }} />
                  <span>{selectedPin.name || "Selected Map Location"}</span>
                </div>
                <div className={styles.popupCoords}>
                  Lat: {selectedPin.lat.toFixed(4)}° • Lng: {selectedPin.lng.toFixed(4)}°
                </div>
                {analysisData && (
                  <div className={styles.popupStats}>
                    <div className={styles.popupStatItem}>
                      <span className={styles.popupStatVal}>{analysisData.tree_summary.total_trees.toLocaleString()}</span>
                      <span className={styles.popupStatLbl}>Trees</span>
                    </div>
                    <div className={styles.popupStatItem}>
                      <span className={styles.popupStatVal}>{analysisData.area.hectares}</span>
                      <span className={styles.popupStatLbl}>Hectares</span>
                    </div>
                    <div className={styles.popupStatItem}>
                      <span className={styles.popupStatVal}>{analysisData.vegetation_health.score}</span>
                      <span className={styles.popupStatLbl}>Health</span>
                    </div>
                  </div>
                )}
              </div>
            </Popup>
          </Marker>
        )}

        {/* Region Bounding Box Outline */}
        {bbox && (
          <Rectangle
            bounds={[[bbox.south, bbox.west], [bbox.north, bbox.east]]}
            pathOptions={{ color: "#10b981", weight: 2.5, dashArray: "6,4", fillColor: "#10b981", fillOpacity: 0.12 }}
          />
        )}
      </MapContainer>

      {/* Floating Right-Side Glassmorphic Control Toolbar */}
      <div className={styles.toolbar}>
        {/* Drop Selection Pin Mode */}
        <button
          className={`${styles.toolBtn} ${pinToolActive && !activeDrawType ? styles.active : ""}`}
          title="Click Map to Drop Selection Pin"
          onClick={handleTogglePinTool}
        >
          <i className="fa-solid fa-location-dot" />
        </button>

        {/* Rectangle Area Selection Mode */}
        <button
          className={`${styles.toolBtn} ${activeDrawType ? styles.active : ""}`}
          title="Draw Area Rectangle on Map"
          onClick={handleStartDraw}
        >
          <i className="fa-solid fa-crop-simple" />
        </button>

        {/* Clear Active Selection */}
        {(bbox || selectedPin) && (
          <button className={styles.toolBtn} title="Clear Pin & Area Selection" onClick={handleClear}>
            <i className="fa-solid fa-trash-can" />
          </button>
        )}

        <div className={styles.divider} />

        {/* Zoom In (+) */}
        <button className={styles.toolBtn} title="Zoom In" onClick={handleZoomIn}>
          <i className="fa-solid fa-plus" />
        </button>

        {/* Zoom Out (-) */}
        <button className={styles.toolBtn} title="Zoom Out" onClick={handleZoomOut}>
          <i className="fa-solid fa-minus" />
        </button>

        {/* Reset View to Bangladesh Center */}
        <button className={styles.toolBtn} title="Reset View to Bangladesh" onClick={handleFitLocation}>
          <i className="fa-solid fa-crosshairs" />
        </button>
      </div>

      {/* Glassmorphic Map Labels Toggle */}
      <div className={styles.labelToggleContainer}>
        <button
          className={`${styles.labelToggleBtn} ${showLabels ? styles.active : ""}`}
          onClick={onToggleLabels}
        >
          <i className={`fa-solid ${showLabels ? "fa-eye" : "fa-eye-slash"}`} />
          <span>{showLabels ? "Labels On" : "Labels Off"}</span>
        </button>
      </div>
    </div>
  );
}
