import { useEffect, useRef, useState, useCallback } from "react";
import { MapContainer, TileLayer, Rectangle, Polygon, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import "leaflet-draw";
import { SATELLITE_LAYER, LABELS_LAYER, BANGLADESH_BOUNDS, BANGLADESH_POLYGON, WORLD_OUTER_BOUNDS } from "../data/constants";
import styles from "./MapView.module.css";

/* ─── Fly-to helper with smooth ease ─── */
function FlyTo({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.flyTo(center, zoom, {
        duration: 2.0,
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

  // Setup drawn items group & event listener
  useEffect(() => {
    const drawnItems = drawnRef.current;
    map.addLayer(drawnItems);

    const handler = (e) => {
      drawnItems.clearLayers();
      drawnItems.addLayer(e.layer);
      e.layer.setStyle({ color: "#0284c7", weight: 2, dashArray: "6,4", fillColor: "#0284c7", fillOpacity: 0.08 });
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

  // Handle trigger of single rectangle area selection tool
  useEffect(() => {
    if (activeHandlerRef.current) {
      activeHandlerRef.current.disable();
      activeHandlerRef.current = null;
    }

    if (activeDrawType === "rectangle") {
      activeHandlerRef.current = new L.Draw.Rectangle(map, {
        shapeOptions: { color: "#0284c7", weight: 2, fillColor: "#0284c7", fillOpacity: 0.08 },
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
  onDrawCreated,
  onClearDraw,
}) {
  const [mapInstance, setMapInstance] = useState(null);
  const [activeDrawType, setActiveDrawType] = useState(null);

  const handleStartDraw = () => {
    setActiveDrawType((prev) => (prev ? null : "rectangle"));
  };

  const handleDrawEnd = useCallback(() => {
    setActiveDrawType(null);
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
    mapInstance?.flyTo([23.685, 90.356], 7, {
      duration: 1.8,
      easeLinearity: 0.25,
    });
  };

  return (
    <div className={styles.container}>
      <MapContainer
        center={[23.685, 90.356]}
        zoom={7}
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

        {/* Satellite Base Layer with High-Performance Tile Buffering */}
        <TileLayer
          url={SATELLITE_LAYER.url}
          attribution={SATELLITE_LAYER.attribution}
          maxZoom={SATELLITE_LAYER.maxZoom}
          keepBuffer={4}
          updateWhenZooming={false}
          updateWhenIdle={true}
        />

        {/* Solid Dark Mask for 100% hiding all non-Bangladesh regions */}
        <Polygon
          positions={[WORLD_OUTER_BOUNDS, BANGLADESH_POLYGON]}
          pathOptions={{
            fillColor: "#070b14",
            fillOpacity: 1.0,
            color: "#16a34a",
            weight: 2.5,
          }}
        />

        {/* Satellite Reference Labels Overlay (Only inside Bangladesh) */}
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

        {/* Region bounding box */}
        {bbox && (
          <Rectangle
            bounds={[[bbox.south, bbox.west], [bbox.north, bbox.east]]}
            pathOptions={{ color: "#0284c7", weight: 2, dashArray: "6,4", fillColor: "#0284c7", fillOpacity: 0.06 }}
          />
        )}
      </MapContainer>

      {/* Floating Right-Side Map Control Toolbar (Single Draw Button, Zoom +, Zoom -, Fit Location, Clear) */}
      <div className={styles.toolbar}>
        {/* Single Unified Area Selection Button */}
        <button
          className={`${styles.toolBtn} ${activeDrawType ? styles.active : ""}`}
          title="Select Area on Map"
          onClick={handleStartDraw}
        >
          <i className="fa-solid fa-crop-simple" />
        </button>

        {/* Clear Selection */}
        {bbox && (
          <button className={styles.toolBtn} title="Clear Selection" onClick={handleClear}>
            <i className="fa-solid fa-eraser" />
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

        {/* Fit Location / Reset View to Bangladesh */}
        <button className={styles.toolBtn} title="Fit Bangladesh View" onClick={handleFitLocation}>
          <i className="fa-solid fa-crosshairs" />
        </button>
      </div>

      {/* Enable/Disable Labels Toggle (Shadcn Rounded Full) */}
      <div className={styles.labelToggleContainer}>
        <button
          className={`${styles.labelToggleBtn} ${showLabels ? styles.active : ""}`}
          onClick={onToggleLabels}
        >
          <i className={`fa-solid ${showLabels ? "fa-eye" : "fa-eye-slash"}`} />
          <span>{showLabels ? "Labels Enabled" : "Labels Disabled"}</span>
        </button>
      </div>
    </div>
  );
}
