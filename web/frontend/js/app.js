/* GeoTree Frontend Interactive Application Script */

let map;
let drawnItems;
let activeDetectionsGroup;
let currentBboxLayer = null;

// Preset Region Coordinates
const REGIONS = {
    sundarbans: { name: "Sundarbans Mangrove Region", center: [21.9497, 89.1833], delta: 0.08 },
    sylhet: { name: "Sylhet Tea Gardens & Reserve", center: [24.8949, 91.8687], delta: 0.08 },
    chittagong: { name: "Chittagong Hill Tracts", center: [22.3569, 91.7832], delta: 0.08 },
    rangamati: { name: "Rangamati Forest Zone", center: [22.6533, 92.1753], delta: 0.08 },
    gazipur: { name: "Bhowal National Park (Gazipur)", center: [24.0958, 90.4125], delta: 0.06 },
    coxs_bazar: { name: "Cox's Bazar Coastal Forest", center: [21.4272, 92.0058], delta: 0.08 }
};

document.addEventListener("DOMContentLoaded", () => {
    initMap();
    loadRegion('sundarbans');
});

function initMap() {
    // Initialize Leaflet Map centered on Bangladesh
    map = L.map('map', {
        center: [23.6850, 90.3563],
        zoom: 8,
        zoomControl: true
    });

    // Base Imagery Layers (Satellite & Topo)
    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    }).addTo(map);

    const osmBase = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{y}/{x}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    });

    L.control.layers({
        "Satellite Imagery (Esri)": esriSatellite,
        "OpenStreetMap Standard": osmBase
    }, null, { position: 'topleft' }).addTo(map);

    // Feature Group to store drawn shapes
    drawnItems = new L.FeatureGroup().addTo(map);
    activeDetectionsGroup = L.FeatureGroup().addTo(map);

    // Leaflet Draw Control
    const drawControl = new L.Control.Draw({
        draw: {
            polygon: {
                allowIntersection: false,
                shapeOptions: { color: '#2ea44f', weight: 2 }
            },
            rectangle: {
                shapeOptions: { color: '#58a6ff', weight: 2 }
            },
            circle: false,
            polyline: false,
            marker: false,
            circlemarker: false
        },
        edit: {
            featureGroup: drawnItems,
            remove: true
        }
    });
    map.addControl(drawControl);

    // Event listener when user finishes drawing an area
    map.on(L.Draw.Event.CREATED, function (event) {
        const layer = event.layer;
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);
        
        const bounds = layer.getBounds();
        analyzeArea(bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast(), "Custom Selected Area");
    });
}

function loadRegion(regionKey) {
    const r = REGIONS[regionKey];
    if (!r) return;

    // Update button states
    document.querySelectorAll('.region-btn').forEach(btn => btn.classList.remove('active'));
    event.target?.classList.add('active');

    // Pan map to target region
    map.flyTo(r.center, 12, { duration: 1.5 });

    // Compute bounding box around center
    const s = r.center[0] - r.delta / 2;
    const w = r.center[1] - r.delta / 2;
    const n = r.center[0] + r.delta / 2;
    const e = r.center[1] + r.delta / 2;

    // Draw active bounding box rectangle on map
    if (currentBboxLayer) map.removeLayer(currentBboxLayer);
    currentBboxLayer = L.rectangle([[s, w], [n, e]], {
        color: "#58a6ff",
        weight: 2,
        dashArray: '4, 4',
        fillColor: "#58a6ff",
        fillOpacity: 0.1
    }).addTo(map);

    analyzeArea(s, w, n, e, r.name);
}

async function analyzeArea(south, west, north, east, titleName) {
    document.getElementById("regionTitle").innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing ${titleName}...`;

    try {
        const response = await fetch(`/api/analyze-bbox?south=${south}&west=${west}&north=${north}&east=${east}`);
        const data = await response.json();

        if (data.status === "success") {
            updateDashboard(data, titleName);
            renderDetectionsOnMap(data.tree_summary.sample_detections);
        }
    } catch (err) {
        console.error("Failed to analyze area:", err);
        document.getElementById("regionTitle").innerText = "Analysis Failed - Check Connection";
    }
}

function updateDashboard(data, titleName) {
    document.getElementById("regionTitle").innerText = titleName;
    document.getElementById("regionSubtitle").innerText = `Area: ${data.area.hectares} ha (${data.area.km2} km²) • Center: [${data.area.center.lat}, ${data.area.center.lng}]`;

    // Metrics
    document.getElementById("valTotalTrees").innerText = data.tree_summary.total_trees.toLocaleString();
    document.getElementById("valDensity").innerText = data.tree_summary.trees_per_ha;
    document.getElementById("valArea").innerText = `${data.area.hectares} ha`;
    document.getElementById("valCarbon").innerText = `${data.carbon_analytics.carbon_tonnes.toLocaleString()} t`;

    // Health
    const healthBadge = document.getElementById("healthBadge");
    healthBadge.innerText = Math.round(data.vegetation_health.score);
    document.getElementById("healthGrade").innerText = `Health Grade: ${data.vegetation_health.grade}`;
    document.getElementById("healthNdvi").innerText = `Mean NDVI: ${data.vegetation_health.mean_ndvi}`;

    // Land Cover Bars
    const lc = data.land_cover_pct;
    setBar("Forest", lc.forest);
    setBar("Veg", lc.other_vegetation);
    setBar("Water", lc.water);
    setBar("Built", lc.built_up);
    setBar("Bare", lc.bare_soil);

    // Carbon & Biomass
    document.getElementById("valBiomass").innerText = `${data.carbon_analytics.biomass_tonnes.toLocaleString()} t`;
    document.getElementById("valCo2").innerText = `${data.carbon_analytics.co2_equivalent_tonnes.toLocaleString()} t`;
}

function setBar(key, pct) {
    document.getElementById(`pct${key}`).innerText = `${pct}%`;
    document.getElementById(`bar${key}`).style.width = `${Math.min(pct, 100)}%`;
}

function renderDetectionsOnMap(detections) {
    activeDetectionsGroup.clearLayers();

    detections.forEach(d => {
        const circle = L.circle([d.lat, d.lng], {
            color: '#7ee787',
            fillColor: '#2ea44f',
            fillOpacity: 0.6,
            radius: d.crown_diameter_m / 2.0
        });

        circle.bindPopup(`
            <div style="font-size:12px; font-weight:600;">
                <span style="color:#2ea44f;">🌳 Tree Crown #${d.id}</span><br>
                Confidence: ${(d.confidence * 100).toFixed(1)}%<br>
                Crown Diameter: ${d.crown_diameter_m} m<br>
                Coord: [${d.lat.toFixed(5)}, ${d.lng.toFixed(5)}]
            </div>
        `);

        activeDetectionsGroup.addLayer(circle);
    });
}
