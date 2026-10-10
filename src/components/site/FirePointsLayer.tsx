"use client";

import { useEffect } from "react";
import type { FirePoint } from "@/lib/data";

interface FirePointsLayerProps {
  points: FirePoint[];
  visible: boolean;
  map: any;
  onPointClick?: (point: FirePoint) => void;
}

function createFireIcon(L: any, frp: number, source: "MODIS" | "VIIRS"): any {
  const size = Math.max(8, Math.min(18, 8 + frp / 10));
  const color = source === "MODIS" ? "#dc2626" : "#f59e0b";
  const glow = source === "MODIS" ? "rgba(220,38,38,0.5)" : "rgba(245,158,11,0.5)";

  return L.divIcon({
    className: "fire-point-icon",
    html: `
      <div style="
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        background: radial-gradient(circle, ${color} 30%, ${color}aa 70%, transparent 100%);
        border: 1.5px solid white;
        box-shadow: 0 0 ${size}px ${size * 0.6}px ${glow};
        cursor: pointer;
      "></div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function createPopupContent(p: FirePoint): string {
  const date = new Date(2023, 0, p.doy);
  const dateStr = `${date.getDate()}/${date.getMonth() + 1}/2023`;
  const confLabel = p.confidence >= 80 ? "Cao" : p.confidence >= 50 ? "TB" : "Thấp";
  const sourceColor = p.source === "MODIS" ? "#dc2626" : "#f59e0b";

  return `
    <div style="font-family: Inter, sans-serif; min-width: 200px;">
      <div style="font-weight: 700; color: #1c1917; font-size: 13px; margin-bottom: 6px;">
        🔥 Điểm cháy #${p.id + 1}
      </div>
      <div style="font-size: 11px; color: #6b7280; margin-bottom: 8px;">
        ${p.lat.toFixed(3)}°N, ${p.lon.toFixed(3)}°E · Vùng: ${
          p.region === "north" ? "Bắc" : p.region === "central" ? "Trung" : "Nam"
        }
      </div>
      <table style="font-size: 11px; width: 100%; border-collapse: collapse;"><tbody>
        <tr>
          <td style="padding: 3px 0; color: #6b7280;">Nguồn:</td>
          <td style="padding: 3px 0; font-weight: 600; color: ${sourceColor};">${p.source}</td>
        </tr>
        <tr>
          <td style="padding: 3px 0; color: #6b7280;">FRP:</td>
          <td style="padding: 3px 0; font-weight: 600; color: #1c1917;">${p.frp.toFixed(1)} MW</td>
        </tr>
        <tr>
          <td style="padding: 3px 0; color: #6b7280;">Brightness:</td>
          <td style="padding: 3px 0; font-weight: 600; color: #1c1917;">${p.brightness.toFixed(1)} K</td>
        </tr>
        <tr>
          <td style="padding: 3px 0; color: #6b7280;">Confidence:</td>
          <td style="padding: 3px 0; font-weight: 600; color: #1c1917;">${p.confidence}/100 (${confLabel})</td>
        </tr>
        <tr>
          <td style="padding: 3px 0; color: #6b7280;">Ngày:</td>
          <td style="padding: 3px 0; font-weight: 600; color: #1c1917;">${dateStr} (DOY ${p.doy})</td>
        </tr>
      </tbody></table>
    </div>
  `;
}

export default function FirePointsLayer({ points, visible, map, onPointClick }: FirePointsLayerProps) {
  useEffect(() => {
    if (!visible || points.length === 0 || !map) return;

    let clusterGroup: any = null;
    let cancelled = false;

    Promise.all([
      import("leaflet"),
      import("leaflet.markercluster"),
    ]).then(([LModule]) => {
      if (cancelled || !map) return;
      const L = LModule.default || LModule;

      clusterGroup = L.markerClusterGroup({
        maxClusterRadius: 35,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        iconCreateFunction: (cluster: any) => {
          const count = cluster.getChildCount();
          const size = Math.min(50, 25 + count);
          return L.divIcon({
            html: `<div style="
              width: ${size}px;
              height: ${size}px;
              border-radius: 50%;
              background: radial-gradient(circle, rgba(220,38,38,0.9) 30%, rgba(220,38,38,0.5) 70%, transparent 100%);
              color: white;
              display: flex;
              align-items: center;
              justify-content: center;
              font-weight: 700;
              font-size: 11px;
              border: 2px solid white;
              box-shadow: 0 2px 8px rgba(0,0,0,0.3);
              font-family: Inter, sans-serif;
            ">${count}</div>`,
            className: "fire-cluster-icon",
            iconSize: [size, size],
          });
        },
      });

      points.forEach(p => {
        const marker = L.marker([p.lat, p.lon], {
          icon: createFireIcon(L, p.frp, p.source),
        });

        marker.bindPopup(createPopupContent(p), {
          maxWidth: 280,
          className: "fire-popup",
        });

        if (onPointClick) {
          marker.on("click", () => onPointClick(p));
        }

        clusterGroup.addLayer(marker);
      });

      clusterGroup.addTo(map);
    });

    return () => {
      cancelled = true;
      if (clusterGroup && map) {
        try { map.removeLayer(clusterGroup); } catch (e) {}
      }
    };
  }, [points, visible, map, onPointClick]);

  return null;
}
