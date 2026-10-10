"use client";

import { useEffect } from "react";
import { useMap } from "react-leaflet";
import type { Map as LeafletMap } from "leaflet";

interface MapRefCaptureProps {
  onMapReady: (map: LeafletMap) => void;
}

export default function MapRefCapture({ onMapReady }: MapRefCaptureProps) {
  const map = useMap();
  useEffect(() => {
    if (map) {
      onMapReady(map);
    }
  }, [map, onMapReady]);
  return null;
}
