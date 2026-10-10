"use client";

import { useState } from "react";
import LandingPage from "@/components/site/LandingPage";
import AppLayout from "@/components/site/AppLayout";

export default function Home() {
  const [view, setView] = useState<"landing" | "app">("landing");

  if (view === "landing") {
    return <LandingPage onTryNow={() => setView("app")} />;
  }
  return <AppLayout onBackToLanding={() => setView("landing")} />;
}
