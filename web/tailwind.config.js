/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        base:     "#0A0F12",
        panel:    "#0A0F12",
        surface:  "#121A1F",
        elevated: "#1A242B",
        "border-subtle": "#1F2F38",
        "border-strong": "#2A3F4A",
        accent:   "#2DD4D4",
        "teal-deep":  "#0E5A6F",
        "teal-soft":  "rgba(45,212,212,0.12)",
        "text-primary":   "#E6F4F4",
        "text-secondary": "#93B3B7",
        "text-muted":     "#5E7378",
        ok:      "#4ADE80",
        warn:    "#FACC15",
        danger:  "#F87171",
        running: "#60A5FA",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      keyframes: {
        beam: {
          "0%,100%": { opacity: "0.2", transform: "translateX(-100%)" },
          "50%":     { opacity: "1",   transform: "translateX(100%)" },
        },
        "pulse-glow": {
          "0%,100%": { boxShadow: "0 0 0px rgba(74,222,128,0)" },
          "50%":     { boxShadow: "0 0 24px rgba(74,222,128,0.5)" },
        },
        "step-pulse": {
          "0%,100%": { opacity: "1" },
          "50%":     { opacity: "0.5" },
        },
      },
      animation: {
        beam:        "beam 1.2s ease-in-out infinite",
        "pulse-glow":"pulse-glow 2s ease-in-out 3",
        "step-pulse":"step-pulse 1.5s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
