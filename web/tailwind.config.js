/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        panel: "#0f1419",
        surface: "#151b24",
        accent: "#22d3ee",
        warn: "#f59e0b",
        danger: "#ef4444",
        ok: "#22c55e",
      },
      fontFamily: {
        sans: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
