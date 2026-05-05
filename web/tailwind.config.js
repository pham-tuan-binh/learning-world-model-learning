/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#FFFBF0",
        dark: "#171717",
        mid: "#7F7D78",
        cloudy: "#B2AFA8",
        mute: "#CCC8C0",
        border: "#E5E1D8",
        rule: "#F2EEE4",
        pampas: "#F9F5E8",
        hover: "#FCF9F0",
        blue: "#005EEA",
        crail: "#D97706",
        yellow: "#FFD500",
      },
      borderRadius: {
        sm: "2px",
      },
      fontFamily: {
        mono: ["IBM Plex Mono", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
