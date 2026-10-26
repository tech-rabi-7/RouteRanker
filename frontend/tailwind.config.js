/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cta: {
          blue: "#00a1de",
          red: "#c60c30",
          navy: "#0f172a",
          dark: "#0b1120",
          card: "#1e293b",
          border: "#334155",
        }
      }
    },
  },
  plugins: [],
}
