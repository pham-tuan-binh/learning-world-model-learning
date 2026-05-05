import React from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import App from "./App.jsx";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <main className="flex min-h-screen w-full items-center justify-center px-4">
          <div className="w-full max-w-sm text-center font-mono">
            <p className="text-sm text-dark">Something went wrong.</p>
            <p className="mt-1 text-xs text-mid">{String(this.state.error)}</p>
            <button
              className="mt-4 rounded-sm border border-border bg-paper px-4 py-2 text-xs text-mid hover:bg-hover"
              onClick={() => window.location.reload()}
            >
              Reload
            </button>
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
