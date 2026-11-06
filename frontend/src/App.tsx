import { NavLink, Route, Routes } from "react-router-dom";
import PostflopTrainer from "./pages/PostflopTrainer";
import PreflopDrill from "./pages/PreflopDrill";
import Progress from "./pages/Progress";
import RangeViewer from "./pages/RangeViewer";

const NAV = [
  { to: "/", label: "Ranges" },
  { to: "/drill", label: "Preflop drill" },
  { to: "/postflop", label: "Postflop" },
  { to: "/tools", label: "Equity & odds" },
  { to: "/progress", label: "Progress" },
  { to: "/editor", label: "Range editor" },
];

export default function App() {
  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-40 border-b border-surface-3 bg-surface-0/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] items-center gap-6 px-4 py-2.5">
          <span className="font-semibold tracking-tight">
            GTO<span className="text-act-call">trainer</span>
          </span>
          <nav className="flex flex-wrap gap-1">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.to === "/"}
                className={({ isActive }) =>
                  `rounded-md px-3 py-1.5 text-sm ${
                    isActive ? "bg-surface-2 text-ink-1" : "text-ink-3 hover:text-ink-1"
                  }`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-5">
        <Routes>
          <Route path="/" element={<RangeViewer />} />
          <Route path="/drill" element={<PreflopDrill />} />
          <Route path="/postflop" element={<PostflopTrainer />} />
          
          <Route path="/progress" element={<Progress />} />
          
        </Routes>
      </main>
      <footer className="border-t border-surface-3 px-4 py-2 text-center text-[11px] text-ink-3">
        Training tool only — not for use during live play. Every strategy shows its source:
        approximate seed charts are not GTO.
      </footer>
    </div>
  );
}
