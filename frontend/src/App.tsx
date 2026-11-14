import { NavLink, Route, Routes } from "react-router-dom";
import AnkiExport from "./pages/AnkiExport";
import EquityTools from "./pages/EquityTools";
import HandReview from "./pages/HandReview";
import PostflopTrainer from "./pages/PostflopTrainer";
import PreflopDrill from "./pages/PreflopDrill";
import Progress from "./pages/Progress";
import PushFold from "./pages/PushFold";
import RangeEditor from "./pages/RangeEditor";
import RangeViewer from "./pages/RangeViewer";

const NAV = [
  {
    group: "Train",
    links: [
      { to: "/drill", label: "Preflop drill" },
      { to: "/postflop", label: "Postflop" },
      { to: "/pushfold", label: "Push/fold (ICM)" },
    ],
  },
  {
    group: "Study",
    links: [
      { to: "/", label: "Ranges" },
      { to: "/editor", label: "Range editor" },
      { to: "/tools", label: "Equity & odds" },
    ],
  },
  {
    group: "Review",
    links: [
      { to: "/review", label: "Hand review" },
      { to: "/progress", label: "Progress" },
      { to: "/anki", label: "Anki" },
    ],
  },
];

export default function App() {
  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-40 border-b border-surface-3 bg-surface-0/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-1 px-4 py-2.5">
          <span className="font-semibold tracking-tight">
            GTO<span className="text-act-call">trainer</span>
          </span>
          <nav className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {NAV.map((g) => (
              <div key={g.group} className="flex flex-wrap items-center gap-1" role="group" aria-label={g.group}>
                <span className="mr-1 text-[10px] font-semibold uppercase tracking-wider text-ink-3">
                  {g.group}
                </span>
                {g.links.map((n) => (
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
              </div>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-5">
        <Routes>
          <Route path="/" element={<RangeViewer />} />
          <Route path="/drill" element={<PreflopDrill />} />
          <Route path="/postflop" element={<PostflopTrainer />} />
          <Route path="/tools" element={<EquityTools />} />
          <Route path="/progress" element={<Progress />} />
          <Route path="/editor" element={<RangeEditor />} />
          <Route path="/review" element={<HandReview />} />
          <Route path="/pushfold" element={<PushFold />} />
          <Route path="/anki" element={<AnkiExport />} />
        </Routes>
      </main>
      <footer className="border-t border-surface-3 px-4 py-2 text-center text-[11px] text-ink-3">
        Training tool only — not for use during live play. Every strategy shows its source:
        approximate seed charts are not GTO.
      </footer>
    </div>
  );
}
