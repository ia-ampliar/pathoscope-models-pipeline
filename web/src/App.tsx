import { useState } from "react";
import { InferTab } from "./components/InferTab";
import { SplitTab } from "./components/SplitTab";
import { TcgaDownloadTab } from "./components/TcgaDownloadTab";
import { TrainTab } from "./components/TrainTab";

type Tab = "tcga" | "train" | "split" | "infer";

export default function App() {
  const [tab, setTab] = useState<Tab>("train");

  return (
    <div className="flex min-h-screen">
      <aside className="w-56 shrink-0 border-r border-slate-800 bg-panel p-4">
        <div className="mb-6 font-semibold tracking-tight text-accent">Pathoscope</div>
        <nav className="flex flex-col gap-1 text-sm">
          {(
            [
              ["tcga", "Download TCGA"],
              ["train", "Treino"],
              ["split", "Split"],
              ["infer", "Inferência"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`rounded-lg px-3 py-2 text-left transition ${
                tab === id ? "bg-surface text-accent" : "text-slate-400 hover:bg-surface/60 hover:text-slate-200"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
        <p className="mt-8 text-[10px] leading-relaxed text-slate-600">
          Dashboard dark para download TCGA (GDC), treino TensorFlow, split de dados e inferência WSI/TFLite.
        </p>
      </aside>
      <main className="min-w-0 flex-1 overflow-auto p-6">
        <header className="mb-6 border-b border-slate-800 pb-4">
          <h1 className="text-lg font-semibold text-slate-100">Control Room</h1>
          <p className="text-sm text-slate-500">
            Formulários espelham os parâmetros do backend; métricas em tempo real via WebSocket.
          </p>
        </header>
        {tab === "tcga" && <TcgaDownloadTab />}
        {tab === "train" && <TrainTab />}
        {tab === "split" && <SplitTab />}
        {tab === "infer" && <InferTab />}
      </main>
    </div>
  );
}
