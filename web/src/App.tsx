import { useState } from "react";
import { StepperRail } from "./components/StepperRail";
import { TcgaDownloadTab } from "./components/TcgaDownloadTab";
import { LabelTab } from "./components/LabelTab";
import { TilingTab } from "./components/TilingTab";
import { SplitTab } from "./components/SplitTab";
import { TrainTab } from "./components/TrainTab";
import { InferTab } from "./components/InferTab";
import type { StepId, StepState, StepStatus, InspectorMode } from "./types";

const LS_KEY = "pathoscope_step_state";
const LS_MODE_KEY = "pathoscope_inspector_mode";

function buildInitialState(): StepState {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as StepState;
      // Migrate any locked → available (old persisted state upgrade)
      const migrated = { ...parsed } as StepState;
      (Object.keys(migrated) as (keyof StepState)[]).forEach((k) => {
        if (migrated[k] === "locked") migrated[k] = "available";
      });
      return migrated;
    }
  } catch {}
  return {
    tcga:   "available",
    label:  "available",
    tiling: "available",
    split:  "available",
    train:  "available",
    infer:  "available",
  };
}

function persist(state: StepState) {
  try { localStorage.setItem(LS_KEY, JSON.stringify(state)); } catch {}
}

export default function App() {
  const [stepState, setStepState] = useState<StepState>(buildInitialState);
  const [activeStep, setActiveStep] = useState<StepId>("tcga");
  const [inspectorMode, setInspectorMode] = useState<InspectorMode>(() => {
    try { return (localStorage.getItem(LS_MODE_KEY) as InspectorMode) || "operator"; } catch { return "operator"; }
  });

  const updateStep = (id: StepId, status: StepStatus) => {
    setStepState((prev) => {
      const next = { ...prev, [id]: status };
      persist(next);
      return next;
    });
  };

  const toggleInspector = () => {
    setInspectorMode((m) => {
      const next: InspectorMode = m === "operator" ? "inspector" : "operator";
      try { localStorage.setItem(LS_MODE_KEY, next); } catch {}
      return next;
    });
  };

  const handleSelect = (id: StepId) => setActiveStep(id);

  const isInspector = inspectorMode === "inspector";

  const commonProps = { inspectorMode: isInspector };

  return (
    <div className="flex h-screen overflow-hidden bg-base text-text-primary font-sans">
      <StepperRail
        statuses={stepState}
        active={activeStep}
        onSelect={handleSelect}
        inspectorMode={isInspector}
        onToggleInspector={toggleInspector}
      />

      <main className="min-w-0 flex-1 overflow-y-auto bg-base text-text-primary">
        {activeStep === "tcga" && (
          <TcgaDownloadTab
            {...commonProps}
            onStatusChange={(s) => updateStep("tcga", s)}
          />
        )}
        {activeStep === "label" && (
          <LabelTab
            {...commonProps}
            onStatusChange={(s) => updateStep("label", s)}
          />
        )}
        {activeStep === "tiling" && (
          <TilingTab
            {...commonProps}
            onStatusChange={(s) => updateStep("tiling", s)}
          />
        )}
        {activeStep === "split" && (
          <SplitTab
            {...commonProps}
            onStatusChange={(s) => updateStep("split", s)}
          />
        )}
        {activeStep === "train" && (
          <TrainTab
            {...commonProps}
            onStatusChange={(s) => updateStep("train", s)}
          />
        )}
        {activeStep === "infer" && (
          <InferTab
            {...commonProps}
            onStatusChange={(s) => updateStep("infer", s)}
          />
        )}
      </main>
    </div>
  );
}
