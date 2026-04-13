import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type MetricPoint = {
  idx: number;
  stage: string;
  epoch: number;
  loss?: number;
  val_loss?: number;
  acc?: number;
  val_acc?: number;
  warn?: boolean;
};

export function TrainCharts({
  data,
  pulse,
}: {
  data: MetricPoint[];
  pulse?: boolean;
}) {
  return (
    <div
      className={`grid gap-4 md:grid-cols-2 ${pulse ? "pulse-glow-green rounded-xl border border-ok/40 p-2" : ""}`}
    >
      <div className="rounded-lg border border-slate-800 bg-surface/60 p-3">
        <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Loss</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="idx" tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: "#151b24", border: "1px solid #334155" }}
                labelFormatter={(_, p) => {
                  const d = p?.[0]?.payload as MetricPoint | undefined;
                  return d ? `${d.stage} · época ${d.epoch}` : "";
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="loss"
                name="train"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={false}
                isAnimationActive
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                name="val"
                stroke="#a78bfa"
                strokeWidth={2}
                dot={(props: { cx?: number; cy?: number; payload?: MetricPoint }) => {
                  const { cx = 0, cy = 0, payload } = props;
                  if (payload?.warn) {
                    return <circle cx={cx} cy={cy} r={5} fill="#f59e0b" stroke="#fbbf24" strokeWidth={1} />;
                  }
                  return <circle cx={cx} cy={cy} r={0} />;
                }}
                isAnimationActive
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="rounded-lg border border-slate-800 bg-surface/60 p-3">
        <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Accuracy</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="idx" tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: "#151b24", border: "1px solid #334155" }}
                labelFormatter={(_, p) => {
                  const d = p?.[0]?.payload as MetricPoint | undefined;
                  return d ? `${d.stage} · época ${d.epoch}` : "";
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="acc"
                name="train acc"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="val_acc"
                name="val acc"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
