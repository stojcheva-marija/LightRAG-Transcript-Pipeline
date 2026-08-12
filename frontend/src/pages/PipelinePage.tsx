import { useState, useRef } from "react";
import {
  Button,
  Form,
  Input,
  Upload,
  Tabs,
  Alert,
  Typography,
  Space,
  Divider,
  Steps,
  Tag,
  Progress,
  DatePicker,
  TimePicker,
} from "antd";
import {
  UploadOutlined,
  PlayCircleOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
  CloseCircleOutlined,
} from "@ant-design/icons";

import type { UploadFile } from "antd/es/upload/interface";
import { runPipeline, resumePipeline, PipelineEvent } from "../api";

const { Paragraph } = Typography;

const RUN_STEPS = [
  { key: "saving",       label: "Saving audio",           match: "Step 1" },
  { key: "diarization",  label: "Analysing speakers",     match: "Step 2" },
  { key: "transcribing", label: "Transcribing",           match: "Transcribing" },
  { key: "summary",      label: "Generating summary",     match: "Generating metadata" },
  { key: "saving",       label: "Saving results",         match: "Uploading outputs" },
];

const RESUME_STEPS = [
  { key: "loading",      label: "Loading cached data",    match: "Step 1" },
  { key: "transcribing", label: "Transcribing",           match: "Step 2" },
  { key: "summary",      label: "Generating summary",     match: "Generating metadata" },
  { key: "saving",       label: "Saving results",         match: "Uploading outputs" },
];

type StepStatus = "wait" | "process" | "finish" | "error";

interface StepState {
  status: StepStatus;
}

function getStepIndex(steps: typeof RUN_STEPS, message: string): number {
  for (let i = steps.length - 1; i >= 0; i--) {
    if (message.includes(steps[i].match)) return i;
  }
  return -1;
}

function PipelineProgress({
  steps,
  stepStates,
  error,
}: {
  steps: typeof RUN_STEPS;
  stepStates: StepState[];
  error: string | null;
}) {
  const finished = stepStates.filter((s) => s.status === "finish").length;
  const hasError = stepStates.some((s) => s.status === "error");
  const percent = Math.round((finished / steps.length) * 100);

  return (
    <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 16, alignItems: "center" }}>
      <div style={{ width: "80%" }}>
        <Progress
          percent={percent}
          status={hasError ? "exception" : percent === 100 ? "success" : "active"}
          strokeColor={hasError ? undefined : { "0%": "#108ee9", "100%": "#1677ff" }}
        />
      </div>
      <div style={{ width: "80%" }}>
        <Steps
          size="small"
          items={steps.map((step, i) => {
            const state = stepStates[i];
            return {
              title: step.label,
              status: state.status,
              icon: state.status === "process" ? <LoadingOutlined /> :
                    state.status === "error"   ? <CloseCircleOutlined /> : undefined,
            };
          })}
        />
      </div>
      {error && (
        <div style={{ width: "80%" }}>
          <Alert type="error" message={error} style={{ marginTop: 4 }} />
        </div>
      )}
    </div>
  );
}

function DoneResult({ data }: { data: Record<string, string> }) {
  const metadata = data.metadata ? (() => { try { return JSON.parse(data.metadata); } catch { return null; } })() : null;

  const collapseItems = [];

  if (data.transcript) {
    const lineRe = /^(.+?) \((\d+:\d+) - (\d+:\d+)\):\s*(.*)$/;
    interface Segment { speaker: string; start: string; end: string; lines: string[] }
    const merged: Segment[] = [];
    for (const raw of data.transcript.split("\n")) {
      const m = raw.match(lineRe);
      if (!m) { if (raw.trim()) merged.push({ speaker: "", start: "", end: "", lines: [raw] }); continue; }
      const [, speaker, start, end, text] = m;
      const prev = merged[merged.length - 1];
      if (prev && prev.speaker === speaker) {
        prev.end = end;
        prev.lines.push(text);
      } else {
        merged.push({ speaker, start, end, lines: [text] });
      }
    }
    collapseItems.push({
      key: "transcript",
      label: "Transcript Preview",
      children: (
        <div style={{ maxHeight: 500, overflowY: "auto", fontSize: 12 }}>
          {merged.map((seg, i) => (
            <div key={i} style={{ marginBottom: 12 }}>
              {seg.speaker && (
                <div style={{ fontWeight: 600, marginBottom: 2 }}>
                  {seg.speaker}{seg.start ? ` (${seg.start} - ${seg.end})` : ""}
                </div>
              )}
              <div style={{ whiteSpace: "pre-wrap" }}>{seg.lines.join(" ")}</div>
            </div>
          ))}
        </div>
      ),
    });
  }

  if (data.resolution_report) {
    const lines = data.resolution_report
      .split("\n")
      .filter((l: string) => l.includes("→"));
    collapseItems.push({
      key: "speakers",
      label: "Speakers",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {lines.map((line: string, i: number) => {
            const match = line.match(/\*\*(.+?)\*\*\s*→\s*`(.+?)`\s*(.*)/);
            if (!match) return <div key={i}>{line}</div>;
            const [, id, name, note] = match;
            const isUnknown = note.includes("unknown");
            return (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Tag color={isUnknown ? "default" : "green"}>{id}</Tag>
                <span style={{ fontWeight: 500 }}>{isUnknown ? "Unknown" : name}</span>
                {!isUnknown && <Tag color="blue">{note.includes("DB") ? "Matched from DB" : "Auto-detected"}</Tag>}
              </div>
            );
          })}
        </div>
      ),
    });
  }

  if (metadata?.summary) {
    collapseItems.push({
      key: "summary",
      label: "Summary",
      children: <Paragraph style={{ margin: 0 }}>{metadata.summary}</Paragraph>,
    });
  }

  if (metadata?.topics?.length > 0) {
    collapseItems.push({
      key: "topics",
      label: "Topics",
      children: (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {metadata.topics.map((t: string, i: number) => (
            <Tag key={i} color="blue">{t}</Tag>
          ))}
        </div>
      ),
    });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {collapseItems.length > 0 && (
        <Tabs
          type="card"
          items={collapseItems.map((item) => ({
            key: item.key,
            label: item.label,
            children: <div style={{ padding: "8px 0" }}>{item.children}</div>,
          }))}
        />
      )}
    </div>
  );
}

function MetadataFields() {
  return (
    <>
      <Form.Item label="Show name" name="show_name">
        <Input placeholder="e.g. Parliament Session 77" />
      </Form.Item>
      <Form.Item label="Date" name="date">
        <DatePicker style={{ width: "100%" }} format="YYYY-MM-DD" placeholder="e.g. 2025-09-12" />
      </Form.Item>
      <Form.Item label="Time" name="time">
        <TimePicker style={{ width: "100%" }} format="HH:mm" placeholder="e.g. 14:30" />
      </Form.Item>
      <Form.Item label="Location" name="location">
        <Input placeholder="e.g. Parliament Hall of Macedonia" />
      </Form.Item>
      <Form.Item
        label="Participants"
        name="known_speakers"
        extra="Comma-separated list of participant names"
      >
        <Input placeholder="e.g. Robert Brown, Sarah Miller" />
      </Form.Item>
    </>
  );
}

function initSteps(steps: typeof RUN_STEPS): StepState[] {
  return steps.map(() => ({ status: "wait" }));
}

function RunTab() {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [running, setRunning] = useState(false);
  const [started, setStarted] = useState(false);
  const [stepStates, setStepStates] = useState<StepState[]>(initSteps(RUN_STEPS));
  const [error, setError] = useState<string | null>(null);
  const [doneData, setDoneData] = useState<Record<string, string> | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  function handleEvent(event: PipelineEvent) {
    if (event.type === "log") {
      const msg = event.data.message;
      const idx = getStepIndex(RUN_STEPS, msg);
      if (idx >= 0) {
        setStepStates((prev) => {
          const next = [...prev];
          // finish all previous
          for (let i = 0; i < idx; i++) next[i] = { status: "finish" };
          next[idx] = { status: "process" };
          return next;
        });
      }
      // "Knowledge graph updated" means last step done
      if (msg.includes("Knowledge graph updated")) {
        setStepStates(RUN_STEPS.map(() => ({ status: "finish" })));
      }
    } else if (event.type === "error") {
      setError(event.data.message);
      setStepStates((prev) => {
        const next = [...prev];
        const idx = next.findIndex((s) => s.status === "process");
        if (idx >= 0) next[idx] = { status: "error" };
        return next;
      });
      setRunning(false);
    } else if (event.type === "done") {
      setDoneData(event.data);
      setRunning(false);
    }
  }

  function handleSubmit(values: Record<string, unknown>) {
    const file = fileList[0]?.originFileObj;
    if (!file) return;

    const formData = new FormData();
    formData.append("audio_file", file);
    Object.entries(values).forEach(([k, v]) => {
      if (!v) return;
      if (k === "date") formData.append(k, (v as { format: (s: string) => string }).format("YYYY-MM-DD"));
      else if (k === "time") formData.append(k, (v as { format: (s: string) => string }).format("HH:mm"));
      else formData.append(k, v as string);
    });

    setStepStates(initSteps(RUN_STEPS));
    setError(null);
    setDoneData(null);
    setStarted(true);
    setRunning(true);
    cancelRef.current = runPipeline(formData, handleEvent);
  }

  function handleCancel() {
    cancelRef.current?.();
    setRunning(false);
    setError("Cancelled by user");
    setStepStates((prev) => {
      const next = [...prev];
      const idx = next.findIndex((s) => s.status === "process");
      if (idx >= 0) next[idx] = { status: "error" };
      return next;
    });
  }

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Form.Item label="Audio file" required>
        <Upload
          beforeUpload={() => false}
          fileList={fileList}
          onChange={({ fileList }) => setFileList(fileList.slice(-1))}
          accept="audio/*"
        >
          <Button icon={<UploadOutlined />}>Upload Audio File</Button>
        </Upload>
      </Form.Item>

      <MetadataFields />

      <Form.Item>
        <Space>
          <Button
            type="primary"
            htmlType="submit"
            icon={<PlayCircleOutlined />}
            loading={running}
            disabled={fileList.length === 0}
          >
            Upload & Transcribe
          </Button>
          {running && (
            <Button danger onClick={handleCancel}>Cancel</Button>
          )}
        </Space>
      </Form.Item>

      {started && (
        <>
          <Divider />
          <PipelineProgress steps={RUN_STEPS} stepStates={stepStates} error={error} />
        </>
      )}
      {doneData && <div style={{ marginTop: 16 }}><DoneResult data={doneData} /></div>}
    </Form>
  );
}

function ResumeTab() {
  const [form] = Form.useForm();
  const [running, setRunning] = useState(false);
  const [started, setStarted] = useState(false);
  const [stepStates, setStepStates] = useState<StepState[]>(initSteps(RESUME_STEPS));
  const [error, setError] = useState<string | null>(null);
  const [doneData, setDoneData] = useState<Record<string, string> | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  function handleEvent(event: PipelineEvent) {
    if (event.type === "log") {
      const msg = event.data.message;
      const idx = getStepIndex(RESUME_STEPS, msg);
      if (idx >= 0) {
        setStepStates((prev) => {
          const next = [...prev];
          for (let i = 0; i < idx; i++) next[i] = { status: "finish" };
          next[idx] = { status: "process" };
          return next;
        });
      }
      if (msg.includes("Knowledge graph updated")) {
        setStepStates(RESUME_STEPS.map(() => ({ status: "finish" })));
      }
    } else if (event.type === "error") {
      setError(event.data.message);
      setStepStates((prev) => {
        const next = [...prev];
        const idx = next.findIndex((s) => s.status === "process");
        if (idx >= 0) next[idx] = { status: "error" };
        return next;
      });
      setRunning(false);
    } else if (event.type === "done") {
      setDoneData(event.data);
      setRunning(false);
    }
  }

  function handleSubmit(rawValues: Record<string, unknown>) {
    const values: Record<string, string> = {};
    Object.entries(rawValues).forEach(([k, v]) => {
      if (!v) return;
      if (k === "date") values[k] = (v as { format: (s: string) => string }).format("YYYY-MM-DD");
      else if (k === "time") values[k] = (v as { format: (s: string) => string }).format("HH:mm");
      else values[k] = v as string;
    });
    setStepStates(initSteps(RESUME_STEPS));
    setError(null);
    setDoneData(null);
    setStarted(true);
    setRunning(true);
    cancelRef.current = resumePipeline(values, handleEvent);
  }

  function handleCancel() {
    cancelRef.current?.();
    setRunning(false);
    setError("Cancelled by user");
    setStepStates((prev) => {
      const next = [...prev];
      const idx = next.findIndex((s) => s.status === "process");
      if (idx >= 0) next[idx] = { status: "error" };
      return next;
    });
  }

  return (
    <Form form={form} layout="vertical" onFinish={handleSubmit}>
      <Alert
        type="info"
        icon={<InfoCircleOutlined />}
        showIcon
        message="Reuses cached speaker analysis from a previous run — skips the diarization step."
        style={{ marginBottom: 16 }}
      />
      <Form.Item
        label="Audio Name"
        name="stem"
        rules={[{ required: true, message: "Enter the audio name" }]}
        extra="Audio name without extension, e.g. parliament_session_77"
      >
        <Input placeholder="parliament_session_77" />
      </Form.Item>

      <MetadataFields />

      <Form.Item>
        <Space>
          <Button
            type="primary"
            htmlType="submit"
            icon={<PlayCircleOutlined />}
            loading={running}
          >
            Resume Processing
          </Button>
          {running && (
            <Button danger onClick={handleCancel}>Cancel</Button>
          )}
        </Space>
      </Form.Item>

      {started && (
        <>
          <Divider />
          <PipelineProgress steps={RESUME_STEPS} stepStates={stepStates} error={error} />
        </>
      )}
      {doneData && <div style={{ marginTop: 16 }}><DoneResult data={doneData} /></div>}
    </Form>
  );
}

export default function PipelinePage() {
  return (
    <Tabs
      items={[
        { key: "run", label: "Upload & Transcribe", children: <RunTab /> },
        { key: "resume", label: "Resume Processing", children: <ResumeTab /> },
      ]}
    />
  );
}
