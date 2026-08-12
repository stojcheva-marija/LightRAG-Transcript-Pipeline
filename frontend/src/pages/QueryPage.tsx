import { useState, useRef, useEffect } from "react";
import {
  Button,
  Input,
  Spin,
  Typography,
  Space,
  Alert,
  Empty,
  Divider,
  Modal,
} from "antd";
import { SendOutlined, UserOutlined, RobotOutlined, SoundFilled } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import { queryRAG, HistoryMessage, SourceStem } from "../api";

const { Text } = Typography;

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: SourceStem[];
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function AudioModal({ source, index, onClose }: { source: SourceStem; index: number; onClose: () => void }) {
  const audioRef = useRef<HTMLAudioElement>(null);

  function handleLoaded() {
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = source.start_seconds;
    el.play().catch(() => {});
  }

  return (
    <Modal
      open
      onCancel={onClose}
      footer={null}
      title={
        <Space>
          <SoundFilled style={{ color: "#1677ff" }} />
          <Text strong>Source {index}</Text>
          <Text type="secondary" style={{ fontSize: 13 }}>
            starts at {formatTime(source.start_seconds)}
          </Text>
        </Space>
      }
      width={480}
      centered
    >
      <audio
        ref={audioRef}
        src={source.audio_url!}
        controls
        onLoadedMetadata={handleLoaded}
        style={{ width: "100%", marginTop: 12 }}
      />
    </Modal>
  );
}

export default function QueryPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSource, setActiveSource] = useState<{ source: SourceStem; index: number } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend() {
    const question = input.trim();
    if (!question) return;

    const history: HistoryMessage[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await queryRAG(question, history);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, sources: res.source_stems },
      ]);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "16px 0",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {messages.length === 0 && (
          <Empty description="Start a conversation" style={{ margin: "auto" }} />
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: msg.role === "user" ? "flex-end" : "flex-start",
              gap: 4,
            }}
          >
            <Space size={6} style={{ marginBottom: 4 }}>
              {msg.role === "user" ? (
                <UserOutlined style={{ color: "#1677ff" }} />
              ) : (
                <RobotOutlined style={{ color: "#52c41a" }} />
              )}
              <Text type="secondary" style={{ fontSize: 12 }}>
                {msg.role === "user" ? "You" : "Assistant"}
              </Text>
            </Space>

            <div
              style={{
                maxWidth: "80%",
                background: msg.role === "user" ? "#1677ff" : "#f5f5f5",
                color: msg.role === "user" ? "#fff" : "inherit",
                borderRadius: 12,
                padding: "10px 14px",
              }}
            >
              {msg.role === "user" ? (
                <span style={{ color: "#fff", whiteSpace: "pre-wrap" }}>{msg.content}</span>
              ) : (
                <div className="markdown-body">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              )}
            </div>

            {msg.sources && msg.sources.filter((s) => s.audio_url).length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 2 }}>
                <Text type="secondary" style={{ fontSize: 12, alignSelf: "center" }}>
                  Listen to source:
                </Text>
                {msg.sources
                  .filter((s) => s.audio_url)
                  .map((s, si) => (
                    <Button
                      key={si}
                      size="small"
                      icon={<SoundFilled />}
                      onClick={() => setActiveSource({ source: s, index: si + 1 })}
                      style={{ borderRadius: 20 }}
                      title={s.stem}
                    >
                      {formatTime(s.start_seconds)}
                    </Button>
                  ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <RobotOutlined style={{ color: "#52c41a" }} />
            <Spin size="small" />
            <Text type="secondary" style={{ fontSize: 12 }}>Thinking...</Text>
          </div>
        )}

        {error && (
          <Alert type="error" description={error} closable={{ onClose: () => setError(null) }} />
        )}

        <div ref={bottomRef} />
      </div>

      <Divider style={{ margin: "8px 0" }} />

      <Space.Compact style={{ width: "100%" }}>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onPressEnter={handleSend}
          placeholder="Type your question here..."
          disabled={loading}
          size="large"
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleSend}
          disabled={loading || !input.trim()}
          size="large"
        >
          Send
        </Button>
      </Space.Compact>

      {activeSource && (
        <AudioModal source={activeSource.source} index={activeSource.index} onClose={() => setActiveSource(null)} />
      )}
    </div>
  );
}
