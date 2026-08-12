import { useEffect, useState } from "react";
import {
  List,
  Button,
  Spin,
  Alert,
  Typography,
  Space,
  Empty,
  Tag,
} from "antd";
import { SoundOutlined, ReloadOutlined, PlayCircleOutlined } from "@ant-design/icons";
import { listTranscripts, getAudioUrl } from "../api";

const { Text } = Typography;

export default function TranscriptsPage() {
  const [stems, setStems] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioLoading, setAudioLoading] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setStems(await listTranscripts());
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function playAudio(stem: string) {
    setAudioLoading(stem);
    try {
      const url = await getAudioUrl(stem);
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.click();
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setAudioLoading(null);
    }
  }

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Text strong>Transcripts in storage</Text>
        <Button
          icon={<ReloadOutlined />}
          onClick={load}
          loading={loading}
          size="small"
        >
          Refresh
        </Button>
      </Space>

      {error && (
        <Alert
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
        />
      )}

      {loading ? (
        <div style={{ textAlign: "center", padding: 40 }}>
          <Spin />
        </div>
      ) : stems.length === 0 ? (
        <Empty description="No transcripts found" />
      ) : (
        <List
          bordered
          dataSource={stems}
          renderItem={(stem) => (
            <List.Item
              actions={[
                <Button
                  key="play"
                  type="link"
                  icon={<PlayCircleOutlined />}
                  loading={audioLoading === stem}
                  onClick={() => playAudio(stem)}
                >
                  Play audio
                </Button>,
              ]}
            >
              <Space>
                <Tag icon={<SoundOutlined />} color="blue">
                  {stem}
                </Tag>
              </Space>
            </List.Item>
          )}
        />
      )}
    </div>
  );
}
