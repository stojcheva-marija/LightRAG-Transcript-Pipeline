import { useState } from "react";
import { Layout, Menu, Typography, ConfigProvider } from "antd";
import {
  MessageOutlined,
  CloudUploadOutlined,
} from "@ant-design/icons";
import QueryPage from "./pages/QueryPage";
import PipelinePage from "./pages/PipelinePage";

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

const PAGES: Record<string, React.ReactNode> = {
  query: <QueryPage />,
  pipeline: <PipelinePage />,
};

export default function App() {
  const [page, setPage] = useState("query");

  return (
    <ConfigProvider theme={{ token: { colorPrimary: "#1677ff" } }}>
    <Layout style={{ minHeight: "100vh" }}>
      <Sider theme="dark" width={220} breakpoint="lg" collapsedWidth={60}>
        <div style={{ padding: "20px 16px 12px", color: "#fff" }}>
          <Title level={5} style={{ color: "#fff", margin: 0 }}>
            Transcript Intelligence
          </Title>
          <div style={{ color: "#aaa", fontSize: 12 }}>Chat with your audio</div>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[page]}
          onClick={({ key }) => setPage(key)}
          items={[
            { key: "pipeline", icon: <CloudUploadOutlined />, label: "Upload Audio" },
            { key: "query", icon: <MessageOutlined />, label: "Chat" },
          ]}
        />
      </Sider>

      <Layout>
        <Header
          style={{
            background: "#fff",
            borderBottom: "1px solid #f0f0f0",
            padding: "0 24px",
            display: "flex",
            alignItems: "center",
          }}
        >
          <Title level={4} style={{ margin: 0 }}>
            {page === "query" && "Chat"}
            {page === "pipeline" && "Upload Audio"}
          </Title>
        </Header>

        <Content
          style={{
            padding: 24,
            display: "flex",
            flexDirection: "column",
            height: "calc(100vh - 64px)",
            overflow: "auto",
          }}
        >
          {PAGES[page]}
        </Content>
      </Layout>
    </Layout>
    </ConfigProvider>
  );
}
