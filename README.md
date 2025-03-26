# Als - 多智能体协作系统

一个基于大语言模型的协作框架，通过定义不同角色的AI智能体并建立它们之间的通信机制，实现复杂任务的自动分解与协作完成。该系统模拟人类团队协作模式，各智能体专注于自身专业领域，通过信息交换和任务流转实现整体目标。

## 项目特点

- 🤖 基于前沿的智能体技术（AutoGen/LangGraph框架）
- 🔄 可扩展的角色定义和协作机制
- 👥 人机协作界面，支持关键节点人工干预
- 🔧 工作流自动化和任务编排

## 技术栈

- **后端框架**: FastAPI
- **前端**: Vue.js/React
- **AI引擎**: LangChain/AutoGen
- **数据存储**: 向量数据库 (Chroma/FAISS)
- **消息队列**: RabbitMQ/Redis
- **部署**: Docker, Kubernetes

## 核心功能

- **多智能体角色定义**: 产品经理、设计师、开发者、测试员等专业角色
- **团队协作流程**: 模拟产品开发全流程的智能协作
- **状态监控与可视化**: 实时查看各智能体工作状态和进展
- **人机协作接口**: 允许人类参与关键决策环节
- **工作记忆管理**: 智能体长期记忆和工作记忆管理

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 16+
- Docker (可选，用于容器化部署)

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/Als.git
cd Als
```

2. 创建虚拟环境并安装依赖
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. 设置配置文件
```bash
cp .env.example .env
# 编辑.env文件，添加必要的API密钥和配置
```

4. 启动后端服务
```bash
cd backend
uvicorn main:app --reload
```

5. 启动前端服务
```bash
cd frontend
npm install
npm run dev
```

## 项目结构

```
Als/
├── backend/              # 后端代码
│   ├── agents/           # 智能体定义
│   ├── api/              # API端点
│   ├── core/             # 核心逻辑
│   └── utils/            # 工具函数
├── frontend/             # 前端代码
├── data/                 # 数据文件
├── tests/                # 测试代码
├── docs/                 # 文档
├── examples/             # 示例配置和场景
└── scripts/              # 辅助脚本
```

## 开发计划

详见[项目行动方案](多智能体协作系统项目行动方案.md)文档。

## 贡献指南

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目维护者: [Your Name](mailto:your.email@example.com)
- 项目主页: [GitHub Repository](https://github.com/yourusername/Als)
