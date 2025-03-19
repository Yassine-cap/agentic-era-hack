**Description & Motivation** 

The following project is designed to address a critical challenge in today's rapidly evolving job market—how businesses can effectively integrate AI into their workforce while ensuring employees remain skilled and adaptable. 
The motivation behind this project stems from the growing gap between AI adoption and workforce readiness. While 97% of executives acknowledge the transformative potential of generative AI, only 5% of companies are reskilling their employees at scale. This gap poses a significant risk, as organizations may struggle to harness AI’s full potential due to workforce skill mismatches.
The idea behind this project is to develop an AI-driven agent that helps businesses simulate workforce transformations, optimize reskilling strategies, and guide leadership in making data-driven talent decisions. A key focus is on providing recommendations for skill-related training opportunities, ensuring employees receive targeted upskilling paths aligned with industry needs. By leveraging real-time labor market intelligence, advanced analytics, and interactive dashboards, this project empowers companies to stay ahead of market shifts and ensure a smooth transition into an AI-augmented workplace. 
The project incorporates technologies such as Gemini 2.0 Flash, Vertex AI, and LangGraph, ensuring a robust and scalable approach to workforce development. Ultimately, the motivation is to reduce skill gaps, enhance job transition planning, and maximize productivity through strategic AI adoption, making businesses and employees future-ready.

Onepoint Team of Hackers:

> Charlotte Gosset-Grainville

> Yassine Boukhari

> Zainab Berrada



A workflow based on agents following LangGraph framework and many other tools to build an extensive and dynamic HR reporting working as HR advisor tools (following the recent advancement made on deep search techniques) 

Agent generated with [`googleCloudPlatform/agent-starter-pack`](https://github.com/GoogleCloudPlatform/agent-starter-pack)

## Project Structure

This project is organized as follows:

```
hack_agent_genai_vertex/
├── app/                 # Core application code
│   ├── agent.py         # Main agent logic
│   ├── agent_engine_app.py # Agent Engine application logic
│   └── utils/           # Utility functions and helpers
├── deployment/          # Infrastructure and deployment scripts
├── notebooks/           # Jupyter notebooks for prototyping and evaluation
├── tests/               # Unit, integration, and load tests
├── Makefile             # Makefile for common commands
└── pyproject.toml       # Project dependencies and configuration
```

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager - [Install](https://docs.astral.sh/uv/getting-started/installation/)
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)
- **Terraform**: For infrastructure deployment - [Install](https://developer.hashicorp.com/terraform/downloads)
- **make**: Build automation tool - [Install](https://www.gnu.org/software/make/) (pre-installed on most Unix-based systems)


### Installation

Install required packages using uv:

```bash
make install
```

### Setup

If not done during the initialization, set your default Google Cloud project and Location:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export LOCATION="us-central1"
gcloud config set project $PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT_ID
```

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `make install`       | Install all required dependencies using uv                                                  |
| `make playground`    | Launch Streamlit interface for testing agent locally and remotely |
| `make backend`       | Deploy agent to Agent Engine service |
| `make test`          | Run unit and integration tests                                                              |
| `make lint`          | Run code quality checks (codespell, ruff, mypy)                                             |
| `uv run jupyter lab` | Launch Jupyter notebook                                                                     |

For full command options and usage, refer to the [Makefile](Makefile).


## Usage

1. **Prototype:** Build your Generative AI Agent using the intro notebooks in `notebooks/` for guidance. Use Vertex AI Evaluation to assess performance.
2. **Integrate:** Import your chain into the app by editing `app/agent.py`.
3. **Test:** Explore your chain's functionality using the Streamlit playground with `make playground`. The playground offers features like chat history, user feedback, and various input types, and automatically reloads your agent on code changes.
4. **Deploy:** Configure and trigger the CI/CD pipelines, editing tests if needed. See the [deployment section](#deployment) for details.
5. **Monitor:** Track performance and gather insights using Cloud Logging, Tracing, and the Looker Studio dashboard to iterate on your application.


## Deployment

### Dev Environment
You can test deployment towards a Dev Environment using the following command:

```bash
gcloud config set project <your-dev-project-id>
make backend
```

The repository includes a Terraform configuration for the setup of the Dev Google Cloud project.
See [deployment/README.md](deployment/README.md) for instructions.

### Production Deployment

The repository includes a Terraform configuration for the setup of a production Google Cloud project. Refer to [deployment/README.md](deployment/README.md) for detailed instructions on how to deploy the infrastructure and application.

## Monitoring and Observability

>> You can use [this Looker Studio dashboard](https://lookerstudio.google.com/c/reporting/fa742264-4b4b-4c56-81e6-a667dd0f853f/page/tEnnC) template for visualizing events being logged in BigQuery. See the "Setup Instructions" tab to getting started.

The application uses OpenTelemetry for comprehensive observability with all events being sent to Google Cloud Trace and Logging for monitoring and to BigQuery for long term storage. 
