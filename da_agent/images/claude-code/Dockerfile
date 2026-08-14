FROM python:3.10-bullseye

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code@2.1.220

# Upgrade pip and install Python packages
RUN pip install --upgrade pip
RUN pip install \
    huggingface-hub requests rich python-dotenv \
    pillow jinja2 pyyaml typeguard \
    rouge-score beautifulsoup4 \
    wikipedia-api youtube-transcript-api \
    PyPDF2 litellm anthropic ollama together \
    datasets \
    openpyxl scipy scikit-learn geopandas geopy \
    matplotlib statsmodels pytz python-dateutil \
    skyfield pyatmos cdflib astropy timezonefinder

# Create non-root user with same UID as host to avoid permission issues
ARG AGENT_USER=agent
ARG AGENT_UID=1000

RUN apt-get update && apt-get install -y sudo && \
    useradd -m -u ${AGENT_UID} -s /bin/bash ${AGENT_USER} && \
    echo "${AGENT_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/${AGENT_USER}/.claude && \
    chown -R ${AGENT_USER}:${AGENT_USER} /workspace /home/${AGENT_USER} && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${AGENT_USER}

# Set the default command to run when starting the container
CMD ["/bin/bash"]