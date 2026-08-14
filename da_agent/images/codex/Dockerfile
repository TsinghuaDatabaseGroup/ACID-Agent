FROM python:3.10-bullseye

WORKDIR /workspace

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

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @openai/codex@latest

RUN pip install --upgrade pip
RUN pip install google-cloud-bigquery db-dtypes pandas matplotlib scikit-learn seaborn numpy scipy statsmodels xgboost lightgbm plotly tabulate snowflake-connector-python duckdb openpyxl pyyaml geopandas

CMD ["/bin/bash"]
